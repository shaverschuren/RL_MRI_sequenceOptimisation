"""EPGX Simulation module

This module performs basic EPG simulations in k-space.
It is based upon some MATLAB code, as was written
and documented at 'https://github.com/mriphysics/EPG-X' by
Shaihan Malik in 06/2017, was reimplemented in Python
by Luuk Jacobs in 01/2022 and then partly repurposed and adjusted
by Sjors Verschuren in 05/2022.

TODO: Needs to be fully reviewed and then repurposed still
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
import time


class EPG_X(torch.nn.Module):
    @staticmethod
    def SuperLorentzian(T2b):
        n = 512
        ff = np.linspace(-35e3, 35e3, n)

        G = np.zeros((1, n))

        u = np.linspace(0, 1, 500)
        du = u[1] - u[0]
        for ii in range(n):
            g = np.sqrt(2 / math.pi) * T2b / abs(3 * pow(u, 2) - 1)
            g *= pow(math.e, (-2 * pow((2 * math.pi * ff[ii] * T2b / (3 * pow(u, 2) - 1)), 2)))
            gg = du * np.sum(g)
            G[0, ii] = gg

        po = [i for i, x in enumerate(ff) if abs(x) < 1.5e3]  # points to interpolate
        pu = [i for i, x in enumerate(ff) if (abs(x) > 1.5e3) and (abs(x) < 2e3)]  # points to use

        cs = CubicSpline(ff[pu], G[0, pu])
        Gi = cs(ff[po])
        G[0, po] = Gi

        return ff, G * 1e6

    @staticmethod
    def EPGX_MT_shift_matrices(device, Nmax):
        N = (Nmax + 1) * 4
        S = torch.zeros((N * N, 1), dtype=torch.float16, device=device)

        kidx = list(range(4, N, 4))  # miss out F1+
        sidx = [x - 4 for x in kidx]  # negative states shift 4 back
        idx = [a + b for a, b in zip(kidx, [N * x for x in sidx])]  # linear indices of S(kidx, sidx)
        S[idx] = 1

        kidx = list(range(5, N, 4))
        kidx.pop()  # most negative state is nothing
        sidx = [x + 4 for x in kidx]  # Positive states shift 4 forward
        ix = [a + b for a, b in zip(kidx, [N * x for x in sidx])]  # linear indices of S(kidx, sidx)
        S[ix] = 1

        kidx = list(range(2, N, 4))  # Za states
        ix = [a + b for a, b in zip(kidx, [N * x for x in kidx])]  # linear indices of S(kidx, sidx)
        S[ix] = 1

        kidx = list(range(3, N, 4))  # Zb states
        ix = [a + b for a, b in zip(kidx, [N * x for x in kidx])]  # linear indices of S(kidx, sidx)
        S[ix] = 1

        S = torch.t(torch.reshape(S, (N, N)))
        S[0:2, 5] = 1  # F0+ sates

        return S

    @staticmethod
    def kron(a, b):
        """yulkang/kron.py"""
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    @staticmethod
    def RF_rot_ex(device, alpha, WT):
        """Along the y direction (phase of pi/2)"""
        cosa2 = torch.cos(alpha / 2.) ** 2
        sina2 = torch.sin(alpha / 2.) ** 2

        cosa = torch.cos(alpha)
        sina = torch.sin(alpha)
        RR = torch.tensor([[cosa2, -sina2, sina, 0],
                           [-sina2, cosa2, sina, 0],
                           [-0.5 * sina, -0.5 * sina, cosa, 0],
                           [0, 0, 0, torch.exp(torch.as_tensor(-WT))]], dtype=torch.float16, device=device)
        return RR

    @staticmethod
    def RF_rot_refoc(device, alpha, WT):
        """Along the x direction (phase of 0)"""
        cosa2 = torch.cos(alpha / 2.) ** 2
        sina2 = torch.sin(alpha / 2.) ** 2

        cosa = torch.cos(alpha)
        sina = torch.sin(alpha)
        RR = torch.tensor([[cosa2, sina2, sina, 0],
                           [sina2, cosa2, -sina, 0],
                           [-0.5 * sina, 0.5 * sina, cosa, 0],
                           [0, 0, 0, torch.exp(torch.as_tensor(-WT))]], dtype=torch.float16, device=device)
        return RR

    def EPGX_TSE_MT(self, device, theta, B1SqrdTau, ESP, R1a, R2a, f, ka, kb, G, z0, TI):
        """
        :param device: GPU device used for training
        :param theta: flip angle train (radians)
        :param B1SqrdTau: integrated square amplitude of each RF pulse (uT² ms)
        :param ESP: echo spacing (ms)
        :param R1a: 1/T1 of free pool (ms⁻¹)
        :param R2a: 1/T2 of free pool (ms⁻¹)
        :param f: fraction of bound pool (-)
        :param ka: exchange rate free to bound pool (ms⁻¹)
        :param kb: exchange rate bound to free pool (ms⁻¹)
        :param G: absorption line value at frequency of interest (us)
        :param z0: initial magnetization (-)
        :param TI: inversion time (ms)
        """

        np2 = len(theta)
        kmax = 2 * (np2 - 1)  # maximum EPG order (can be lowered to save memory and time)

        kmax_per_pulse = np.array(
            [2 * x + 1 for x in list(range(1, math.ceil(np2 / 2) + 1)) + list(range(math.floor(np2 / 2), 0, -1))])
        kmax_per_pulse[kmax_per_pulse > kmax] = kmax
        if max(kmax_per_pulse) < kmax:
            kmax = max(kmax_per_pulse)
        N = 4 * (kmax + 1)  # number of states

        im_dim = R1a.shape[0]  # number of voxels we simulate simultaneously

        S = self.EPGX_MT_shift_matrices(device, kmax)  # shift matrix

        Lambda_T = torch.diag_embed(-R2a.repeat(1, 2)).float()
        Xi_T = torch.matrix_exp(0.5 * ESP * Lambda_T).to(dtype=torch.float16)  # relaxation-exchange matrix for transverse components

        Lambda_L = torch.reshape(torch.stack((-R1a - ka.clone(), kb, ka.clone(), -R1a - kb), 2), (-1, 2, 2)).float()
        Xi_L = torch.matrix_exp(0.5 * ESP * Lambda_L).to(dtype=torch.float16)  # relaxation-exchange matrix for longitudinal components

        C_L = torch.unsqueeze(torch.cat(((1 - f) * R1a, f * R1a), 1), 2).float()
        Zoff1 = (Xi_L - torch.reshape(torch.eye(2, device=device), (1, 2, 2)))
        Zoff2 = torch.linalg.solve(Lambda_L, C_L)
        Zoff = torch.bmm(Zoff1, Zoff2)

        Xi = torch.cat((torch.cat((Xi_T, Xi_T * 0), 1), torch.cat((Xi_L * 0, Xi_L), 1)), 2)

        b = torch.zeros((im_dim, N, 1), dtype=torch.float16, device=device)
        b[:, 2:4, :] = Zoff

        Xi = self.kron(torch.unsqueeze(torch.eye(kmax + 1, dtype=torch.float16, device=device), 0), Xi)  # no diffusion

        XS = torch.matmul(S, Xi)

        G *= 1e-3  # us to ms
        gam = 267.5221e-3  # rad/ms/uT
        WT = [math.pi * gam * gam * x * G for x in B1SqrdTau]  # for bound pool saturation

        F = torch.zeros((im_dim, N, np2 - 1), dtype=torch.float16, device=device)
        FF = torch.zeros((im_dim, N, 1), dtype=torch.float16, device=device)

        FF[:, 2] = z0[:, 0]  # initial states
        FF[:, 3] = z0[:, 1]

        if TI > 0:
            WT180 = math.pi * gam * gam * 213.1 * G  # 180 degrees inversion pulse

            R = torch.zeros((1, 2, 2), dtype=torch.float16, device=device)
            R[:, 0, 0] = -1  # 180 degrees inversion pulse
            R[:, 1, 1] = torch.exp(torch.as_tensor(-WT180))
            FF[:, 2:4] = torch.matmul(R, FF[:, 2:4])

            Xi_L_prep = torch.matrix_exp(TI[0] * Lambda_L).to(dtype=torch.float16)
            Zoff_prep1 = Xi_L_prep - torch.eye(2, dtype=torch.float16, device=device)
            Zoff_prep2 = torch.linalg.solve(Lambda_L, C_L).to(dtype=torch.float16)
            Zoff_prep = torch.bmm(Zoff_prep1, Zoff_prep2)
            FF[:, 2:4] = torch.bmm(Xi_L_prep, FF[:, 2:4].clone()) + Zoff_prep

        A = self.RF_rot_ex(device, torch.as_tensor(abs(theta[0])), WT[0])  # excitation pulse (phase pi/2)
        FF[:, :4] = torch.matmul(A, FF[:, :4])
        FF[:, :8] = torch.matmul(XS[:, :8, :8], FF[:, :8].clone()) + b[:, :8]

        A = self.RF_rot_refoc(device, torch.as_tensor(abs(theta[1])), WT[1])  # refocussing pulse (phase 0)
        T = torch.zeros((1, N*N, 1), dtype=torch.float16, device=device)
        i1 = [y + ii for y in [x * N for x in range(4)] for ii in range(4)]
        ksft = 4 * (4 * (kmax + 1) + 1)
        for i2 in range(16):
            ind_lin = list(range(i1[i2], N * N + 1, ksft))
            T[:, ind_lin] = A.reshape(1, -1, 1, 1)[:, i2, 0]
        T = T.reshape(1, N, N)

        for jj in range(1, np2):
            kidx = range(4 * kmax_per_pulse[jj])

            FF[:, kidx] = torch.matmul(T[:, :kidx[-1]+1, :kidx[-1]+1], FF[:, kidx])
            F[:, kidx, jj - 1] = torch.squeeze(torch.bmm(XS[:, :kidx[-1]+1, :kidx[-1]+1], FF[:, kidx]) + b[:, kidx])
            if jj == np2-1:
                break
            FF[:, kidx] = torch.bmm(XS[:, :kidx[-1]+1, :kidx[-1]+1], torch.unsqueeze(F[:, :kidx[-1]+1, jj - 1].clone(), 2)) + b[:, kidx]

        idx = list(range(5, F.shape[1], 4))[::-1] + [0] + list(range(4, F.shape[1], 4))
        idx.pop(0)
        idx.pop(0)

        Fn = F[:, idx, :]

        ZnA = F[:, 2::4, :]
        ZnB = F[:, 3::4, :]
        Zn = torch.stack((ZnA, ZnB), 3)

        return F[:, 0, :], Fn, Zn

    def forward(self, device, nrefocus, nslice, soi, packages, ESP, TE, TR, TI, fake_param, test=False):
        """
        :param device: GPU device used for training
        :param nrefocus: number of refocusing pulses
        :param nslice: number of total slices
        :param soi: slice of interest
        :param packages: number of packages
        :param ESP: echo spacing (ms)
        :param TE: echo time (ms)
        :param TR: repetition time (ms)
        :param TI: inversion time (ms)
        :param fake_param: synthesized q*-maps
        :param test: used for testing or not (bool)
        """
        if test:  # fix tissue parameters with predefined values for testing
            PD = torch.ones((300, 1), dtype=torch.float16, device=device) * 0.6
            R1f = torch.ones((300, 1), dtype=torch.float16, device=device) * (1 / 779)
            R2f = torch.ones((300, 1), dtype=torch.float16, device=device) * (1 / 45)
            f = torch.ones((300, 1), dtype=torch.float16, device=device) * 0.1166
            kf = torch.ones((300, 1), dtype=torch.float16, device=device) * 4.3e-3
        else:
            PD = fake_param[0, 0, :, :]
            R1f = 1 / (fake_param[0, 1, :, :] * 5000 + 1e-2)
            R2f = 1 / (fake_param[0, 2, :, :] * 1500 + 1e-2)
            f = fake_param[0, 3, :, :] + 1e-3
            kf = fake_param[0, 4, :, :] / 100

        if nrefocus == 1:  # T1w
            a0 = [70 * math.pi / 180] + [math.pi]  # FA train
            b1sqrdtau = [30] + [213.1]  # integrated square amplitude of each RF pulse (uT² ms)
            Ntr = 5  # number of TR repetitions (to ensure equilibrium)
            Gss = 10.9  # slice-selective gradient [mT/m]
        else:  # PDw, T2w, FLAIR
            a0 = [math.pi / 2] + [120 * math.pi / 180] * nrefocus  # FA train
            b1sqrdtau = [32.7] + [106.5] * nrefocus  # integrated square amplitude of each RF pulse (uT² ms)
            Ntr = 2  # number of TR repetitions (to ensure equilibrium)
            Gss = 12.27  # slice-selective gradient [mT/m]

        # division into packages increases the slice distance
        df = Gss * 42.57e3 * 4.5e-3 * packages  # slice spacing in Hz (G * gamma * dx)
        # dx is defined from center to center

        T2b = 12e-6
        ff, G = self.SuperLorentzian(T2b)

        nslice = int(nslice/packages)  # divide slices into packages
        while soi >= nslice:
            soi -= nslice  # treat packages independently
        slice_order = list(range(0, nslice, 2)) + list(range(1, nslice, 2))  # odd then even

        fs = [df * x for x in range(int(-1 * np.floor(nslice / 2)), int(np.floor(nslice / 2)) + 1)]
        li = interp1d(ff, G)  # G is function of FF (described by super Lorenzian)
        GG = li(fs)

        Ntot = Ntr * len(slice_order)
        slice_order = np.tile(slice_order, (1, Ntr))

        Tdelay = TR/nslice - ESP * (nrefocus + 0.5)

        kb = kf * (1 - f) / f  # Exchange rate from free to bound (ms⁻¹)
        # Assumption: R1b = R1f --> replaced directly throughout code to save memory
        z0 = torch.stack((1 - f, f), 1)  # initial magnetization

        L = torch.reshape(torch.stack((-R1f - kf, kb, kf, -R1f - kb), 2), (-1, 2, 2)).float()
        C = torch.reshape(torch.stack((R1f * (1 - f), R1f * f), 2), (-1, 2, 1)).float()
        Xi = torch.matrix_exp(L * Tdelay).to(dtype=torch.float16)
        Xi_min = Xi - torch.reshape(torch.eye(2, device=device), (1, 2, 2))
        Zoff = torch.bmm(torch.bmm(Xi_min, torch.inverse(L)), C).to(dtype=torch.float16)
        for ii in range(Ntot):
            if slice_order[0, ii] == soi:  # final TR and soi
                s, _, Zn = self.EPGX_TSE_MT(device, a0, b1sqrdtau, ESP, R1f, R2f, f, kf, kb, GG[0, soi], z0, TI)
            else:
                _, _, Zn = self.EPGX_TSE_MT(device, [0] * len(a0), b1sqrdtau, ESP, R1f, R2f, f, kf, kb, GG[0, slice_order[0, ii]], z0, TI)

            z0 = Zn[:, 0, -1, :]
            z0 = torch.bmm(Xi, torch.unsqueeze(z0, 2)) + Zoff

        # contrast is determined by echo that fills central k-space (~effective echo time)
        return torch.abs(s[:, math.ceil(TE / ESP) - 1]) * PD[:, 0]


# use for testing
if __name__ == "__main__":
    EPGX_model = EPG_X()
    for i in range(10):  # evaluate multiple times
        now = time.time()
        fake = EPGX_model.forward(torch.device('cpu'), 15, 30, 15, 2, 10, 80, 3254, 0, 0, True)
        print(time.time() - now, fake[0])

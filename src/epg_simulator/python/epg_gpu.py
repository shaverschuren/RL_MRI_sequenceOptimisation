"""EPGX Simulation module

This module performs EPG simulations on the GPU.
It is based upon some MATLAB code, as was written
and documented at 'https://github.com/mriphysics/EPG-X' by
Shaihan Malik in 06/2017, was reimplemented in Python
by Luuk Jacobs in 01/2022 and then partly repurposed and adjusted
by Sjors Verschuren in 05/2022.

TODO: Needs to be fully reviewed and then repurposed still
"""

import math
import numpy as np
import torch
import time


class EPG(torch.nn.Module):
    @staticmethod
    def EPG_shift_matrices(device, Nmax):
        N = (Nmax + 1) * 3
        S = torch.zeros((N * N, 1), dtype=torch.float16, device=device)

        kidx = list(range(3, N, 3))  # miss out F1+
        sidx = [x - 3 for x in kidx]  # negative states shift 4 back
        idx = [a + b for a, b in zip(kidx, [N * x for x in sidx])]  # linear indices of S(kidx, sidx)
        S[idx] = 1

        kidx = list(range(4, N, 3))
        kidx.pop()  # most negative state is nothing
        sidx = [x + 3 for x in kidx]  # Positive states shift 4 forward
        ix = [a + b for a, b in zip(kidx, [N * x for x in sidx])]  # linear indices of S(kidx, sidx)
        S[ix] = 1

        kidx = list(range(2, N, 3))  # Za states
        ix = [a + b for a, b in zip(kidx, [N * x for x in kidx])]  # linear indices of S(kidx, sidx)
        S[ix] = 1

        S = torch.t(torch.reshape(S, (N, N)))
        S[0:2, 4] = 1  # F0+ sates

        return S

    @staticmethod
    def kron(a, b):
        """yulkang/kron.py"""
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    @staticmethod
    def RF_rot_ex(device, alpha):
        """Along the y direction (phase of pi/2)"""
        cosa2 = torch.cos(alpha / 2.) ** 2
        sina2 = torch.sin(alpha / 2.) ** 2

        cosa = torch.cos(alpha)
        sina = torch.sin(alpha)
        RR = torch.tensor([[cosa2, -sina2, sina],
                           [-sina2, cosa2, sina],
                           [-0.5 * sina, -0.5 * sina, cosa]], dtype=torch.float16, device=device)
        return RR

    @staticmethod
    def RF_rot_refoc(device, alpha):
        """Along the x direction (phase of 0)"""
        cosa2 = torch.cos(alpha / 2.) ** 2
        sina2 = torch.sin(alpha / 2.) ** 2

        cosa = torch.cos(alpha)
        sina = torch.sin(alpha)
        RR = torch.tensor([[cosa2, sina2, sina],
                           [sina2, cosa2, -sina],
                           [-0.5 * sina, 0.5 * sina, cosa]], dtype=torch.float16, device=device)
        return RR

    def EPG_TSE(self, device, theta, ESP, T1, T2, z0, inversion, TI):
        """
        :param device: GPU device used for training
        :param theta: flip angle train (radians)
        :param ESP: echo spacing (ms)
        :param T1: longitudinal relaxation rate (ms)
        :param T2: transverse relaxation rate (ms)
        :param z0: initial magnetization (-)
        :param inversion: whether inversion pulse if applied or not (bool)
        :param TI: inversion time (ms)
        """
        np2 = len(theta)
        kmax = 2 * (np2 - 1)  # maximum EPG order (can be lowered to save memory and time)

        kmax_per_pulse = np.array(
            [2 * x + 1 for x in list(range(1, math.ceil(np2 / 2) + 1)) + list(range(math.floor(np2 / 2), 0, -1))])
        kmax_per_pulse[kmax_per_pulse > kmax] = kmax
        if max(kmax_per_pulse) < kmax:
            kmax = max(kmax_per_pulse)
        N = 3 * (kmax + 1)  # number of states

        im_dim = T1.shape[0]  # number of voxels we simulate simultaneously

        S = self.EPG_shift_matrices(device, kmax)  # shift matrix

        E1 = torch.exp(-0.5 * ESP / T1)
        E2 = torch.exp(-0.5 * ESP / T2)
        E = torch.diag_embed(torch.stack([E2, E2, E1], dim=2))[:, 0, :, :]

        b = torch.zeros((im_dim, N, 1), dtype=torch.float16, device=device)
        b[:, 2, :] = 1 - E1

        E = self.kron(torch.unsqueeze(torch.eye(kmax + 1, dtype=torch.float16, device=device), 0), E)  # no diffusion

        ES = torch.matmul(E, S)

        F = torch.zeros((im_dim, N, np2 - 1), dtype=torch.float16, device=device)
        FF = torch.zeros((im_dim, N, 1), dtype=torch.float16, device=device)

        FF[:, 2] = z0  # initial state

        if inversion:
            FF[:, 2] = -FF[:, 2]  # 180 degrees inversion pulse
            FF[:, 2] = torch.exp(-TI/T1) * FF[:, 2].clone() + 1 - torch.exp(-TI/T1)  # recovery

        A = self.RF_rot_ex(device, torch.as_tensor(abs(theta[0])))  # excitation pulse
        FF[:, :3] = torch.matmul(A, FF[:, :3])
        FF[:, :6] = torch.matmul(ES[:, :6, :6], FF[:, :6].clone()) + b[:, :6]

        A = self.RF_rot_refoc(device, torch.as_tensor(abs(theta[1])))  # refocussing pulses (contstant)
        T = torch.zeros((1, N*N, 1), dtype=torch.float16, device=device)
        i1 = [y + ii for y in [x * N for x in range(3)] for ii in range(3)]
        ksft = 3 * (3 * (kmax + 1) + 1)
        for i2 in range(9):
            ind_lin = list(range(i1[i2], N * N + 1, ksft))
            T[:, ind_lin] = A.reshape(1, -1, 1, 1)[:, i2, 0]
        T = T.reshape(1, N, N)

        for jj in range(1, np2):
            kidx = range(3 * kmax_per_pulse[jj])

            FF[:, kidx] = torch.matmul(T[:, :kidx[-1]+1, :kidx[-1]+1], FF[:, kidx])
            F[:, kidx, jj - 1] = torch.squeeze(torch.bmm(ES[:, :kidx[-1]+1, :kidx[-1]+1], FF[:, kidx]) + b[:, kidx])
            if jj == np2 - 1:
                break
            FF[:, kidx] = torch.bmm(ES[:, :kidx[-1]+1, :kidx[-1]+1], torch.unsqueeze(F[:, :kidx[-1]+1, jj - 1].clone(), 2)) + b[:, kidx]

        idx = list(range(4, F.shape[1], 3))[::-1] + [0] + list(range(3, F.shape[1], 3))
        idx.pop(0)
        idx.pop(0)

        Fn = F[:, idx, :]
        Zn = F[:, 2::3, :]

        return F[:, 0, :], Fn, Zn

    def EPG_GRE(self):
        """Function for GPU-accelerated single-compartment Gradient Echo EPG

        TODO: Write this whole thing... MATLAB counterpart (not GPU optimized):
        https://github.com/mriphysics/EPG-X/blob/master/EPGX-src/EPG_GRE.m
        """

    def forward(self, device, nrefocus, ESP, TE, TR, inversion, TI, fake_param, test=False):
        """
        :param device: GPU device used for training
        :param nrefocus: number of refocusing pulses
        :param ESP: echo spacing (ms)
        :param TE: echo time (ms)
        :param TR: repetition time (ms)
        :param inversion: whether inversion pulse is applied or not (bool)
        :param TI: inversion time (ms)
        :param fake_param: synthesized q*-maps
        :param test: used for testing or not (bool)
        """
        if test:  # fix tissue parameters with predefined values for testing
            PD = torch.ones((100, 1), dtype=torch.float16, device=device) * 0.6
            T1 = torch.ones((100, 1), dtype=torch.float16, device=device) * 779
            T2 = torch.ones((100, 1), dtype=torch.float16, device=device) * 45
        else:
            PD = fake_param[0, 0, :, :]
            T1 = fake_param[0, 1, :, :] * 5000 + 1e-2
            T2 = fake_param[0, 2, :, :] * 1500 + 1e-2

        # refocusing pulse series (rad)
        if nrefocus == 1:  # T1w
            a0 = [70 * math.pi / 180] + [math.pi]
            Ntr = 5
        else:  # PDw, T2w, FLAIR
            a0 = [math.pi / 2] + [120 * math.pi / 180] * nrefocus
            Ntr = 2

        Tdelay = TR - ESP*(nrefocus+0.5)
        Xi = torch.exp(-Tdelay/T1)

        for ii in range(Ntr):  # simulate multiple TRs
            if ii == 0:
                s, _, Zn = self.EPG_TSE(device, a0, ESP, T1, T2, PD, inversion, TI)  # overwrite signal each TR
            else:
                s, _, Zn = self.EPG_TSE(device, a0, ESP, T1, T2, z0, inversion, TI)  # overwrite signal each TR

            z0 = Zn[:, 0, -1]  # at final shot
            z0 = Xi * torch.unsqueeze(z0, 1) + 1 - Xi  # regrow

        # contrast is determined by echo that fills central k-space (~effective echo time)
        return torch.abs(s[:, math.ceil(TE / ESP) - 1])*PD[:, 0]


# use for testing
if __name__ == "__main__":
    EPG_model = EPG()
    for i in range(10):  # evaluate multiple times
        now = time.time()
        fake = EPG_model.forward('cuda:1', 1, 14, 14, 864, False, 0,  0, True)
        print(time.time() - now, fake[0])

"""EPGX Simulation module

This module performs EPG simulations on the GPU.
It is based upon some MATLAB code, as was written
and documented at 'https://github.com/mriphysics/EPG-X' by
Shaihan Malik in 06/2017, was reimplemented in Python
by Luuk Jacobs in 01/2022 and then partly repurposed and adjusted
by Sjors Verschuren in 05/2022.
"""

import math
from typing import Union
import numpy as np
import torch
import time


class EPG(torch.nn.Module):
    """Module implementing a GPU-accelerated EPG simulator

    This may be used to perform EPG simulations on multiple
    pixels/voxels in parallel, greatly improving computing times
    on full images.
    """

    @staticmethod
    def EPG_shift_matrices(device, Nmax):
        N = (Nmax + 1) * 3
        S = torch.zeros((N * N, 1), dtype=torch.complex64, device=device)

        kidx = list(range(3, N, 3))  # miss out F1+
        sidx = [x - 3 for x in kidx]  # negative states shift 4 back
        # linear indices of S(kidx, sidx)
        idx = [a + b for a, b in zip(kidx, [N * x for x in sidx])]
        S[idx] = 1

        kidx = list(range(4, N, 3))
        kidx.pop()  # most negative state is nothing
        sidx = [x + 3 for x in kidx]  # Positive states shift 4 forward
        # linear indices of S(kidx, sidx)
        ix = [a + b for a, b in zip(kidx, [N * x for x in sidx])]
        S[ix] = 1

        kidx = list(range(2, N, 3))  # Za states
        # linear indices of S(kidx, kidx)
        ix = [a + b for a, b in zip(kidx, [N * x for x in kidx])]
        S[ix] = 1

        S = torch.t(torch.reshape(S, (N, N)))
        S[0:2, 4] = 1  # F0+ sates

        return S

    @staticmethod
    def kron(a: torch.Tensor, b: torch.Tensor):
        """yulkang/kron.py"""
        siz1 = torch.Size(
            torch.tensor(a.shape[-2:])
            * torch.tensor(b.shape[-2:])  # type: ignore
        )
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    @staticmethod
    def RF_rot(
        device: torch.device,
        alpha: torch.Tensor,
        phi: torch.Tensor
    ):
        """RF rotation along an arbitrary direction/phase

        Here, note that by convention:
        Along x-axis -> phi = 0
        Along y-axis -> phi = pi / 2
        """

        RR = torch.zeros((3, 3), device=device, dtype=torch.complex64)

        RR[0, 0] = torch.cos(alpha / 2) ** 2
        RR[1, 0] = torch.exp(-2 * phi * 1j) * torch.sin(alpha / 2) ** 2
        RR[2, 0] = -0.5 * 1j * torch.exp(-1j * phi) * torch.sin(alpha)
        RR[0, 1] = torch.conj(RR[1, 0])
        RR[1, 1] = RR[0, 0]
        RR[2, 1] = 0.5 * 1j * torch.exp(1j * phi) * torch.sin(alpha)
        RR[0, 2] = -1j * torch.exp(1j * phi) * torch.sin(alpha)
        RR[1, 2] = 1j * torch.exp(-1j * phi) * torch.sin(alpha)
        RR[2, 2] = torch.cos(alpha)

        return RR

    def EPG_GRE(
        self,
        device: torch.device,
        theta: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        TR: torch.Tensor
    ):
        """Method for GPU-accelerated single-compartment Gradient Echo EPG

        This code was adapted from
        https://github.com/mriphysics/EPG-X/EPGX-src/EPG_GRE.m, rewritten in
        Python and enabled for GPU-accelerated parallel computing of multiple
        voxels.

        # TODO: Check for possible efficiency gains

        Parameters
        -----------
            device : torch.device
                Device used for tensor computation
            theta: torch.Tensor(dtype=complex64)
                Tensor containing the RF pulses in complex form
                i.e. with amplitude and phase
            T1 : torch.Tensor(dtype=float32)
                Tensor containing the T1 values of the to-be-simulated
                voxels
            T2 : torch.Tensor(dtype=float32)
                Tensor containing the T2 values of the to-be-simulated
                voxels
            TR : torch.Tensor(dtype=float32)
                Tensor containing a single scalar value, being the repetition
                time of the sequence

        Returns
        -------
            F0 : torch.Tensor(dtype=complex64)
                Tensor containing the F0 state (signal) after each excitation,
                for all voxels
                Size = (#voxels, #pulses)
            Fn : torch.Tensor(dtype=complex64)
                Full EPG diagram for all EPG transverse states
            Zn : torch.Tensor(dtype=complex64)
                Full EPG diagram for all EPG longitudinal states
        """

        # Define flip angle train length and max EPG order
        np2 = len(theta)
        kmax = np2 - 1

        # Define max EPG order per pulse (TODO: Check this.)
        kmax_per_pulse = np.array([
            x
            for x in (
                list(range(1, math.ceil(np2 / 2) + 1))
                + list(range(math.floor(np2 / 2), 0, -1))
            )
        ])
        kmax_per_pulse[kmax_per_pulse > kmax] = kmax
        # Adjust kmax
        if max(kmax_per_pulse) < kmax:
            kmax = max(kmax_per_pulse)

        # Determine the number of total states
        N = 3 * (kmax + 1)

        # Define the number of voxels/pixels we simulate in parallel
        im_dim = T1.shape[0]

        # Define the shift matrix
        S = self.EPG_shift_matrices(device, kmax)

        # Define relaxation matrix
        E1 = torch.exp(-TR / T1)
        E2 = torch.exp(-TR / T2)
        E = torch.diag_embed(
            torch.stack([E2, E2, E1], dim=2)
        )[:, 0, :, :].to(device)

        # Define regrowth matrix (Naturally only Z0 grows back)
        b = torch.zeros((im_dim, N, 1), dtype=torch.complex64, device=device)
        b[:, 2, :] = 1 - E1

        # We don't use diffusion, so E is the same for all EPG orders
        E = self.kron(torch.unsqueeze(
            torch.eye(kmax + 1, dtype=torch.complex64, device=device), 0
        ), E)

        # Define composite relaxation-shift matrix
        ES = torch.matmul(E, S)

        # Pre-allocate RF matrix and store the indices of the top left corner
        T = torch.zeros((1, N, N), dtype=torch.complex64, device=device)

        i1 = [y + ii for y in [x * N for x in range(3)] for ii in range(3)]
        ksft = 3 * (3 * (kmax + 1) + 1)

        # Define F matrix
        F = torch.zeros((im_dim, N, np2), dtype=torch.complex64, device=device)

        # Define initial state
        FF = torch.zeros((im_dim, N, 1), dtype=torch.complex64, device=device)
        FF[:, 2] = 1.  # initial state

        # Main body: loop over the pulse train
        for jj in range(np2):
            # Get the RF rotation matrix
            A = self.RF_rot(
                device,
                torch.abs(theta[jj]),
                torch.angle(theta[jj])
            )

            # Get max number of states
            kmax_current = kmax_per_pulse[jj]
            kidx = list(range(3 * (kmax_current + 1)))

            # Build new T, given the current rotation matrix
            T = T.reshape(1, N * N)
            for i2 in range(9):
                ind_lin = list(range(i1[i2], N * N + 1, ksft))
                T[:, ind_lin] = A.reshape(1, -1, 1, 1)[:, i2, 0]
            T = T.reshape(1, N, N)

            # Apply T to the EPG state matrix
            F[:, :kidx[-1] + 1, jj] = torch.matmul(
                T[:, :kidx[-1] + 1, :kidx[-1] + 1],
                FF[:, kidx, :]
            ).squeeze()

            # If we're at the end of the echo train, break the loop
            if jj == np2 - 1:
                break

            # Calculate state evolution
            FF[:, kidx] = torch.matmul(
                ES[:, :kidx[-1] + 1, :kidx[-1] + 1],
                F[:, :kidx[-1] + 1, jj].unsqueeze(-1)
            ) + b[:, kidx]

            # Deal with complex conjugate after shift
            FF[:, 0] = torch.conj(FF[:, 0])

        idx = (
            list(range(4, F.shape[1], 3))[::-1]
            + [0]
            + list(range(3, F.shape[1], 3))
        )
        idx.pop(0)
        idx.pop(0)

        Fn = F[:, idx, :]
        Zn = F[:, 2::3, :]

        return F[:, 0, :], Fn, Zn

    def forward(
        self,
        device: torch.device,
        theta: torch.Tensor,
        TR: torch.Tensor,
        quantitative_maps: Union[None, torch.Tensor],
        test: bool = False
    ):
        """Main forward method

        This function simply takes in the simulation parameters
        and returns the signal after each pulse for each voxel.
        """

        # Fix tissue parameters with predefined values if testing.
        # Else, just use the tensors passed
        if test:
            PD = torch.ones((256, 1), dtype=torch.float32, device=device) * 0.6
            T1 = torch.ones((256, 1), dtype=torch.float32, device=device) * 700
            T2 = torch.ones((256, 1), dtype=torch.float32, device=device) * 40
        else:
            if quantitative_maps is not None:
                PD = quantitative_maps[0, :, :]
                T1 = quantitative_maps[1, :, :]
                T2 = quantitative_maps[2, :, :]
            else:
                raise ValueError(
                    "If not in test mode, quantitative_maps must be passed"
                )

        # Cast params to tensors if applicable
        if type(TR) != torch.Tensor:
            TR = torch.tensor([TR], dtype=torch.float32, device=device)
        if type(theta) != torch.Tensor:
            theta = torch.tensor(theta, dtype=torch.complex64, device=device)

        # Run GRE simulation
        s, _, _ = self.EPG_GRE(device, theta, T1, T2, TR)

        return torch.abs(s) * PD


if __name__ == "__main__":

    # Either plot the results for validity checking
    # or don't and speed-test
    plot = False

    # Import plt for plotting
    if plot:
        import matplotlib.pyplot as plt

    # Create EPG model
    EPG_model = EPG()

    # Run the simulation and plot results if applicable
    for i in range(1 if plot else 10):
        now = time.time()
        signals = EPG_model.forward(
            torch.device('cuda'),
            torch.tensor(
                [0.25 * torch.pi] * 100,
                dtype=torch.complex64
            ), torch.tensor([50], device=torch.device('cuda')), None, True
        )
        print(
            f"Simulation done. Took {(time.time() - now) * 1000.:.1f} ms"
            f" - Result: {signals[0][-1]:.4f}"
        )

        # Plot the results if applicable
        if plot:
            plt.plot(list(range(100)), np.array(signals[0]))  # type: ignore
            plt.show()                                        # type: ignore

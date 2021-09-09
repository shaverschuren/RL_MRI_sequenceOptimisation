"""EPG Simulation module

This module performs basic EPG simulations.
It is based upon /epg_code/matlab/EPG.m, as was written
and documented at 'http://epg.matthias-weigel.net' by
Matthias Weigel in 11/2014, and was reimplemented in Python
by Sjors Verschuren in 09/2021."""

from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt


def epg(
        N_in: int,
        alpha: Union[float, np.ndarray],
        TR: float,
        T1: float,
        T2: float,
        SP: complex = complex(1, 0),
        spoil: bool = True,
        rf_phase_mode: int = 1,
        rf_phase_inc: float = 150.
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Extended Phase Graph (EPG) for some
    variants of gradient echo (GE) / steady state free
    precession (SSFP) sequences.

        Parameters:
            N_in  (int)                  : Number of repetitions
            alpha ((float|np.ndarray)    : Flip angle (constant or list) [deg]
            TR    (float)                : Repetition time               [ms]
            T1    (float)                : Longitudinal relaxation time  [ms]
            T2    (float)                : Transverse relaxation time    [ms]
            SP    (complex)              : Slice profile (default 1 + 0j)
            spoil (bool)                 : True*  - Unspoiled, balanced
                                           False  - Spoiled, unbalanced
            rf_phase_mode (int)          : +1* = RF spoiling with increment
                                           -1  - Alternating RF flip angles
                                           0   - For all pulses, phi=0deg
            rf_phase_inc  (float)        : Linear RF phase increment,
                                           used only if rf_phase_mode == +1
                                           (default 150) [deg]

        Returns:
            F0_vector (np.array[float]) : Vector of resulting F0 states
            Xi_F      (np.array[float]) : State evolution matrix of F states
            Xi_Z      (np.array[float]) : State evolution matrix of Z states
    """

    # Define max number of states
    maxstate = 150

    # Convert flip angle into [rad]
    alpha = (alpha / 180.0) * np.pi

    # Determine effective alpha and phi arrays and dedicated
    # length parameters: SSFP variant settings
    if type(alpha) == np.ndarray:
        # Calculate phi, flip angle array
        phi = np.array([-np.angle(SP)] * len(alpha))
        fa = np.array([abs(SP) * alpha] * len(alpha))
        # The flip angle (array) is repeated N_in times
        for pn in range(1, N_in):
            fa = np.concatenate(fa, alpha)
    elif type(alpha) == float:
        # Calculate phi, flip angle array
        phi = np.array(-np.angle(SP))
        fa = np.array(abs(SP) * alpha)
        # The flip angle is repeated N_in times
        fa_arr = np.zeros((N_in))
        fa_arr[0] = fa
        for pn in range(1, N_in):
            fa_arr[pn] = alpha
        fa = fa_arr
    else:
        raise TypeError("alpha should be either an array or float!")

    # fa_arr = np.zeros((1, len(alpha) * N_in))
    # fa_arr[:len(alpha)] = fa
    # for pn in range(1, N_in):  # The flip angle (array) is repeated N_in times
    #     fa_arr[len(alpha) * pn:len(alpha) * (pn + 1)] = np.array(alpha)

    # fa = fa_arr

    N = len(fa)
    Nm1 = N - 1
    Np1 = N + 1

    if rf_phase_mode == -1:
        # Realize alternating RF flip angles
        for i in range(len(fa)):
            if i % 2 == 0:
                fa[i] = -fa[i]

        phi[:N] = 0
        if N > 1:
            # More than one alternating flip angle? Use initial fa/2 pulse!
            pass  # fa(1)=fa(1)/2;

    elif rf_phase_mode == +1:
        # Realize RF spoiling with dedicated RF phase angle
        pn = np.array(range(N)) + 1
        phi = (pn - 1) * pn / 2.0 * rf_phase_inc / 180.0 * np.pi

    elif rf_phase_mode == 0:
        # Realize "normal" flip angles with phi=0
        phi[:N] = 0

    else:
        raise ValueError("rf_phase_mode should be in [0, -1, +1]")

    # Define relaxation operator E elements
    E1 = np.exp(-TR / T1)  # Eq.[23] in EPG-R
    E2 = np.exp(-TR / T2)  # Eq.[23] in EPG-R

    # Generate state matrices Omega before and after RF: Eq.[26] in EPG-R
    Omega_preRF = np.zeros((3, Np1), dtype=complex)
    Omega_postRF = np.zeros((3, N), dtype=complex)

    # Generate state evolution matrices Xi_F and Xi_Z: Eqs.[27] in EPG-R
    # Here, they will (only) contain all post-RF states == output variables
    Xi_F_out = np.zeros((2 * N - 1, N), dtype=complex)  # Xi_F with F+ and F- states, Eq.[27a]
    Xi_Z_out = np.zeros((N, N), dtype=complex)          # Xi_Z with Z states, Eq.[27b]

    # Starting with equilibrium magnetization M0=1 for t<0
    # You could change that !
    Omega_preRF[2, 0] = 1

    # Starting the calculation of EPG states over time
    # ,or "flow of magnetization in the EPG"
    # -------------------------------------------------------------------------------------------
    for pn in range(N):  # Loop over RF pulse #pn

        # Generate T matrix operator elements (RF pulse representation):
        # Eq.[15] or Eq.[18] in EPG-R
        # Since this generic SSFP example may change RF flip angle and/or
        # phase, the T matrix has to be updated within the RF pulse loop:
        T = np.zeros((3, 3), dtype=complex)
        T[0, 0] = np.cos(fa[pn] / 2) ** 2
        T[0, 1] = np.exp(+2j * phi[pn]) * np.sin(fa[pn] / 2) ** 2
        T[0, 2] = -1.0j * np.exp(+1j * phi[pn]) * np.sin(fa[pn])
        T[1, 0] = np.exp(-2j * phi[pn]) * np.sin(fa[pn] / 2) ** 2
        T[1, 1] = np.cos(fa[pn] / 2) ** 2
        T[1, 2] = +1.0j * np.exp(-1j * phi[pn]) * np.sin(fa[pn])
        T[2, 0] = -0.5j * np.exp(-1j * phi[pn]) * np.sin(fa[pn])
        T[2, 1] = +0.5j * np.exp(+1j * phi[pn]) * np.sin(fa[pn])
        T[2, 2] = np.cos(fa[pn])

        # In the following, further loops over the states' dephasing k will be
        # needed to realize operators T, E, and S.
        # --> Note that we deal with integral k units here, see EPG-R
        # Intead of "for loops over k", numpy's indexing will
        # be used, which is much faster for large values.
        # To improve efficiency, only states are calculated that could
        # be generated at all in the EPG before.
        if pn < maxstate:
            k = range(pn)  # "k loop index" with limit pn
        else:
            k = range(maxstate)
        k = np.array(list(k), dtype=int)

        # T matrix operator: RF pulse acting, mixing of F+, F-, and Z states
        # Expand T matrix relations from Eq.[15] or Eq.[18] in EPG-R
        Omega_postRF[:, k] = np.matmul(T, Omega_preRF[:, k])  # matrix product!

        # Store these post-RF states of the current Omega state matrix in
        # the Xi state evolution matrices
        Xi_F_out[Np1 - k - 2, pn] = Omega_postRF[0, k]
        Xi_F_out[Nm1 + k[1:], pn] = np.conj(Omega_postRF[1, k[1:]])
        Xi_Z_out[Np1 - k - 2, pn] = Omega_postRF[2, k]

        # E matrix operator: Experienced relaxation from the states
        # until the next TR.
        # Expand E matrix relations from Eqs.[23] and [24] in EPG-R
        Omega_preRF[0:1, k] = E2 * Omega_postRF[0:1, k]       # T2 relaxation F
        Omega_preRF[2, k[1:]] = E1 * Omega_postRF[2, k[1:]]   # T1 relaxation Z (k>0)
        Omega_preRF[2, 0] = E1 * Omega_postRF[2, 0] + 1 - E1  # T1 recovery Z (k=0)

        # S operator: Further dephasing / shifting of F+ and F- states
        # to the next TR.
        # With integral k units (see EPG-R):
        #   Delta(k) = +1 (unbalanced, up shifting),
        #   Delta(k) = 0 (balanced, skip S operator)
        if spoil:
            Omega_preRF[0, k + 1] = Omega_preRF[0, k]       # dephase F+
            Omega_preRF[1, k] = Omega_preRF[1, k + 1]       # dephase F-
            # generate conjugate pendant F+0 from F-0, see EPG-R
            Omega_preRF[0, 0] = np.conj(Omega_preRF[1, 0])

    # Output: "make nice zeros"
    # Erase some float point accuracy errors
    Xi_F_out[np.abs(Xi_F_out) < 1e-8] = 0
    Xi_Z_out[np.abs(Xi_Z_out) < 1e-8] = 0

    # Output: define "echoes" separately
    F0_vector_out = Xi_F_out[N - 1, :]

    # Subtract the RF phase from the simulations for each readout
    if rf_phase_mode == 1:
        F0_vector_out = F0_vector_out * np.exp(-1j * phi)

    return F0_vector_out, Xi_F_out, Xi_Z_out


def example(plot: bool = True):
    """Example function for using epg()"""

    # Define parameters
    T1 = .583       # T1 relaxation time of the spin [s]
    T2 = 0.055      # T2 relaxation time of the spin [s]
    B1 = 1          # B1 field of the spin [-]
    fa = 20         # Flip angle of the sequence [deg] (Can also be an array)
    Nfa = 500       # Number of flip angles to achieve a steady state [-]
    tr = 10E-03     # Repetition time [s]
    spoil = 1       # 0 = balanced, 1 = spoiled

    # Perform EPG Simulation
    F0, Xi_F, Xi_Z = epg(Nfa, fa, tr, T1, T2, complex(1, 0), spoil)

    # Plot results (if applicable)
    if plot:
        plt.figure()  # Create figure

        plt.subplot(2, 2, 1)  # Absolute signal
        plt.plot(range(len(F0)), np.abs(F0))
        plt.title("Abs(signal)")
        plt.xlabel("Pulse n")
        plt.ylabel("Signal")
        plt.subplot(2, 2, 2)  # Real part
        plt.plot(range(len(F0)), np.real(F0))
        plt.title("Real(signal)")
        plt.xlabel("Pulse n")
        plt.ylabel("Signal")
        plt.subplot(2, 2, 3)  # Imaginary part
        plt.plot(range(len(F0)), np.imag(F0))
        plt.title("Imag(signal)")
        plt.xlabel("Pulse n")
        plt.ylabel("Signal")
        plt.subplot(2, 2, 4)  # Angle
        plt.plot(range(len(F0)), np.angle(F0))
        plt.title("Angle(signal)")
        plt.xlabel("Pulse n")
        plt.ylabel("Angle [rad]")

        plt.show()  # Show figure


if __name__ == "__main__":
    example()

"""Module implementing the k-space acquisition simulator class"""


# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# Specific imports
from typing import Union                            # noqa: E402
import time                                         # noqa: E402
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402
from epg_simulator.python.epg_gpu import EPG        # noqa: E402


class SimulatorObject():
    """Class representing a 2D simulator object"""

    def __init__(
        self,
        quantative_maps_path: Union[str, os.PathLike],
        sequence: str = "GRE",
        device: Union[None, torch.device] = None
    ):

        """Initializes and builds attributes for this class

            Parameters
            ----------
                quantative_maps_path : str | os.PathLike
                    Path to the quantative maps for this simulator object
                sequence : str
                    String containing the sequence type. For now, only
                    "GRE" - Gradient Echo is implemented.
                device: None | torch.device
                    Torch device. If None, it's assigned below
            """

        # Build attributes for this class
        self.quantative_maps_path = quantative_maps_path
        self.sequence = sequence

        # Setup torch device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Intiialize quantative maps
        self.init_quantative_maps()

        # Check sequence and initialize EPG model
        if sequence == "GRE":
            # Initialize EPG model
            self.epg = EPG()
        else:
            raise NotImplementedError()

    def init_quantative_maps(self):
        """Function initializing the quantative maps for this object"""

        # Get directory path
        map_dir = self.quantative_maps_path

        # Check path validity
        if not os.path.isdir(map_dir):
            raise FileNotFoundError(
                f"This directory doesn't exist. Got '{map_dir}'"
            )
        else:
            if any(
                file_name not in os.listdir(map_dir)
                for file_name in ['T1.npy', 'T2.npy', 'PD.npy']
            ):
                raise FileNotFoundError(
                    "The given directory exists, but it does not contain "
                    "the required files. "
                    "Expected ['T1.npy', 'T2.npy', 'PD.npy'] but got "
                    f"{os.listdir(map_dir)}."
                    f"\nGiven directory is {map_dir}"
                )

        # If everything is valid, import the quantitative maps
        self.T1_map_np = np.load(os.path.join(map_dir, "T1.npy"))
        self.T2_map_np = np.load(os.path.join(map_dir, "T2.npy"))
        self.PD_map_np = np.load(os.path.join(map_dir, "PD.npy"))

        # Load into tensors (used for GPU acceleration)
        self.T1_map_torch = torch.FloatTensor(self.T1_map_np)
        self.T2_map_torch = torch.FloatTensor(self.T2_map_np)
        self.PD_map_torch = torch.FloatTensor(self.PD_map_np)

        # Store original shape and then flatten the tensors
        self.img_shape = self.T1_map_torch.size()

        self.T1_map_torch = torch.reshape(
            self.T1_map_torch, (self.img_shape[0] * self.img_shape[1], 1)
        )
        self.T2_map_torch = torch.reshape(
            self.T2_map_torch, (self.img_shape[0] * self.img_shape[1], 1)
        )
        self.PD_map_torch = torch.reshape(
            self.PD_map_torch, (self.img_shape[0] * self.img_shape[1], 1)
        )

        # Stack maps
        self.maps = torch.stack(
            (self.T1_map_torch, self.T2_map_torch, self.PD_map_torch)
        ).to(self.device)

    def forward(
        self,
        theta: torch.Tensor,
        tr: float = 0.050,
        n_prep: int = 1
    ):
        """Method implementing a forward simulation of a pulse train

        Note that we've only implemented GRE so far.
        The k-space filling trajectory is (for now) only cartesian sequential.
        We built in oversampling in the readout direction for better
        simulation of actual acquisition.

        Parameters
        ----------
            theta : torch.Tensor
                Tensor containing the (complex) RF pulse info, being
                amplitude and phase. Shape is (n_pulses,)
            tr : float
                Repetition time used in the simulation. Default: 50 ms
            n_prep : int
                Number of preparation pulses used in the simulation.
                Here, the length of theta minus number of preperation pulses
                should equal the amount of k-space lines in the phase
                encoding direction!
        """

        # Check the validity of the passed parameters
        if len(theta) - n_prep != self.img_shape[0]:
            raise ValueError(
                "The amount of pulses (i.e. length of theta) minus "
                "the number of preparation pulses (n_prep) should equal "
                "the amount of k-space lines in the phase encoding direction."
                f"\nGot {len(theta)} (n_pulses) - {n_prep} (n_prep) = "
                f"{len(theta) - n_prep} != {self.img_shape[0]}"
            )

        # Run EPG simulation
        signals = self.epg.forward(
            self.device, theta, torch.tensor([tr], device=self.device),
            self.maps
        )

        # Reshape into stack of original image shape
        signals = torch.abs(
            signals.reshape((*self.img_shape, len(theta))).permute(2, 0, 1)
        )

        # Pad signals in the readout direction for effective RO oversampling
        signals = F.pad(
            signals,
            (self.img_shape[1] // 2, self.img_shape[1] // 2),
            "constant", 0.
        )

        # Fast Fourier Transform (2D)
        k_spaces = torch.fft.fft2(signals)

        # Simulate k-space filling
        k_space = torch.zeros(
            (self.img_shape[0], self.img_shape[1] * 2),
            dtype=torch.complex64
        )

        for line in range(self.img_shape[0]):
            k_space[line] = k_spaces[line + n_prep][line]

        # Inverse Fourier
        image = torch.fft.ifft2(k_space)

        # Remove padding
        image = image[:, self.img_shape[1] // 2:(self.img_shape[1] * 3) // 2]

        # Return image
        return torch.abs(image), signals


if __name__ == "__main__":

    # Include timer for debugging
    start_time = time.time()

    # Initialize the simulator
    simulator = SimulatorObject('tmp/')
    initialization_time = time.time() - start_time
    print(f"Initialization done!    Took {initialization_time:.4f} seconds")

    # Run a tryout simulation
    image, _ = simulator.forward(theta=torch.ones((65)) * .25 * torch.pi)
    simulation_time = time.time() - initialization_time - start_time
    print(f"Simulation done!        Took {simulation_time:.4f} seconds")

    # For debugging purposes, plot result
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.plot(np.array(list(range(1, len(signals) + 1))), np.abs(signals))
    # plt.show()

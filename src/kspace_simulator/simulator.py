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
from epg_simulator.python import epg                # noqa: E402


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

        # Check sequence
        if sequence != "GRE":
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

    def forward(
        self,
        alphas: np.ndarray,
        tr: float = 0.050
    ):
        """Function implementing a forward simulation of a pulse train

        Note that we've only implemented GRE so far.
        The k-space filling trajectory is (for now) only cartesian sequential.
        """

        # Quick tryout
        F0, _, _ = epg.epg_as_numpy(
            len(alphas), alphas, tr,
            0.6, 0.01
        )

        return F0


if __name__ == "__main__":

    # Include timer for debugging
    start_time = time.time()

    # Initialize the simulator
    simulator = SimulatorObject('tmp/')
    initialization_time = time.time() - start_time
    print(f"Initialization done!    Took {initialization_time:.4f} seconds")

    # Run a tryout simulation
    F0 = simulator.forward(alphas=np.arange(50, 0, -0.5))
    simulation_time = time.time() - initialization_time - start_time
    print(f"Simulation done!        Took {simulation_time:.4f} seconds")

    # For debugging purposes, plot result
    import matplotlib.pyplot as plt
    plt.plot(np.array(list(range(1, len(F0) + 1))), np.abs(F0))
    plt.show()

"""Module implementing the k-space acquisition simulator class"""

import os
from typing import Union
import numpy as np
import torch
from epg_simulator.python import epg  # TODO: Make this gpu-compatible


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
            if ["T1.npy", "T2.npy", "PD.npy"] not in os.listdir(map_dir):
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
        alphas: Union[float, np.ndarray]
    ):
        """Function implementing a forward simulation of a pulse train

        Note that we've only implemented GRE so far.
        The k-space filling trajectory is (for now) only cartesian sequential.
        """

        raise NotImplementedError()

"""Module implementing a scanner simulator class"""

import os
from typing import Union
import h5py
import time
import json
import numpy as np
from epg_simulator.python import epg


class Simulator(object):
    """Class implementing a MRI phantom scan simulator"""

    def __init__(
            self,
            config_path: Union[str, os.PathLike],
            n_phantoms: int = 1,
            resolution: int = 256,
            T1a: float = 0.500,
            T2a: float = 0.050,
            T1b: Union[float, None] = None,
            T2b: Union[float, None] = None,
            noise_level: float = 0.05):
        """Initializes and builds attributes for this class

            Parameters
            ----------
                config_path : str | os.PathLike
                    Path to config file
                n_phantoms : int
                    Number of phantoms to simulate
                resolution : int
                    Number of pixels per dimention (NxN image)
                T1a : float
                    T1 of first phantom
                T2a : float
                    T2 of first phantom
                T1b : float | None
                    T1 of second phantom (optional)
                T2b : float | None
                    T2 of second phantom (optional)
                noise_level ; float
                    Noise level for this scanner simulator
            """

        # Build attributes for this class
        self.config_path = config_path
        self.n_phantoms = n_phantoms
        self.resolution = resolution
        self.T1a = T1a
        self.T2a = T2a
        self.T1b = T1b
        self.T2b = T2b
        self.noise_level = noise_level

        # Read config file
        self.read_config()
        # Create phantom mask
        self.init_mask()

    def read_config(self):
        """Read info from config file for scanner interaction"""

        # Read config file
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Define communication paths
        self.txt_path = self.config["param_loc"]
        self.data_path = self.config["data_loc"]

    def init_mask(self):
        """Initialize the phantom mask(s) required for the simulator"""

        if self.n_phantoms == 1:
            # If n_phantoms is 1, create a single phantom mask
            self.phantom_mask = self.create_phantom(
                a=int(self.resolution / 2),
                b=int(self.resolution / 2),
                r=int(self.resolution / 3)
            )
        elif self.n_phantoms == 2:
            # If n_phantoms is 2, create two phantom masks
            self.phantom_mask_a = self.create_phantom(
                a=int(self.resolution / 4),
                b=int(self.resolution / 4),
                r=int(self.resolution / 5)
            )
            self.phantom_mask_b = self.create_phantom(
                a=int(self.resolution * 3 / 4),
                b=int(self.resolution * 3 / 4),
                r=int(self.resolution / 5)
            )
        else:
            raise ValueError(
                "The value of n_phantoms should be 1 or 2, "
                f"but got {self.n_phantoms}."
            )

    def create_phantom(self, a, b, r):
        """Create a phantom mask

        Parameters
        ----------
            a : int
                Center of phantom (x)
            b : int
                Center of phantom (y)
            r : int
                Radius of phantom
        """

        # Define resolution
        n = self.resolution

        # Define grid with coordinates as data
        y, x = np.ogrid[-a: n - a, -b: n - b]

        # Return boolean array (either within radius or not)
        return np.array(x * x + y * y <= r * r, dtype=np.uint0)

    def read_txt(self, path):
        """Read text file and extract new flip angle"""

        # Read txt file
        with open(self.txt_path, 'r') as f:
            fa_str = f.read()

        # Attempt to convert str to float
        fa = float(fa_str)

        # Return float fa
        return fa

    def run_epg(
            self,
            T1: float,
            T2: float,
            TR: float,
            flip_angle: float,
            n_pulse_train: int = 100):
        """Run EPG simulator for a set of scan and material parameters"""

        # Run EPG and extract F0
        F0, _, _ = epg.epg_as_numpy(
            n_pulse_train, flip_angle, TR,
            T1, T2
        )

        # Extract and return signal
        signal = float(np.abs(F0[-1])) * 2.  # Multiply by 2 (scaling factor)

        return signal

    def simulate_image(self, fa):
        """Simulate an image based on a passed flip angle"""

        # Initialize empty image
        img = np.zeros((self.resolution, self.resolution), dtype=np.float64)

        # If n_phantoms == 1, run single simulation
        if self.n_phantoms == 1:
            # Run epg simulation and get signal
            signal = self.run_epg(
                self.T1a, self.T2a, 0.050, fa
            )
            # Get image contribution of phantom
            phantom_image = \
                np.array(self.phantom_mask, dtype=np.float64) * signal

        # if n_phantoms == 2, run double simulation
        elif self.n_phantoms == 2:
            # Run epg simulation and get signal
            signal_a = self.run_epg(
                self.T1a, self.T2a, 0.050, fa
            )
            signal_b = self.run_epg(
                self.T1b if self.T1b else 0.0,
                self.T2b if self.T2b else 0.0,
                0.050, fa
            )
            # Get image contribution of phantoms
            phantom_image_a = \
                np.array(self.phantom_mask_a, dtype=np.float16) * signal_a
            phantom_image_b = \
                np.array(self.phantom_mask_b, dtype=np.float16) * signal_b

            # Combine phantom signal contributions
            phantom_image = phantom_image_a + phantom_image_b

        else:
            raise RuntimeError()

        # Create noise image
        noise_image = self.noise_level * np.random.random(
            size=(self.resolution, self.resolution)
        )

        # Create full image (signal + noise)
        img += phantom_image
        img += noise_image

        # Return image
        return img

    def write_data(self, img):
        """Write data to h5 file"""

        # Write image to (locked) h5 file
        hf = h5py.File(self.data_path + ".lck", 'w')
        hf.create_dataset('/img', data=np.array(img, dtype=np.float16))
        hf.close()

        # Move data file to actual reading location
        os.system(f"mv {self.data_path + '.lck'} {self.data_path}")

    def run(self):
        """Run the simulator to mimic scanner (RTRabbit behaviour)"""

        # Send initial image (this is default scanner behaviour)
        img = self.simulate_image(fa=30.)
        self.write_data(img)

        # Loop indefinitely
        while True:

            # Wait for text file containing fa info
            while not os.path.exists(self.txt_path):
                # Refresh file table
                os.system(f"ls {os.path.dirname(self.data_path)} > /dev/null")
                # Wait for a while
                time.sleep(0.05)

            # Read txt file and extract new flip angle
            fa = self.read_txt(self.txt_path)

            # Remove txt file
            os.remove(self.txt_path)

            # Print info
            print(
                f"Simulating image with flip angle: {fa:5.2f} [deg]...",
                end="", flush=True
            )

            # Simulate image for given flip angle
            img = self.simulate_image(fa)

            # Print info
            print(" Done")

            # Write image to h5 file
            self.write_data(img)

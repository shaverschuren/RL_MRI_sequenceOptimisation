"""Scanner CNR Validator

This module implements a validator for the CNR
optimizer with scanner interface. It simply loops over
a set of flip angles and creates a validation curve.
"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
import time                                                 # noqa: E402
from datetime import datetime                               # noqa: E402
import h5py                                                 # noqa: E402
import numpy as np                                          # noqa: E402
from util import loggers                                    # noqa: E402


class CNRValidator():
    """
    A class to represent a validator for the cnr
    optimization via flip angle modulation.
    """

    def __init__(
            self,
            fa_range: list[float] = [1., 90.],
            n_steps: int = 50,
            verbose: bool = True,
            log_dir=os.path.join(root, "logs", "cnr_validator")):
        """Constructs attributes for this validator"""

        # Setup attributes
        self.fa_range = fa_range
        self.n_steps = n_steps
        self.verbose = verbose
        self.log_dir = log_dir

        self.fa = fa_range[0]

        # Setup scanner interaction environment
        self.init_env()

        # Setup logger
        self.init_logger()

    def init_env(self):
        """Constructs the interaction environment

        i.e., provide paths to file locations on the server
        we use to communicate to the scanner.
        """

        # Define communication paths
        self.txt_path = \
            '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt'
        self.lck_path = \
            '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt.lck'
        self.data_path = '/nfs/rtsan01/RT-Temp/TomBruijnen/img_data.h5'

    def init_logger(self):
        """Sets up logger and appropriate directories"""

        # Create logs dir if not already there
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # Generate logs file path
        now = datetime.now()
        logs_filename = str(now.strftime("%Y_%m_%d-%H_%M_%S")) + ".csv"
        self.logs_path = os.path.join(self.log_dir, logs_filename)

        # Setup logger object
        self.logger = loggers.GeneralLogger(
            self.logs_path,
            columns=["step", "cnr", "fa"]
        )

    def perform_scan(self, fa=None):
        """Perform scan by passing parameters to scanner"""

        # Set flip angle we'll communicate to the scanner
        if not fa:
            fa = self.fa

        # Write new flip angle to appropriate location
        with open(self.lck_path, 'w') as txt_file:
            txt_file.write(f"{int(fa)}")
        os.system(f"mv {self.lck_path} {self.txt_path}")

        # Wait for image to come back by checking the data file
        while not os.path.isfile(self.data_path):
            time.sleep(0.1)

        # When the image is returned, load it and store the results
        with h5py.File(self.data_path, "r") as f:
            img = np.asarray(f['/img'])

        # Remove the data file
        os.remove(self.data_path)

        return img

    def calculate_cnr(self, image: np.ndarray):
        """Calculate CNR of a given image (np array)

        Here, we assume that the left side of the FOV is tissue 1, while
        the right side is tissue 2. The shape of the image is assumed
        to be `(N,N)` or `(N,N,1)` with axes `[x,y]` or `[x,y,z]`"""

        # Check image shape
        shape = np.shape(image)

        if len(shape) == 2:
            pass
        elif len(shape) == 3:
            # Check shape
            if shape[-1] != 1:
                raise ValueError(
                    "\nExpected shape of image to be either (N,N) or (1,N,N)."
                    f"\nFound {shape} instead.")
            else:
                # Squeeze image (if applicable)
                image = np.squeeze(image, -1)
                shape = shape[:-1]
        else:
            raise ValueError(
                "\nExpected shape of image to be either (N,N) or (1,N,N)."
                f"\nFound {shape} instead.")

        # Extract number of voxels in x direction
        N_x = shape[1]
        # Extract left and right parts of image
        img_left = image[:N_x // 2, :]
        img_right = image[N_x // 2:, :]

        # Calculate and return CNR
        cnr = abs(np.mean(img_left) - np.mean(img_right)) / np.std(image)

        return float(cnr)

    def run(self):
        """Runs the validation/calibration loop"""

        # Create list of flip angles we wish to try
        fa_list = np.linspace(
            self.fa_range[0], self.fa_range[1], self.n_steps
        )

        # Print start info
        if self.verbose:
            print(
                "\n======= Running CNR validation ======="
                "\n\n--------------------------------------"
                f"\nFA Range [deg]: [{self.fa_range[0]}, {self.fa_range[1]}]"
                f"\nN_scans:        {self.n_steps}"
                "\n--------------------------------------\n"
            )

        # Loop over flip angles, create scans and log results
        step = 0
        for fa in fa_list:
            # If applicable, print some info
            if self.verbose:
                print(
                    f"Scan #{step+1:2d}: fa = {fa:4.1f} [deg] -> scanning...",
                    end="", flush=True
                )

            # Perform scan
            img = self.perform_scan(fa=fa)
            # Calculate cnr
            cnr = self.calculate_cnr(img)
            # Log results
            self.logger.push([int(step), float(cnr), float(fa)])

            # If applicable, print some info
            if self.verbose:
                print(
                    f"\rScan #{step+1:2d}: fa = {fa:4.1f} [deg] -> "
                    f"cnr = {cnr:5.2f} [-]"
                )

            # Update step counter
            step += 1

        # Print logs location
        if self.verbose:
            print(
                "\nValidation complete! Logs stored at:"
                f"\n{self.logs_path}\n"
            )


if __name__ == "__main__":
    validator = CNRValidator()
    validator.run()

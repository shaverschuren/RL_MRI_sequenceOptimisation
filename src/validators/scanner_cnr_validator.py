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
import json                                                 # noqa: E402
import time                                                 # noqa: E402
from datetime import datetime                               # noqa: E402
import h5py                                                 # noqa: E402
import numpy as np                                          # noqa: E402
from util import loggers, roi                               # noqa: E402


class CNRValidator():
    """
    A class to represent a validator for the cnr
    optimization via flip angle modulation.
    """

    def __init__(
            self,
            fa_range: list[float] = [1., 50.],
            n_steps: int = 50,
            verbose: bool = True,
            overwrite_roi: bool = False,
            log_dir=os.path.join(root, "logs", "cnr_validator"),
            config_path=os.path.join(root, "config.json")):
        """Constructs attributes for this validator"""

        # Setup attributes
        self.fa_range = fa_range
        self.n_steps = n_steps
        self.verbose = verbose
        self.overwrite_roi = overwrite_roi
        self.log_dir = log_dir
        self.config_path = config_path

        self.fa = fa_range[0]

        # Setup scanner interaction environment
        self.init_env()

        # Setup logger
        self.init_logger()

        # Setup roi
        self.init_roi()

    def init_env(self):
        """Constructs the interaction environment

        i.e., provide paths to file locations on the server
        we use to communicate to the scanner.
        """

        # Read config file
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Define communication paths
        self.txt_path = self.config["param_loc"]
        self.lck_path = self.txt_path + ".lck"
        self.data_path = self.config["data_loc"]

    def init_logger(self):
        """Sets up logger and appropriate directories"""

        # Create logs dir if not already there
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # Generate logs file path and store tag
        now = datetime.now()
        logs_dirname = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
        self.logs_tag = logs_dirname
        self.logs_path = os.path.join(self.log_dir, logs_dirname)

        # Define datafields
        self.logs_fields = ["fa", "cnr", "img"]

        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def init_roi(self):
        """Setup ROI for this SNR optimizer"""

        # Generate ROI path and calibration image
        self.roi_path = os.path.join(self.log_dir, "roi.npy")
        calibration_image = self.perform_scan(pass_fa=False)

        # Check for existence of ROI file
        if not self.overwrite_roi and os.path.exists(self.roi_path):
            # Load ROI data
            self.roi = np.load(self.roi_path)
        else:
            # Generate new ROI data
            self.roi = roi.generate_rois(calibration_image, self.roi_path)

        # Check whether number of ROIs is appropriate
        if not np.shape(self.roi)[0] == 2:
            raise UserWarning(
                f"Expected a single ROI to be selected but got "
                f"{np.shape(self.roi)[0]}.\n"
                f"ROIs are stored in {self.roi_path}"
            )

    def perform_scan(self, fa=None, pass_fa=True):
        """Perform scan by passing parameters to scanner"""

        # If pass_fa, generate a flip angle file
        if pass_fa:
            # Set flip angle we'll communicate to the scanner
            if not fa:
                fa = self.fa

            # Write new flip angle to appropriate location
            with open(self.lck_path, 'w') as txt_file:
                txt_file.write(f"{fa:.2f}")
            os.system(f"mv {self.lck_path} {self.txt_path}")

        # Wait for image to come back by checking the data file
        while not os.path.exists(self.data_path):
            # Refresh file table
            os.system(f"ls {os.path.dirname(self.data_path)} > /dev/null")
            # Wait for a while
            time.sleep(0.05)

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

        # Extract left and right parts of image
        img_roi_1 = image[
            int(self.roi[0][0]):int(self.roi[0][1]),
            int(self.roi[0][2]):int(self.roi[0][3])
        ]
        img_roi_2 = image[
            int(self.roi[1][0]):int(self.roi[1][1]),
            int(self.roi[1][2]):int(self.roi[1][3])
        ]

        # Calculate and return CNR
        cnr = abs(
            np.mean(img_roi_1) - np.mean(img_roi_2)
        ) / np.std(np.concatenate((img_roi_1, img_roi_1)))

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
            # Log this step (scalars + image)
            self.logger.log_scalar(
                field="fa",
                tag=f"{self.logs_tag}_validation",
                value=float(fa),
                step=step
            )
            self.logger.log_scalar(
                field="cnr",
                tag=f"{self.logs_tag}_validation",
                value=float(cnr),
                step=step
            )
            self.logger.log_image(
                field="img",
                tag=f"{self.logs_tag}_validation",
                image=np.array(img),
                step=step
            )

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

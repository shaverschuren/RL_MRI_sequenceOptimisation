"""Module used for comparing our patient-specific results with 'normal' scan"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

from copy import deepcopy
import warnings
import glob
import numpy as np
import pandas as pd
import torch
import tqdm
import pickle
import kspace_simulator.simulator as kspace_sim  # noqa: E402


def get_subject_dirs(data_dir):

    subject_dirs = glob.glob(
        os.path.join(data_dir, "[0-9][0-9]_*/")
    )

    # Check all these subjects for the appropriate files
    if len(subject_dirs) > 0:
        # Check whether all appropriate files are present
        incomplete_subjects = []
        for dir in subject_dirs:
            if not (
                os.path.exists(os.path.join(dir, "T1.npy"))
                and os.path.exists(os.path.join(dir, "T2.npy"))
                and os.path.exists(os.path.join(dir, "PD.npy"))
                and os.path.exists(os.path.join(dir, "mask_1.npy"))
                and os.path.exists(os.path.join(dir, "mask_2.npy"))
            ):
                incomplete_subjects.append(dir)
        # Remove incomplete subject folders from list and give warning
        if len(incomplete_subjects) > 0:
            # Remove incomplete subjects from the subject directory list
            subject_dirs = [
                dir for dir in subject_dirs
                if dir not in incomplete_subjects
            ]
            # Throw warning
            incomplete_subjects.sort()
            warnings.warn(
                "\nWarning! Some of the data directories passed are "
                "incomplete.\nEvery directory is required to contain the "
                "following files: T1.npy; T2.npy; PD.npy; mask_1.npy; "
                "mask_2.npy."
                "\nThe directories that didn't match this criterium are:\n"
                + "\n".join(incomplete_subjects) + "\n"
            )

            # Check whether any comlete subjects remain
            if len(subject_dirs) == 0:
                raise UserWarning(
                    "No valid data directories remain. "
                    "Please consult te previous warning for instructions."
                )

        # Read one file to determine the number of acquisition pulses.
        # We won't check whether these are the same for every image for
        # now, but if this is not the case, we'll get errors later on!
        tryout_map = np.load(os.path.join(subject_dirs[0], "T1.npy"))
        img_shape = np.shape(tryout_map)

        # Store subject directories in self
        subject_dirs = subject_dirs

    else:
        raise FileNotFoundError(
            f"The given data directory ({data_dir}) "
            "does not contain folders with the appropriate "
            "name convention ('[0-999]_[0-999]')"
        )

    return subject_dirs, img_shape


def main():
    # Create to_dir
    if not os.path.isdir("tmp/figures_for_publication/b_eval"):
        os.mkdir("tmp/figures_for_publication/b_eval")

    # Set hyperparams
    data_dir = "data/processed"
    n_prep_pulses = 10
    alpha = 34. * (np.pi / 180.)

    # Retrieve subject dirs
    subject_dirs, img_shape = get_subject_dirs(data_dir)

    # Set number of pulses
    n_acq_pulses = img_shape[0]
    n_pulses = n_acq_pulses + n_prep_pulses

    img_list = []
    k_space_list = []
    Mz_list = []
    F0_list = []
    cnr_list = []
    imgs_cnr_dict = {}
    # Loop over subjects
    for subject_dir in tqdm.tqdm(subject_dirs):

        # Load masks
        mask_1 = np.load(os.path.join(subject_dir, "mask_1.npy"))
        mask_2 = np.load(os.path.join(subject_dir, "mask_2.npy"))
        # Store mask in self
        roi = np.array([mask_1, mask_2], dtype=bool)

        # Initialize simulator
        simulator = kspace_sim.SimulatorObject(subject_dir)

        # Set pulse train
        theta = torch.tensor([alpha] * n_pulses, dtype=torch.complex64)

        # Run simulation
        img, k_space, _ = simulator.forward(theta, n_prep=n_prep_pulses)

        # Log Mz (sum all longitudinal states and average all pixels)
        Mz = torch.mean(torch.sum(torch.abs(
            simulator.epg.Zn
        ), dim=1), dim=0)
        # Log F0 (average all pixels)
        F0 = torch.mean(torch.abs(
            simulator.epg.F0
        ), dim=0)

        # Extract rois
        img_roi = [img[roi[0]], img[roi[1]]]
        # Calculate CNR (signal difference / variances)
        cnr = float(
            torch.abs(
                torch.mean(img_roi[0]) - torch.mean(img_roi[1])
            )  # /
            # torch.sqrt(
            #     torch.var(img_roi[0]) + torch.var(img_roi[1])
            # )
        )

        # Store image in dict
        imgs_cnr_dict[os.path.split(subject_dir[:-1])[-1]] = (img.cpu().numpy(), cnr)
        cnr_list.append(cnr)

        # Add results to lists
        img_list.append(img.cpu().numpy())
        k_space_list.append(k_space.cpu().numpy())
        Mz_list.append(Mz.cpu().numpy())
        F0_list.append(F0.cpu().numpy())

    # Cast to arrays and store for backup
    with open("tmp/figures_for_publication/b_eval/img.pickle", 'wb') as f:
        pickle.dump(imgs_cnr_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    img_array = np.array(img_list)
    k_space_array = np.array(k_space_list)
    Mz_array = np.array(Mz_list)
    F0_array = np.array(F0_list)
    np.save("tmp/figures_for_publication/b_eval/img.npy", img_array)
    np.save("tmp/figures_for_publication/b_eval/k_space.npy", k_space_array)
    np.save("tmp/figures_for_publication/b_eval/Mz.npy", Mz_array)
    np.save("tmp/figures_for_publication/b_eval/F0.npy", F0_array)


if __name__ == "__main__":
    main()

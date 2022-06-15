"""Script to extract data in proper format from raw dicoms"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File specific imports
from typing import Union                # noqa: 402
from glob import glob                   # noqa: 402
from scipy.io import loadmat            # noqa: 402
from skimage.transform import resize    # noqa: 402
import warnings                         # noqa: 402
import numpy as np                      # noqa: 402
import nibabel as nib                   # noqa: 402
import matplotlib.pyplot as plt         # noqa: 402


def main(
    raw_dir: Union[str, os.PathLike],
    processed_dir: Union[str, os.PathLike],
    subject_regex: str = "MRSTAT[0-9][0-9]",  # Two-digit subject number
    resolution_limit: int = 128
):
    """Main extraction funciton"""

    # Check whether data dirs exist
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError("Raw data directory doesn't exist")
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    print(
        "\n================ Extracting data ================"
        f"\n\nFrom:\t{raw_dir}\nTo:\t{processed_dir}"
    )

    # Get subject list from regex
    subject_dirs = glob(os.path.join(raw_dir, subject_regex))
    subject_dirs.sort()

    print(f"\n\n{len(subject_dirs)} subjects found! Extracting from:")
    for dir in subject_dirs: print(f"{dir}")
    print("\n------------------------------")

    # Loop over subject dirs to check for the appropriate content
    # and copy the data into the 2D .npy format we want
    for subject_dir in subject_dirs:
        # Get subject number
        subject_i = int(str(subject_dir)[-2:])

        print(f"Processing subject #{subject_i}...")

        # Get the path for the qmap data and masks
        qmaps_path = os.path.join(subject_dir, f"MRSTAT{subject_i}.mat")
        mask_paths = [
            os.path.join(subject_dir, f"c{mask_i}t1w_q_{subject_i}.nii")
            for mask_i in [1, 2, 3]
        ]

        # Check whether all these scans are actually there
        if not all(
            [os.path.exists(path) for path in [qmaps_path, *mask_paths]]
        ):
            warnings.warn(
                "\nNot all required files are present for subject "
                f"in dir: {subject_dir}\nSkipping this subject."
            )
            continue

        # Import the data
        qmaps = np.moveaxis(
            np.array(loadmat(qmaps_path)["parMaps"]),
            [3, 2, 0, 1], [0, 1, 2, 3]
        )
        # Import the masks
        masks = [
            np.moveaxis(
                np.array(
                    nib.load(path).dataobj,
                    dtype=np.float64
                )[16:-16, 16:-16, :],
                [2, 0, 1], [0, 1, 2]
            )
            for path in mask_paths
        ]

        # Load in separate arrays
        PD = qmaps[0]
        T1 = qmaps[1]
        T2 = qmaps[2]
        GM_mask = masks[0]
        WM_mask = masks[1]
        CSF_mask = masks[2]

        # Check whether all array shapes are equal
        if not all([
            T1.shape == array.shape for array in
            [T2, PD, GM_mask, WM_mask, CSF_mask]
        ]):
            warnings.warn(
                f"\nFor subject {subject_i}, not all maps and masks have the "
                f"same shape! Skipping this subject. Check {subject_dir} "
                "for the files."
            )
            continue

        # Get the number of slices
        n_slices = np.shape(T1)[0]

        # Check for resample necessity and resample if necessary
        if any(dim > resolution_limit for dim in T1.shape):
            # maps
            T1 = resize(T1, tuple([n_slices] + [resolution_limit] * 2))
            T2 = resize(T2, tuple([n_slices] + [resolution_limit] * 2))
            PD = resize(PD, tuple([n_slices] + [resolution_limit] * 2))
            # masks
            GM_mask = np.array(
                resize(GM_mask, tuple([n_slices] + [resolution_limit] * 2)) > 0.5,  # noqa: E501
                dtype=np.uint8
            )
            WM_mask = np.array(
                resize(WM_mask, tuple([n_slices] + [resolution_limit] * 2)) > 0.5,  # noqa: E501
                dtype=np.uint8
            )
            CSF_mask = np.array(
                resize(CSF_mask, tuple([n_slices] + [resolution_limit] * 2)) > 0.5,  # noqa: E501
                dtype=np.uint8
            )

        # Create directories for all 2D slices of this subject
        # and fill them up with 2D slices of all maps and masks
        for slice_i in range(n_slices):
            # Create directory (if it doesn't already exist)
            slice_dir = os.path.join(processed_dir, f"{subject_i}_{slice_i}")
            if not os.path.isdir(slice_dir):
                os.mkdir(slice_dir)

            # Save the appropriate slices to this folder
            np.save(os.path.join(slice_dir, "T1.npy"), T1[slice_i])
            np.save(os.path.join(slice_dir, "T2.npy"), T2[slice_i])
            np.save(os.path.join(slice_dir, "PD.npy"), PD[slice_i])
            np.save(os.path.join(slice_dir, "GM_mask.npy"), GM_mask[slice_i])
            np.save(os.path.join(slice_dir, "WM_mask.npy"), WM_mask[slice_i])
            np.save(os.path.join(slice_dir, "CSF_mask.npy"), CSF_mask[slice_i])

            np.save(os.path.join(slice_dir, "mask_1.npy"), GM_mask[slice_i])
            np.save(os.path.join(slice_dir, "mask_2.npy"), WM_mask[slice_i])

            # Also create a single image per folder for visual inspection
            fig, axs = plt.subplots(2, 3)

            fig.suptitle(
                f"Visual inspection for subject #{subject_i}, slice #{slice_i}"
            )

            # Plot images
            axs[0, 0].imshow(T1[slice_i], cmap='gray')          # type: ignore
            axs[0, 1].imshow(T2[slice_i], cmap='gray')          # type: ignore
            axs[0, 2].imshow(PD[slice_i], cmap='gray')          # type: ignore
            axs[1, 0].imshow(GM_mask[slice_i], cmap='gray')     # type: ignore
            axs[1, 1].imshow(WM_mask[slice_i], cmap='gray')     # type: ignore
            axs[1, 2].imshow(CSF_mask[slice_i], cmap='gray')    # type: ignore

            # Set titles
            axs[0, 0].set_title("T1")                           # type: ignore
            axs[0, 1].set_title("T2")                           # type: ignore
            axs[0, 2].set_title("PD")                           # type: ignore
            axs[1, 0].set_title("GM mask")                      # type: ignore
            axs[1, 1].set_title("WM mask")                      # type: ignore
            axs[1, 2].set_title("CSF mask")                     # type: ignore

            # Remove axes
            axs[0, 0].axis('off')                               # type: ignore
            axs[0, 1].axis('off')                               # type: ignore
            axs[0, 2].axis('off')                               # type: ignore
            axs[1, 0].axis('off')                               # type: ignore
            axs[1, 1].axis('off')                               # type: ignore
            axs[1, 2].axis('off')                               # type: ignore

            plt.savefig(os.path.join(slice_dir, "vis_inspection.png"))
            plt.close()

        print(f"\033[FProcessing subject #{subject_i}... Done")

    print("------------------------------\n\nExtraction finished. Exiting.")


if __name__ == "__main__":
    # Run the extraction
    main(
        os.path.join(root, "data", "raw"),
        os.path.join(root, "data", "processed")
    )

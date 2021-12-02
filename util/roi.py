"""Module used to store roi-related utilities

They are e.g. used to perform an ROI determination
and extraction.
"""

import os
import cv2
from typing import Union
import numpy as np


def generate_rois(
        image: np.ndarray,
        roi_path: Union[str, bytes, os.PathLike],
        overwrite: bool = True) -> np.ndarray:
    """Function for defining and storing an ROI based
    on a single image.

    Parameters
    ----------
    image : np.ndarray
        Numpy array of image to be viewed
    roi_path : str | bytes | os.PathLike
        Path in which we will store the ROI
    overwrite : bool
        Whether to overwrite existing ROI data

    Returns
    -------
    rois : np.ndarray
        Numpy array of edge points for all ROIs
    """

    # If already there and not overwrite, skip this function.
    if not overwrite and os.path.exists(roi_path):
        # Print skipping message
        print(f"ROI data found at {roi_path}. Skipping...")

        # Load and return previously defined ROIs
        rois = np.load(roi_path)
        return rois

    else:
        # Print short explanation message
        print(
            "Please draw the required ROI(s) on the open window\n"
            "-----------------------------------"
        )

        # define ROIs
        image_rgb = np.moveaxis(np.array([image] * 3), 0, -1)
        rois = cv2.selectROIs(
            "ROI selection", image_rgb,
            showCrosshair=True, fromCenter=False
        )

        # Print some info
        print(
            "-----------------------------------\n"
            f"Defined {np.size(rois) // 4} ROIs succesfully!"
        )

        # Save and return ROIs
        np.save(roi_path, rois)
        return rois


def extract_rois(
        image: np.ndarray,
        rois: np.ndarray) -> list[np.ndarray]:
    """Extract the appropriate parts of the image
    by using previously defined ROIs.

    Parameters
    ----------
    image: np.ndarray
        Image to be analyzed
    rois: np.ndarray
        Array of corner points for ROIs to be extracted

    Returns
    -------
    roi_images : list[np.ndarray]
        List of ROI images. Likely all different sizes.
    """

    # Define ROI list
    roi_images = []

    # Loop over ROI definitions
    for i in range(np.shape(rois)[0]):
        # Extract current ROI info
        r = rois[i]
        # Crop ROI image
        roi_image = image[
            int(r[1]):int(r[1] + r[3]),
            int(r[0]):int(r[0] + r[2])
        ]
        # Append ROI list
        roi_images.append(roi_image)

    # Return list
    return roi_images

"""Module used to store roi-related utilities

They are e.g. used to perform an ROI determination
and extraction.
"""

import matplotlib

# Using TkAgg framework for SSH funcitonality.
# If this gives errors, reset the terminal.
matplotlib.use('TkAgg')

import os                                           # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.widgets import RectangleSelector    # noqa: E402
from typing import Union                            # noqa: E402
import numpy as np                                  # noqa: E402


class ROISelector(object):
    def __init__(self, image: np.ndarray):
        """Construct object and attributes"""

        # Load image and construct roi list
        self.image = image
        self.roi_list = []
        self.patch_list = []

        # Plot image and setup selector
        self.init_selector()

    def init_selector(self):
        """Constructs the figure and selector"""

        # Construct figure
        self.fig, self.ax = plt.subplots(num='ROI selector')

        # Setup figure name and decoration
        self.ax.set_title(
            'Add ROI: [Enter | Space]; Remove ROI: Backspace; Finalize: Esc'
        )
        self.ax.set_xlabel(
            'Number of ROIs: 0'
        )
        # Setup events
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        # Setup selector
        rs = RectangleSelector(
            self.ax, self.line_select_callback,
            drawtype='box', useblit=False, button=[1],
            minspanx=5, minspany=5, spancoords='pixels',
            interactive=True)

        # Display image
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.imshow(
            np.array(self.image, dtype=float),
            cmap='gray'
        )

        # Show figure
        plt.show()

    def line_select_callback(self, eclick, erelease):
        """Handle dragging behaviour"""
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata

    def on_press(self, event):
        """Handle key-press"""

        # Store current selection in ROI list
        if event.key in [' ', 'enter']:
            # Draw ROI
            self.patch_list.append(plt.Rectangle(
                (min(self.x1, self.x2), min(self.y1, self.y2)),
                np.abs(self.x1 - self.x2), np.abs(self.y1 - self.y2),
                linewidth=1, edgecolor='r', facecolor='none'))
            self.ax.add_patch(self.patch_list[-1])
            # Add ROI to list
            self.roi_list.append(np.array(
                [self.x1, self.x2, self.y1, self.y2]
            ))
            # Update label
            self.ax.set_xlabel(
                f'Number of ROIs: {len(self.roi_list)}'
            )
            # Update plot
            self.fig.canvas.draw()
        # Remove previous selection from ROI list
        elif event.key == 'backspace':
            # Check whether roi_list isn't empty
            if len(self.roi_list) > 0:
                # Remove patch from image
                self.patch_list[-1].remove()
                # Remove ROI from lists
                self.roi_list.pop(-1)
                self.patch_list.pop(-1)
                # Update label
                self.ax.set_xlabel(
                    f'Number of ROIs: {len(self.roi_list)}'
                )
                # Update plot
                self.fig.canvas.draw()
        # Close window and return ROIs
        elif event.key == 'escape':
            plt.close(self.fig)
        else:
            pass


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
            "Please draw the required ROI(s) on the open window"
        )

        # define ROIs
        roi_selector = ROISelector(image)
        rois = np.array(roi_selector.roi_list)

        # Print some info
        print(
            f"Defined {np.size(rois) // 4} ROIs succesfully!"
        )

        # Save and return ROIs
        np.save(roi_path, rois)
        return rois


def extract_rois(
        image: np.ndarray,
        rois: np.ndarray) -> np.ndarray:
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
    return np.array(roi_images)

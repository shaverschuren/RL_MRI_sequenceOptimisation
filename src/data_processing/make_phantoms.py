"""Simple script to make some mock-up 2D data"""

import os

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)

import numpy as np      # noqa: 402

# Define resolution
n = 64

# Define parameters used for disk construction
a = n // 2  # x-coordinate of centre
b = n // 2  # y-coordinate of centre
r = n // 4  # radius of disk

# Define grid with coordinates as data
y, x = np.ogrid[-a: n - a, -b: n - b]

# Return boolean array (either within radius or not)
mask = np.array(x * x + y * y <= r * r, dtype=np.uint0)

# Generate np arrays of T1, T2, PD
T1 = mask * 0.600
T2 = mask * 0.040
PD = mask * 1.0

# Create dir if applicable
if not os.path.isdir(os.path.join(root, "tmp")):
    os.mkdir(os.path.join(root, "tmp"))

# Save maps and masks
np.save(os.path.join(root, "tmp", "T1.npy"), T1)
np.save(os.path.join(root, "tmp", "T2.npy"), T2)
np.save(os.path.join(root, "tmp", "PD.npy"), PD)
np.save(os.path.join(root, "tmp", "mask_1.npy"), mask)
np.save(os.path.join(root, "tmp", "mask_2.npy"), 1. - mask)

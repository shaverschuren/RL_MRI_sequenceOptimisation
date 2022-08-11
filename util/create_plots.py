"""Module used for creating plots for the paper from .csv files"""

import os
import sys

if __name__ == '__main__':
    # Add root to path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path: sys.path.append(root)

    # Setup directories we wish to get the data from
    a_snr_train_dir = "logs/final_a_snr_rdpg"
    a_snr_test_dir = "logs/final_a_snr_rdpg_scan"
    a_cnr_train_dir = "logs/final_a_cnr_rdpg"
    a_cnr_test_dir = "logs/final_a_snr_rdpg_scan"
    b_train_dir = "logs/final_b_cnr"
    b_test_dir = "logs/final_b_cnr"

    # Setup output directory for our figures
    to_dir = "tmp/figures_for_publication"
    if not os.path.isdir(to_dir): os.mkdir(to_dir)

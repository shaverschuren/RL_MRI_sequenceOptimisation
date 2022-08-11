"""Module used for creating plots for the paper from .csv files"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tb2csv


def generate_plots(dirs: dict[str, str]):
    """Generate plots"""

    # Training curve A - SNR
    df_a_snr_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "a_snr_train", "train_episodes.csv")
    )

    # Training curve A - CNR
    df_a_cnr_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "a_cnr_train", "train_episodes.csv")
    )

    # Training curve B
    df_b_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_train", "train_episodes.csv")
    )

    # Test curves A - SNR

    # Test curves A - CNR


def extract_tb_logs(dirs: dict[str, str]):
    """Extract appropriate log files"""

    # Define dirnames
    dirnames = [
        "a_snr_train", "a_snr_test", "a_cnr_train",
        "a_cnr_test", "b_train", "b_test"
    ]

    # Extract proper event files
    tb2csv.main(
        [dirs[dirname] for dirname in dirnames],
        [os.path.join(dirs["to_dir"], dirname) for dirname in dirnames]
    )


if __name__ == '__main__':
    # Add root to path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path: sys.path.append(root)

    # Setup directories we wish to get the data from
    dirs = {
        "a_snr_train": "logs/final_a_snr_rdpg",
        "a_snr_test": "logs/final_a_snr_rdpg_scan",
        "a_cnr_train": "logs/final_a_cnr_rdpg",
        "a_cnr_test": "logs/final_a_snr_rdpg_scan",
        "b_train": "logs/final_b_cnr",
        "b_test": "logs/final_b_cnr",
        "to_dir": "tmp/figures_for_publication"
    }

    # Setup output directory for our figures
    if not os.path.isdir(dirs["to_dir"]): os.mkdir(dirs["to_dir"])

    # Extract tensorboard event files
    extract_tb_logs(dirs)

    # Generate plots
    generate_plots(dirs)

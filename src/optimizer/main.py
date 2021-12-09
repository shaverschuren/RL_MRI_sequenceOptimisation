"""Main launcher for RL-based optimizers"""

import argparse


if __name__ == "__main__":

    # Setup arg parser
    parser = argparse.ArgumentParser(
        description="RL-based MRI sequence design optimizer"
    )

    # Add arguments
    parser.add_argument(
        "--mode", default="snr", type=str,
        help="Available modes: snr/cnr"
    )
    parser.add_argument(
        "--env", default="mri", type=str,
        help="Type of environment. Available environments: mri/epg"
    )
    parser.add_argument(
        "--agent", default="ddpg", type=str,
        help="Type of optimizer agent. Available agents: dqn/ddpg/rdpg"
    )

    # Parse arguments
    args = parser.parse_args()

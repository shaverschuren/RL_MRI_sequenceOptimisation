"""Main launcher for scanner simulator"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
import argparse                             # noqa: E402
from scanner_simulator import simulator     # noqa: E402


def parse_args():
    """Parse arguments for this script"""

    # Setup arg parser
    parser = argparse.ArgumentParser(
        description="MRI Scanner simulator for single or double phantom"
    )
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    # Add arguments
    required.add_argument(
        "-n", "--n_phantoms", type=int, required=True,
        help="How many phantoms to simulate. Available: 1/2"
    )
    optional.add_argument(
        "--resolution", default=256, type=int,
        help="Resolution of the simulation (NxN). Default: 256"
    )
    optional.add_argument(
        "--T1a", default=0.500, type=float,
        help="T1 to simulate phantom with [s]. Default: 0.500"
    )
    optional.add_argument(
        "--T2a", default=0.050, type=float,
        help="T1 to simulate phantom with [s]. Default: 0.050"
    )
    optional.add_argument(
        "--T1b", default=1.000, type=float,
        help="T1 to simulate second phantom with [s]. Default: 1.000"
    )
    optional.add_argument(
        "--T2b", default=0.050, type=float,
        help="T2 to simulate second phantom with [s]. Default: 0.050"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check argument validity
    if args.n_phantoms not in [1, 2]:
        raise ValueError(
            "The 'mode' argument options are "
            f"(1, 2), but got '{args.n_phantoms}'."
        )

    # Return arguments
    return args


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Initialize simulator
    sim = simulator.Simulator(
        config_path=os.path.join(root, "config.json"),
        n_phantoms=args.n_phantoms,
        resolution=args.resolution,
        T1a=args.T1a, T2a=args.T2a,
        T1b=args.T1b, T2b=args.T2b
    )

    # Run simulator
    sim.run()

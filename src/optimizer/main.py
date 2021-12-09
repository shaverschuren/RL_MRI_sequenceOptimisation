"""Main launcher for RL-based optimizers"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
import argparse                 # noqa: E402
import algorithms               # noqa: E402
import environments             # noqa: E402


def parse_args():
    """Parse arguments for this script"""

    # Setup arg parser
    parser = argparse.ArgumentParser(
        description="RL-based MRI sequence design optimizer"
    )

    # Add arguments
    parser.add_argument(
        "--mode", default="snr", type=str,
        help="Metric to optimize. Available: snr/cnr"
    )
    parser.add_argument(
        "--env", default="mri", type=str,
        help="Type of environment. Available: mri/epg"
    )
    parser.add_argument(
        "--agent", default="ddpg", type=str,
        help="Type of optimizer agent. Available: dqn/ddpg/rdpg/validator"
    )

    # Parse and return arguments
    return parser.parse_args()


def init_environment(args: argparse.Namespace):
    """Function used to select the appropriate environment for a run"""

    # Check for either scanner or simulation platform
    if args.env.lower() == "mri":
        platform = "scanner"
    elif args.env.lower() == "epg":
        platform = "simulation"
    else:
        raise RuntimeError(
            "Value of 'env' argument should be either 'epg' or 'mri'"
        )

    # Check for discrete or continuous action space
    if args.agent.lower() == "dqn":
        action_space = "discrete"
    elif args.agent.lower() in ["ddpg", "rdpg"]:
        action_space = "continuous"
    elif args.agent.lower() == "validation":
        action_space = None
    else:
        raise RuntimeError(
            "Value of 'agent' should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    # Check for either cnr or snr optimization
    if args.mode.lower() == "snr":
        mode = "snr"
    elif args.mode.lower() == "cnr":
        mode = "cnr"
    else:
        raise RuntimeError(
            "Value of 'mode' argument should be either 'snr' or 'cnr'"
        )

    # Check whether in validation mode or not
    validation_mode = (args.agent.lower() == "validation")

    # Initialize environment
    # TODO: Add action space, validation and mode
    if platform == "scanner":
        env = environments.ScannerEnv()
    elif platform == "simulation":
        env = environments.SimulationEnv()
    else:
        return

    return env


def init_optimizer(env, args: argparse.Namespace):
    """Function to select and initialize optimizer for this run"""

    # Select appropriate optimizer
    # TODO: Pass environment and arguments
    if args.agent.lower() == "dqn":
        optimizer = algorithms.DQN()
    elif args.agent.lower() == "ddpg":
        optimizer = algorithms.DDPG()
    elif args.agent.lower() == "rdpg":
        optimizer = algorithms.RDPG()
    else:
        raise RuntimeError(
            "Value of 'agent' argument should be in ('dqn', 'ddpg', 'rdpg')"
        )

    return optimizer


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Initialize environment
    env = init_environment(args)

    # Initialize optimizer
    optimizer = init_optimizer(env, args)

    # Run optimizer
    # TODO: optimizer.run()

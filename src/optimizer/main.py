"""Main launcher for RL-based optimizers"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
import argparse                                 # noqa: E402
from optimizer import algorithms, environments  # noqa: E402


def parse_args():
    """Parse arguments for this script"""

    # Setup arg parser
    parser = argparse.ArgumentParser(
        description="RL-based MRI sequence design optimizer"
    )

    # Add arguments
    parser.add_argument(
        "--metric", default="snr", type=str,
        help="Metric to optimize. Available: snr/cnr"
    )
    parser.add_argument(
        "--platform", default="scanner", type=str,
        help="Type of platform/environment. Available: scanner/sim"
    )
    parser.add_argument(
        "--agent", default="ddpg", type=str,
        help="Type of optimizer agent. Available: dqn/ddpg/rdpg/validator"
    )
    parser.add_argument(
        "--mode", default="both", type=str,
        help="Mode to operate in. Available: train/test/both"
    )
    parser.add_argument(
        "--pretrained_path", type=str,
        help="Optional: Path to pretrained model"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check argument validity
    args.metric = args.metric.lower()
    if args.metric not in ["snr", "cnr"]:
        raise ValueError(
            "The 'metric' argument options are ('snr', 'cnr'), "
            f"but got '{args.metric}'."
        )
    args.platform = args.platform.lower()
    if args.platform not in ["scanner", "sim"]:
        raise ValueError(
            "The 'platform' argument options are ('scanner', 'sim'), "
            f"but got '{args.platform}'."
        )
    args.agent = args.agent.lower()
    if args.agent not in ['dqn', 'ddpg', 'rdpg', 'validator']:
        raise ValueError(
            "The 'agent' argument options are "
            "('dqn', 'ddpg', 'rdpg', 'validator'), "
            f"but got '{args.agent}'."
        )
    args.mode = args.mode.lower()
    if args.mode not in ['train', 'test', 'both']:
        raise ValueError(
            "The 'mode' argument options are ('train', 'test', 'both'), "
            f"but got '{args.mode}'."
        )
    if args.pretrained_path:
        if not os.path.exists(args.pretrained_path):
            raise ValueError(
                "The file passed in argument 'pretrained_path' doesn't exist"
                f"\nGot '{args.pretrained_path}'"
            )

    # Return arguments
    return args


def init_paths():
    """Setup some required paths for this software"""

    # Extract global vars
    global args, root, src

    # Setup root, src
    args.root = root
    args.src = src

    # Setup log_dir
    args.log_dir = os.path.join(
        args.root, "logs", f"{args.platform}_{args.metric}_{args.agent}"
    )
    # Setup config path
    args.config_path = os.path.join(args.root, "config.json")


def init_environment(args: argparse.Namespace):
    """Function used to select the appropriate environment for a run"""

    # Check for discrete or continuous action space
    if args.agent == "dqn":
        action_space_type = "discrete"
    elif args.agent in ["ddpg", "rdpg"]:
        action_space_type = "continuous"
    elif args.agent == "validation":
        raise NotImplementedError()
    else:
        raise ValueError(
            "Value of 'agent' should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    # Check for recurrent or non-recurrent model
    if args.agent in ["dqn", "ddpg"]:
        recurrent_model = False
    elif args.agent == "rdpg":
        recurrent_model = True
    elif args.agent == "validation":
        raise NotImplementedError()
    else:
        raise ValueError(
            "Value of 'agent' should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    # Check whether in validation mode or not
    validation_mode = (args.agent.lower() == "validation")

    # Initialize environment
    if args.platform == "scanner":
        env = environments.ScannerEnv(
            config_path=args.config_path,
            log_dir=args.log_dir,
            metric=args.metric, action_space_type=action_space_type,
            recurrent_model=recurrent_model
        )
    elif args.platform == "sim":
        env = environments.SimulationEnv(
            mode=args.metric, action_space_type=action_space_type,
            recurrent_model=recurrent_model
        )
    else:
        raise RuntimeError(
            "This shouldn't happen"
        )

    return env


def init_optimizer(env, args: argparse.Namespace):
    """Function to select and initialize optimizer for this run"""

    # Select appropriate optimizer
    if args.agent == "dqn":
        optimizer = algorithms.DQN(
            env=env, log_dir=args.log_dir,
            n_episodes=750 if args.metric == "snr" else 2500,
            pretrained_path=args.pretrained_path
        )
    elif args.agent == "ddpg":
        optimizer = algorithms.DDPG(
            env=env, log_dir=args.log_dir,
            n_episodes=1500 if args.metric == "snr" else 2500,
            pretrained_path=args.pretrained_path
        )
    elif args.agent == "rdpg":
        optimizer = algorithms.RDPG()
    elif args.agent == "validation":
        optimizer = algorithms.Validator()
    else:
        raise ValueError(
            "Value of 'agent' argument should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    return optimizer


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Setup paths
    init_paths()

    # Initialize environment
    env = init_environment(args)

    # Initialize optimizer
    optimizer = init_optimizer(env, args)

    # Run optimizer training and testing
    if args.mode in ["train", "both"]:
        # Run training loop
        optimizer.run(train=True)
    if args.mode in ["test", "both"]:
        # Run testing loop
        optimizer.run(train=False)

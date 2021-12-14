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

    # Parse and return arguments
    return parser.parse_args()


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

    # Check for either scanner or simulation platform
    if args.platform.lower() == "scanner":
        platform = "scanner"
    elif args.platform.lower() == "sim":
        platform = "sim"
    else:
        raise RuntimeError(
            "Value of 'env' argument should be either 'epg' or 'mri'"
        )

    # Check for discrete or continuous action space
    if args.agent.lower() == "dqn":
        action_space_type = "discrete"
    elif args.agent.lower() in ["ddpg", "rdpg"]:
        action_space_type = "continuous"
    elif args.agent.lower() == "validation":
        raise NotImplementedError()
    else:
        raise RuntimeError(
            "Value of 'agent' should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    # Check for recurrent or non-recurrent model
    if args.agent.lower() in ["dqn", "ddpg"]:
        recurrent_model = False
    elif args.agent.lower() == "rdpg":
        recurrent_model = True
    elif args.agent.lower() == "validation":
        raise NotImplementedError()
    else:
        raise RuntimeError(
            "Value of 'agent' should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    # Check for either cnr or snr optimization
    if args.metric.lower() == "snr":
        metric = "snr"
    elif args.metric.lower() == "cnr":
        metric = "cnr"
    else:
        raise RuntimeError(
            "Value of 'mode' argument should be either 'snr' or 'cnr'"
        )

    # Check whether in validation mode or not
    validation_mode = (args.agent.lower() == "validation")

    # Initialize environment
    if platform == "scanner":
        env = environments.ScannerEnv(
            config_path=args.config_path,
            log_dir=args.log_dir,
            metric=args.metric, action_space_type=action_space_type,
            recurrent_model=recurrent_model
        )
    elif platform == "sim":
        env = environments.SimulationEnv(
            mode=metric, action_space_type=action_space_type,
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
    if args.agent.lower() == "dqn":
        optimizer = algorithms.DQN(
            env=env, log_dir=args.log_dir,
            n_episodes=750 if args.metric == "snr" else 2500
        )
    elif args.agent.lower() == "ddpg":
        optimizer = algorithms.DDPG()
    elif args.agent.lower() == "rdpg":
        optimizer = algorithms.RDPG()
    elif args.agent.lower() == "validation":
        optimizer = algorithms.Validator()
    else:
        raise RuntimeError(
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

"""Main launcher for RL-based optimizers"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
print("Importing dependencies... ", end="", flush=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'            # Remove tensorflow verbose

import time                                         # noqa: E402
start = time.time()                                 # Start timer

import argparse                                     # noqa: E402
from optimizer import algorithms, environments      # noqa: E402

print(f"Took {time.time() - start:.2f} seconds")    # Print timer results


def parse_args():
    """Parse arguments for this script"""

    # Setup arg parser
    parser = argparse.ArgumentParser(
        description="RL-based MRI sequence design optimizer"
    )
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    # Add arguments
    required.add_argument(
        "--metric", "-m", type=str, required=True,
        help="Metric to optimize. Available: snr/cnr"
    )
    required.add_argument(
        "--platform", "-p", type=str, required=True,
        help="Type of platform/environment. Available: scan/epg"
    )
    required.add_argument(
        "--agent", "-a", type=str, required=True,
        help="Type of optimizer agent. Available: dqn/ddpg/rdpg/validation"
    )
    required.add_argument(
        "--mode", "-mo", type=str, required=True,
        help="Mode to operate in. Available: train/test/both/validation"
    )
    optional.add_argument(
        "--pretrained_path", type=str,
        help="Optional: Path to pretrained model"
    )
    optional.add_argument(
        "--episodes", type=int,
        help="Optional: Override the number of episodes to train with"
    )
    optional.add_argument(
        "--suppress_done", type=bool, default=False,
        help="Optional: Override the 'done' signal given by the model"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if all required parameters are provided
    if not args.metric: raise ValueError("'metric' argument is required")
    if not args.platform: raise ValueError("'platform' argument is required")
    if not args.agent: raise ValueError("'agent' argument is required")
    if not args.mode: raise ValueError("'mode' argument is required")

    # Check argument validity
    args.metric = args.metric.lower()
    if args.metric not in ["snr", "cnr"]:
        raise ValueError(
            "The 'metric' argument options are ('snr', 'cnr'), "
            f"but got '{args.metric}'."
        )
    args.platform = args.platform.lower()
    if args.platform not in ["scan", "epg"]:
        raise ValueError(
            "The 'platform' argument options are ('scan', 'epg'), "
            f"but got '{args.platform}'."
        )
    args.agent = args.agent.lower()
    if args.agent not in ['dqn', 'ddpg', 'rdpg', 'validation']:
        raise ValueError(
            "The 'agent' argument options are "
            "('dqn', 'ddpg', 'rdpg', 'validator'), "
            f"but got '{args.agent}'."
        )
    args.mode = args.mode.lower()
    if args.mode not in ['train', 'test', 'both', 'validation']:
        raise ValueError(
            "The 'mode' argument options are "
            "('train', 'test', 'both', 'validation'), "
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
        action_space_type = None
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
        recurrent_model = None
    else:
        raise ValueError(
            "Value of 'agent' should be in "
            "('dqn', 'ddpg', 'rdpg', 'validation')"
        )

    # Check whether in validation mode or not
    validation_mode = (args.agent == "validation")

    # Initialize environment
    if args.platform == "scan":
        env = environments.ScannerEnv(
            config_path=args.config_path,
            log_dir=args.log_dir,
            metric=args.metric, action_space_type=action_space_type,
            model_done=not args.suppress_done,
            recurrent_model=recurrent_model,
            validation_mode=validation_mode
        )
    elif args.platform == "epg":
        env = environments.SimulationEnv(
            mode=args.metric, action_space_type=action_space_type,
            model_done=not args.suppress_done,
            recurrent_model=recurrent_model,
            lock_material_params=validation_mode,
            validation_mode=validation_mode
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
        # Set n_epochs
        if args.episodes:
            n_episodes = args.episodes
        else:
            n_episodes = 750 if args.metric == "snr" else 2500
        # Define optimizer
        optimizer = algorithms.DQN(
            env=env, log_dir=args.log_dir,
            n_episodes=n_episodes,
            pretrained_path=args.pretrained_path
        )
    elif args.agent == "ddpg":
        # Set n_epochs
        if args.episodes:
            n_episodes = args.episodes
        else:
            n_episodes = 1500 if args.metric == "snr" else 2500
        # Define optimizer
        optimizer = algorithms.DDPG(
            env=env, log_dir=args.log_dir,
            n_episodes=n_episodes,
            pretrained_path=args.pretrained_path
        )
    elif args.agent == "rdpg":
        # Set n_epochs
        if args.episodes:
            n_episodes = args.episodes
        else:
            n_episodes = 3000   # if args.metric == "snr" else 5000
        # Define optimizer
        optimizer = algorithms.RDPG(
            env=env, log_dir=args.log_dir,
            n_episodes=n_episodes,
            pretrained_path=args.pretrained_path
        )
    elif args.agent == "validation":
        optimizer = algorithms.Validator(
            env=env, log_dir=args.log_dir
        )
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

    if args.mode == "validation":
        # Run validation loop
        print("Starting validation run")
        optimizer.run()
    else:
        # Run optimizer training and testing
        if args.mode in ["train", "both"]:
            # Run training loop
            optimizer.run(train=True)
        if args.mode in ["test", "both"]:
            # Run testing loop
            optimizer.run(train=False)

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

import gym                                          # noqa: E402
from optimizer import algorithms                    # noqa: E402

print(f"Took {time.time() - start:.2f} seconds")    # Print timer results


# def init_environment(args: argparse.Namespace):
#     """Function used to select the appropriate environment for a run"""

#     # Check for discrete or continuous action space
#     if args.agent == "dqn":
#         action_space_type = "discrete"
#     elif args.agent in ["ddpg", "rdpg"]:
#         action_space_type = "continuous"
#     elif args.agent == "validation":
#         action_space_type = None
#     else:
#         raise ValueError(
#             "Value of 'agent' should be in "
#             "('dqn', 'ddpg', 'rdpg', 'validation')"
#         )

#     # Check for recurrent or non-recurrent model
#     if args.agent in ["dqn", "ddpg"]:
#         recurrent_model = False
#     elif args.agent == "rdpg":
#         recurrent_model = True
#     elif args.agent == "validation":
#         recurrent_model = None
#     else:
#         raise ValueError(
#             "Value of 'agent' should be in "
#             "('dqn', 'ddpg', 'rdpg', 'validation')"
#         )

#     # Check whether in validation mode or not
#     validation_mode = (args.agent == "validation")

#     # Determine fa range
#     if args.metric == "snr": fa_range = [3., 30.]
#     elif args.metric == "cnr": fa_range = [10., 40.]
#     else: raise RuntimeError()

#     # Initialize environment
#     if args.platform == "scan":
#         env = environments.ScannerEnv(
#             config_path=args.config_path,
#             log_dir=args.log_dir,
#             fa_range=fa_range,
#             metric=args.metric, action_space_type=action_space_type,
#             model_done=not args.suppress_done,
#             recurrent_model=recurrent_model,
#             validation_mode=validation_mode
#         )
#     elif args.platform == "epg":
#         env = environments.SimulationEnv(
#             mode=args.metric, fa_range=fa_range,
#             action_space_type=action_space_type,
#             model_done=not args.suppress_done,
#             recurrent_model=recurrent_model,
#             lock_material_params=validation_mode,  # TODO: True
#             validation_mode=validation_mode
#         )
#     else:
#         raise RuntimeError(
#             "This shouldn't happen"
#         )

#     return env


# def init_optimizer(env, args: argparse.Namespace):
#     """Function to select and initialize optimizer for this run"""

#     # Select appropriate optimizer
#     if args.agent == "dqn":
#         # Set n_epochs
#         if args.episodes:
#             n_episodes = args.episodes
#         else:
#             n_episodes = 750 if args.metric == "snr" else 2500
#         # Define optimizer
#         optimizer = algorithms.DQN(
#             env=env, log_dir=args.log_dir,
#             n_episodes=n_episodes,
#             pretrained_path=args.pretrained_path
#         )
#     elif args.agent == "ddpg":
#         # Set n_epochs
#         if args.episodes:
#             n_episodes = args.episodes
#         else:
#             n_episodes = 1500 if args.metric == "snr" else 2500
#         # Define optimizer
#         optimizer = algorithms.DDPG(
#             env=env, log_dir=args.log_dir,
#             n_episodes=n_episodes,
#             pretrained_path=args.pretrained_path
#         )
#     elif args.agent == "rdpg":
#         # Set n_epochs
#         if args.episodes:
#             n_episodes = args.episodes
#         else:
#             n_episodes = 2000   # if args.metric == "snr" else 5000
#         # Define optimizer
#         optimizer = algorithms.RDPG(
#             env=env, log_dir=args.log_dir,
#             n_episodes=n_episodes,
#             model_done=not args.suppress_done,
#             pretrained_path=args.pretrained_path
#         )
#     elif args.agent == "validation":
#         optimizer = algorithms.Validator(
#             env=env, log_dir=args.log_dir
#         )
#     else:
#         raise ValueError(
#             "Value of 'agent' argument should be in "
#             "('dqn', 'ddpg', 'rdpg', 'validation')"
#         )

#     return optimizer


if __name__ == "__main__":

    mode = "both"

    # Initialize environment
    env = gym.make('MountainCarContinuous-v0')

    # Initialize optimizer
    optimizer = algorithms.RDPG(
        env, log_dir=os.path.join(root, "logs", "rdpg_testing"),
        n_episodes=500
    )

    if mode == "validation":
        # Run validation loop
        print("Starting validation run")
        optimizer.run()
    else:
        # Run optimizer training and testing
        if mode in ["train", "both"]:
            # Run training loop
            optimizer.run(train=True)
        if mode in ["test", "both"]:
            # Run testing loop
            optimizer.run(train=False)

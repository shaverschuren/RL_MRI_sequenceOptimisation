"""Module implementing several RL algorithms

Here, we give implementations of several algorithms, namely:

- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- RDPG (Recurrent Deterministic Policy Gradient)
"""

import os
from typing import Union
import numpy as np
from datetime import datetime
import torch
from optimizer import agents
from util import training, loggers


class DQN(object):
    """Class to represent a DQN optimizer algorithm"""

    def __init__(
            self,
            env,
            log_dir: Union[str, os.PathLike],
            n_episodes: int = 500,
            n_ticks: int = 30,
            batch_size: int = 128,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            env : optimizer.environments.* object
                Environment to be optimized
            log_dir : str | os.PathLike
                Directory in which we store the logs
            n_episodes : int
                Number of episodes we'll run
            n_ticks : int
                Maximum number of ticks in an episode
            batch_size: int
                Batch size used for optimization
        """

        # Build attributes
        self.env = env
        self.log_dir = log_dir
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Setup agent
        self.agent = agents.DQNAgent(env.n_states, env.n_actions)
        # Setup memory
        self.memory = training.LongTermMemory(10000)
        # Setup logger
        self.setup_logger()

    def log_step(self):
        """Log a single step"""

        run_type = "train" if self.train else "test"

        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.fa),
            step=self.tick
        )
        self.logger.log_scalar(
            field="fa_norm",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.fa_norm),
            step=self.tick
        )
        self.logger.log_scalar(
            field=self.env.metric,
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(getattr(self.env, self.env.metric)),
            step=self.tick
        )
        self.logger.log_scalar(
            field="done",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.done),
            step=self.tick
        )

    def log_episode(self):
        """Log a single episode"""

        # Find optimal fa and snr/cnr
        optimal_fa = self.env.optimal_fa
        optimal_metric = getattr(self.env, f"optimal_{self.env.metric}")

        # Extract recent memory
        recent_memory = self.memory.get_recent_memory(5)
        recent_states = [transition.next_state for transition in recent_memory]
        recent_metrics = [float(state[0]) for state in recent_states]
        recent_fa = [float(state[1]) for state in recent_states]

        # Find "best" fa/metric in recent memory
        best_idx = np.argmax(recent_metrics)
        best_metric = recent_metrics[best_idx]
        best_fa = recent_fa[best_idx]

        # Calculate relative SNR/CNR error
        if best_metric == 0.:
            relative_error = 1.
        else:
            relative_error = abs(
                optimal_metric - best_metric
            ) / best_metric

        # Log scalars
        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_train_episodes",
            value=best_fa,
            step=self.episode
        )
        self.logger.log_scalar(
            field=self.env.metric,
            tag=f"{self.logs_tag}_train_episodes",
            value=best_metric,
            step=self.episode
        )
        self.logger.log_scalar(
            field="error",
            tag=f"{self.logs_tag}_train_episodes",
            value=min(relative_error, 1.),
            step=self.episode
        )

    def setup_logger(self):
        """Sets up logger and appropriate directories"""

        # Create logs dir if not already there
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # Generate logs file path and store tag
        now = datetime.now()
        logs_dirname = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
        self.logs_tag = logs_dirname
        self.logs_path = os.path.join(self.log_dir, logs_dirname)

        # Setup model checkpoint path
        self.model_path = os.path.join(self.logs_path, "model.pt")

        # Define datafields
        self.logs_fields = [
            "fa", "fa_norm", self.env.metric, "error", "done", "epsilon"
        ]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def run(self, train=True):
        """Run either training or testing loop"""

        # Set training variable
        self.train = train

        # Print some info
        print(
            "\n\n==================== "
            f"Running {'training' if train else 'test'} loop"
            " ====================\n\n"
        )

        # Create training initial condition distributions
        if train:
            self.env.n_episodes = self.n_episodes
            self.env.homogeneous_initialization = True
            self.env.set_homogeneous_dists()
        else:
            self.env.n_episodes = None
            self.env.homogeneous_initialization = False

        # Episode loop
        for self.episode in range(self.n_episodes) if train else range(10):

            # Reset environment
            self.env.reset()

            # Print some info
            print(
                "\n========== "
                f"Episode {self.episode + 1:3d}/{self.n_episodes:3d}"
                " ==========\n"
            )

            # Loop over ticks/steps
            for self.tick in range(self.n_ticks):

                # Extract current state
                state = self.env.state

                # Choose action
                action = self.agent.select_action(state)

                # Simulate step
                next_state, reward, done = self.env.step(action)

                # Add to memory
                self.memory.push(
                    state, action, reward, next_state, done
                )

                # If training, update model
                if train and self.batch_size <= len(self.memory):
                    batch = self.memory.sample(self.batch_size)
                    self.agent.update(batch)

                # Print some info
                print(
                    f"Step {self.tick + 1:3d}/{self.n_ticks:3d} - "
                    f"Action: {int(action):2d} - "
                    f"FA: {float(next_state[1]) * 180.:5.1f} - "
                    f"{self.env.metric.upper()}: {float(next_state[0]):5.2f} -"
                    f" Reward: {float(reward):5.2f}"
                )

                # Log step results
                self.log_step()

                # Check if done
                if done:
                    break

            # Log episode results
            if train:
                self.log_episode()

            # Backup model
            self.agent.save(self.model_path)


class DDPG(object):
    """Class to represent a DDPG optimizer"""

    def __init__(self):
        """Initializes and builds the attributes for this class"""

        raise NotImplementedError()

    def run(self, train=True):
        """Run either training or testing loop"""

        pass


class RDPG(object):
    """Class to represent an RDPG optimizer"""

    def __init__(self):
        """Initializes and builds the attributes for this class"""

        raise NotImplementedError()

    def run(self, train=True):
        """Run either training or testing loop"""

        pass


class Validator(object):
    """Class to represent a validator algorithm"""

    def __init__(self):
        """Initializes and builds the attributes for this class"""

        raise NotImplementedError()

    def run(self, train=True):
        """Run either training or testing loop"""

        pass

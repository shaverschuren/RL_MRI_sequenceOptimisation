"""Module implementing several RL algorithms

Here, we give implementations of several algorithms, namely:

- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- RDPG (Recurrent Deterministic Policy Gradient)
"""

import os
from typing import Union
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
            n_episodes: int = 50,
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
            "fa", "fa_norm", "snr", "error", "done", "epsilon"
        ]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )


class DDPG(object):
    pass


class RDPG(object):
    pass


class Validator(object):
    pass

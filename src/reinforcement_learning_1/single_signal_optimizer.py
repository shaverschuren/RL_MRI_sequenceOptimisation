"""Single Signal Optimizer

This module implements a reinforcement model that
optimizes the flip angle for a single (simulated) tissue
to maximize signal."""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
from typing import Union                # noqa: E402
from collections import deque           # noqa: E402
import math                             # noqa: E402
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn as nn                   # noqa: E402
import torch.optim as optim             # noqa: E402
import torch.functional as F            # noqa: E402
from epg_code.python import epg         # noqa: E402


class SingleSignalOptimizer():
    """
    A class to represent an optimizer model for
    the flip angle to maximize the signal of a single
    simulated tissue.

        Parameters:

    """

    def __init__(
            self,
            n_episodes: int = 100,
            n_ticks: int = 50,
            batch_size: int = 64,
            fa_initial: float = 20.,
            fa_delta: float = 1.,
            gamma: float = 1.,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.995,
            alpha: float = 0.01,
            alpha_decay: float = 0.01,
            log_dir: Union[str, bytes, os.PathLike] =
            os.path.join(root, "logs", "model_1"),
            verbose: int = 1,
            device: Union[torch.device, None] = None):
        """Constructs model and attributes for this optimizer

            Parameters:
        """

        # Setup attributes
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size
        self.fa_initial = fa_initial
        self.fa_delta = fa_delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.log_dir = log_dir
        self.verbose = verbose
        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # Setup memory
        self.memory = deque(maxlen=100000)
        # Setup model
        self.init_model()

    def init_model(self):
        """Constructs reinforcement learning model

        Neural net: Fully connected 2-12-24-2
        Loss: L1 (MAE) Loss
        Optimizer: Adam with lr alpha and decay alpha_decay
        """

        # Construct neural net
        self.model = nn.Sequential(
            nn.Linear(2, 12),
            nn.Tanh(),
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, 2)
        )
        # Setup loss
        self.loss = nn.L1Loss()
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.alpha, weight_decay=self.alpha_decay
        )

    def remember(self, state, action, reward, next_state, done):
        """Update memory for this tick"""
        self.memory.append((state, action, reward, next_state, done))

    def get_epsilon(self, t):
        return max(
            self.epsilon_min,
            min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay))
        )

    def forward(self, x):
        """Forward run through model"""
        return self.model(x)

    def train(self):
        """Training loop"""
        pass


if __name__ == "__main__":
    optimizer = SingleSignalOptimizer()
    optimizer.train()

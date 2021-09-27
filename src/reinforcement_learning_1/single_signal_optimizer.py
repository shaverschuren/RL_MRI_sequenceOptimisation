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
from typing import Union                    # noqa: E402
from collections import namedtuple, deque   # noqa: E402
import math                                 # noqa: E402
import random                               # noqa: E402
import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import torch.nn as nn                       # noqa: E402
import torch.optim as optim                 # noqa: E402
import torch.functional as F                # noqa: E402
from epg_code.python import epg             # noqa: E402


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
        self.fa = fa_initial
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
        self.Transition = namedtuple(
            'Transition',
            ('state', 'action', 'next_state', 'reward')
        )
        # Setup environment
        self.init_env()
        # Setup model
        self.init_model()

    def init_env(self):
        """Constructs the environment

        Includes action space, [...]
        """

        self.action_space = np.array([0, 1])

    def init_model(self):
        """Constructs reinforcement learning model

        Neural net: Fully connected 2-12-24-2
        Loss: L1 (MAE) Loss
        Optimizer: Adam with lr alpha and decay alpha_decay
        """

        # Construct neural nets
        self.policy_net = nn.Sequential(
            nn.Linear(2, 12),
            nn.Tanh(),
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, 2)
        ).to(self.device)
        self.target_net = nn.Sequential(
            nn.Linear(2, 12),
            nn.Tanh(),
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, 2)
        ).to(self.device)
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.alpha, weight_decay=self.alpha_decay
        )

    def step(self, action):
        """Run step of the environment simulation

        - Perform selected action
        - Run EPG simulation
        - Update state
        - Update reward
        - Update done
        """

        # Adjust flip angle according to action
        if int(action) == 0:
            # Decrease flip angle
            self.fa -= self.fa_delta
            if self.fa < 0.0: self.fa = 0.0
        elif int(action) == 1:
            # Increase flip angle
            self.fa += self.fa_delta
            if self.fa > 360.0: self.fa = 360.0

        # Run simulation with updated parameters
        F0, _, _ = epg.epg_as_torch(
            500, self.fa, 10E-03, .583, 0.055, device=self.device
        )

        # Update state
        state = torch.tensor(
            [float(np.abs(F0.cpu()[-1])), self.fa],
            device=self.device
        )
        # Define reward
        reward = torch.tensor(
            [state[0]], device=self.device
        )
        # Define "done"
        done = False

        return state, reward, done

    def remember(self, state, action, reward, next_state, done):
        """Update memory for this tick"""
        self.memory.append((state, action, next_state, reward))

    def choose_action(self, state, epsilon):
        """Choose action

        Choose the action for this step.
        This is either random (exploration) or determined
        via the model (exploitation). This rate is determined by
        the epsilon parameter.
        """

        if np.random.random() <= epsilon:
            # Exploration (random choice)
            return torch.tensor(
                [np.random.choice(self.action_space)],
                device=self.device
            )
        else:
            # Exploitation (max expected reward)
            with torch.no_grad():
                return torch.tensor(
                    [torch.argmax(self.policy_net(state))],
                    device=self.device
                )

    def get_epsilon(self, t):
        return max(
            self.epsilon_min,
            min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay))
        )

    def optimize_model(self, batch_size):
        """Optimize model based on previous episode"""

        # Check whether memory is long enough
        if len(self.memory) < batch_size:
            return

        # Create batch
        transitions = random.sample(
            self.memory, min(len(self.memory), batch_size))

        batch = self.Transition(*zip(*transitions))

        # Omit final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # Split state, action and reward batches
        state_batch = torch.as_tensor([*batch.state], dtype=torch.double)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute state action values
        state_action_values = \
            self.policy_net(state_batch).gather(0, action_batch)

        # Compute next state values
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states).max().detach()

        # Compute the expected Q values
        expected_state_action_values = \
            (next_state_values * self.gamma) + reward_batch

        # Optimize the model
        self.optimizer.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values,
            expected_state_action_values  # .unsqueeze(1)
        )
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def run(self):
        """Run the training loop"""

        # Set initial state
        state = torch.tensor(
            [0.0, self.fa],
            device=self.device
        )

        # Loop over episodes
        for episode in range(self.n_episodes):
            # Reset done and tick counter
            done = False
            tick = 0

            # Loop over steps/ticks
            while tick < self.n_ticks - 1 and not done:
                # Choose action
                action = self.choose_action(state, self.get_epsilon(episode))
                # Simulate step
                next_state, reward, done = self.step(action)
                # Add to memory
                self.remember(state, action, reward, next_state, done)
                # Update state
                state = next_state
                # Update tick counter
                tick += 1

                # TODO: Remove. Added for debugging purposes
                print(state, action, len(self.memory))

            # Optimize model
            self.optimize_model(self.batch_size)

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    optimizer = SingleSignalOptimizer()
    optimizer.run()

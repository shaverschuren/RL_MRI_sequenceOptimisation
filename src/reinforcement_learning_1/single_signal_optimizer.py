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
import torch.nn.functional as F             # noqa: E402
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
            fa_initial: float = 15.,
            fa_delta: float = 0.5,
            gamma: float = 1.,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.995,
            alpha: float = 0.01,
            alpha_decay: float = 0.01,
            log_dir: Union[str, bytes, os.PathLike] =
            os.path.join(root, "logs", "model_1"),
            verbose: bool = True,
            device: Union[torch.device, None] = None):
        """Constructs model and attributes for this optimizer

            Parameters:
        """

        # Setup attributes
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size
        self.fa_initial = fa_initial
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

        # Construct policy net
        self.policy_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
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
            [state[0] * 100.], device=self.device
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
            if self.verbose: print("Exploration ", end="", flush=True)
            return torch.tensor(
                [np.random.choice(self.action_space)],
                device=self.device
            )
        else:
            # Exploitation (max expected reward)
            if self.verbose: print("Exploitation", end="", flush=True)
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

        # Split state, action and reward batches

        # States
        state_batch_list = []
        for tensor_i in range(len(batch.state)):
            state_batch_list.append(np.array(batch.state[tensor_i].cpu()))
        state_batch_np = np.array(state_batch_list)
        state_batch = torch.as_tensor(state_batch_np, device=self.device)
        # Next states
        next_state_batch_list = []
        for tensor_i in range(len(batch.next_state)):
            next_state_batch_list.append(
                np.array(batch.next_state[tensor_i].cpu())
            )
        next_state_batch_np = np.array(next_state_batch_list)
        next_state_batch = torch.as_tensor(
            next_state_batch_np, device=self.device
        )
        # Actions and rewards
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Loss
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_state_batch, reward_batch)

        policy_output = self.policy_net(state_batch)
        Q_expected = torch.gather(
            policy_output, dim=-1, index=action_batch.unsqueeze(1)
        )

        loss = F.mse_loss(Q_expected, Q_targets)

        # Perform optimisation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_q_targets(self, next_states, rewards):
        """Computes the Q targets we will compare to the predicted Q values"""

        # Calculate Q values for next states
        Q_targets_next = \
            self.policy_net(next_states).detach().max(1)[0].unsqueeze(1)

        # Calculate Q values for current states
        Q_targets_current = \
            rewards.unsqueeze(1) + self.gamma * Q_targets_next

        return Q_targets_current

    def run(self):
        """Run the training loop"""

        # Loop over episodes
        for episode in range(self.n_episodes):
            # Print some info
            if self.verbose:
                print(f"=== Episode {episode + 1:3d}/{self.n_episodes:3d} ===")
            # Reset done and tick counter
            done = False
            tick = 0
            # Set initial state
            self.fa = self.fa_initial + (np.random.random() * 20.0 - 10.0)
            state = torch.tensor(
                [0.0, self.fa],
                device=self.device
            )

            # Loop over steps/ticks
            while tick < self.n_ticks - 1 and not done:
                # Print some info
                print(f"Step {tick + 1:2d}/{self.n_ticks:2d} - ", end="")
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

                # Print some info
                print(
                    f" - Action: {int(action)}"
                    f" - FA: {float(state[1]):.1f}"
                    f" - Signal: {float(state[0]):.3f}"
                )

            # Optimize model
            self.optimize_model(self.batch_size)

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    optimizer = SingleSignalOptimizer()
    optimizer.run()

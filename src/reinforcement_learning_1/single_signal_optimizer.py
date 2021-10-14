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
from typing import Union                                    # noqa: E402
from collections import namedtuple, OrderedDict, deque      # noqa: E402
import random                                               # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn as nn                                       # noqa: E402
import torch.optim as optim                                 # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from epg_code.python import epg                             # noqa: E402


class SingleSignalOptimizer():
    """
    A class to represent an optimizer model for
    the flip angle to maximize the signal of a single
    simulated tissue.

        Parameters:

    """

    def __init__(
            self,
            n_episodes: int = 1000,             # TODO: For prototyping
            n_ticks: int = 10,                  # TODO: For prototyping
            batch_size: int = 8,                # TODO: For prototyping
            fa_initial: float = 25.,
            fa_delta: float = 1.0,
            gamma: float = 1.,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 5e-2,
            alpha: float = 0.005,
            target_update_period: int = 3,
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
        self.target_update_period = target_update_period
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
            ('state', 'action', 'next_state', 'reward', 'done')
        )
        # Setup environment
        self.init_env()
        # Setup model
        self.init_model()

    def init_env(self):
        """Constructs the environment

        Includes action space, [...]
        """

        self.action_space = np.array([0, 1])  # TODO: Maybe 3rd state

    def init_model(self):
        """Constructs reinforcement learning model

        Neural net: Fully connected 2-4-2
        Loss: L1 (MAE) Loss
        Optimizer: Adam with lr alpha (and decay alpha_decay*)
        * Removed for now
        """

        # Construct policy net
        self.prediction_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, 4)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4, 4)),
            ('relu2', nn.ReLU()),
            ('output', nn.Linear(4, 2))
        ])).to(self.device)
        # Construct target net
        self.target_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, 4)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4, 4)),
            ('relu2', nn.ReLU()),
            ('output', nn.Linear(4, 2))
        ])).to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.prediction_net.parameters(),
            lr=self.alpha  # , weight_decay=self.alpha_decay
        )

    def step(self, old_state, action):
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
            if self.fa > 180.0: self.fa = 180.0

        # Run simulation with updated parameters
        F0, _, _ = epg.epg_as_torch(
            100, self.fa, 50E-03, .500, 0.025, device=self.device
        )

        # Update state
        state = torch.tensor(
            [float(np.abs(F0.cpu()[-1])), self.fa],
            device=self.device
        )
        # Define reward
        # (as either +/- 1 for an increase or decrease in signal)
        reward_float = 1.0 if state[0] > old_state[0] else -1.0
        reward = torch.tensor(
            [reward_float], device=self.device
        )
        # Define "done"
        done = torch.tensor(0, device=self.device)

        return state, reward, done

    def remember(self, state, action, reward, next_state, done):
        """Update memory for this tick"""
        self.memory.append((state, action, next_state, reward, done))

    def choose_action(self, state, epsilon):
        """Choose action

        Choose the action for this step.
        This is either random (exploration) or determined
        via the model (exploitation). This rate is determined by
        the epsilon parameter.
        """

        if np.random.random() <= epsilon:
            # Exploration (random choice) -> *R*andom
            if self.verbose: print("R", end="", flush=True)
            return torch.tensor(
                [np.random.choice(self.action_space)],
                device=self.device
            )
        else:
            # Exploitation (max expected reward) -> *P*olicy
            if self.verbose: print("P", end="", flush=True)
            with torch.no_grad():
                return torch.tensor(
                    [torch.argmax(self.prediction_net(state))],
                    device=self.device
                )

    def update_target(self):
        """Updates the target model weights to match the prediction model"""

        # Loop over the layers of the prediction and target nets
        for layer in range(len(self.prediction_net)):
            # Check whether we have a layer that stores weights
            if hasattr(self.target_net[layer], 'weight'):
                # Update the target net weights to match the prediction net
                self.target_net[layer].weight = \
                    self.prediction_net[layer].weight

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

        # Compute Q targets
        Q_targets = self.compute_q_targets(next_state_batch, reward_batch)
        # Compute Q predictions
        Q_predictions = self.compute_q_predictions(state_batch, action_batch)
        # Compute loss (=MSE(predictions, targets))
        loss = F.mse_loss(Q_predictions, Q_targets)

        # Set gradients to zero
        self.optimizer.zero_grad()

        # Perform backwards pass and calculate gradients
        loss.backward()

        # Step optimizer
        self.optimizer.step()

    def compute_q_targets(self, next_states, rewards):
        """Computes the Q targets we will compare to the predicted Q values"""

        # Compute output of the target net for next states
        # Keep the gradients for backwards loss pass: with torch.no_grad():
        target_output = self.target_net(next_states)

        # Select appropriate Q values based on which one is higher
        # (since this would have been the one selected in the next state)
        Q_targets_next = target_output.max(1).values

        # Calculate Q values for current states
        Q_targets_current = \
            rewards + self.gamma * Q_targets_next

        return Q_targets_current.unsqueeze(1)

    def compute_q_predictions(self, states, actions):
        """Computes the Q predictions for a given state batch"""

        # Compute output of the policy net for given states
        # Keep the gradients for backwards loss pass: with torch.no_grad():
        policy_output = self.prediction_net(states)

        # Select appropriate Q values from output by indexing with
        # the actual actions
        Q_predictions = torch.gather(
            policy_output, dim=-1, index=actions.unsqueeze(1)
        )

        return Q_predictions

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

            # Set initial flip angle
            self.fa = self.fa_initial + (np.random.random() * 40.0 - 20.0)
            # Run initial simulation
            F0, _, _ = epg.epg_as_torch(
                100, self.fa, 50E-03, .500, 0.025, device=self.device
            )
            # Set initial state
            state = torch.tensor(
                [float(np.abs(F0.cpu()[-1])), self.fa],
                device=self.device
            )

            # Loop over steps/ticks
            while tick < self.n_ticks and not done:
                # Print some info
                print(f"Step {tick + 1:2d}/{self.n_ticks:2d} - ", end="")
                # Choose action
                action = self.choose_action(state, self.epsilon)
                # Simulate step
                next_state, reward, done = self.step(state, action)
                # Add to memory
                self.remember(state, action, reward, next_state, done)
                # Update state
                state = next_state
                # Update tick counter
                tick += 1

                # Print some info
                print(
                    f" - Action: {int(action):1d}"
                    f" - FA: {float(state[1]):4.1f}"
                    f" - Signal: {float(state[0]):5.3f}"
                    f" - Reward: {float(reward):5.2f}"
                )

            # Optimize prediction/policy model
            self.optimize_model(self.batch_size)

            # Update target model
            if episode % self.target_update_period == 0:
                self.update_target()

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    optimizer = SingleSignalOptimizer()
    optimizer.run()

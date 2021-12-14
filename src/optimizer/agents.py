"""Module implementing several agents used in RL algorithms

Here, we give implementations for agents used in several algorithms, namely:

- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- RDPG (Recurrent Deterministic Policy Gradient)
"""

from typing import Union
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from optimizer import models


class DQNAgent(object):
    """Class to represent a Deep Q-Network agent"""

    def __init__(
            self,
            n_states: int = 4,
            n_actions: int = 4,
            gamma: float = 0.99,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 5e-2,
            alpha: float = 0.005,
            tau: float = 1e-2,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            n_states : int
                Number of values passed in "state" (amount of input neurons)
            n_actions : int
                Number of possible actions (amount of output neurons)
            epochs_per_episode : int
                Number of epochs run per training episode
            gamma : float
                Discount factor for future rewards
            epsilon : float
                Exploration/exploitation factor
            epsilon_min : float
                Minimal epsilon
            epsilon_decay : float
                Decay factor for epsilon
            alpha : float
                Learning rate for Adam optimizer
            tau : float
                Measure of "lag" between policy and target models
            device : torch.device | None
                The torch device. If None, assign one.
        """

        # Setup attributes
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.tau = tau

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model
        self.init_model()

    def init_model(self):
        """Constructs model for this agent"""

        # Define architecture
        layers = [self.n_states, 8, 8, self.n_actions]
        activation_funcs = ['relu', 'relu', 'relu', 'none']

        # Construct policy net
        self.prediction_net = models.FullyConnectedModel(
            layers,
            activation_funcs,
            self.device
        )
        # Construct target net
        self.target_net = models.FullyConnectedModel(
            layers,
            activation_funcs,
            self.device
        )

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.prediction_net.parameters(), lr=self.alpha
        )

        # Setup loss criterion
        self.criterion = F.mse_loss

    def select_action(self, state, train=True):
        """Select action for a certain state

        Choose the action for this step.
        This is either random (exploration) or determined
        via the model (exploitation). This rate is determined by
        the epsilon parameter. If train=False, don't give random
        actions.
        """

        if np.random.random() <= self.epsilon and train:
            # Exploration (random choice)
            return torch.tensor(
                [np.random.choice(np.arange(
                    0, self.n_actions
                ))],
                device=self.device
            )
        else:
            # Exploitation (max expected reward)
            with torch.no_grad():
                return torch.tensor(
                    [torch.argmax(self.prediction_net(state))],
                    device=self.device
                )

    def update(self, batch):
        """Update model based on a provided batch"""

        # Split batch
        states, actions, rewards, next_states, _ = batch
        # Cast to tensors
        states = torch.cat(
            [state.unsqueeze(0) for state in states]
        ).to(self.device)
        actions = torch.cat(
            [action.unsqueeze(0) for action in actions]
        ).to(self.device)
        rewards = torch.cat(
            [reward.unsqueeze(0) for reward in rewards]
        ).to(self.device)
        next_states = torch.cat(
            [next_state.unsqueeze(0) for next_state in next_states]
        ).to(self.device)

        # Compute Q targets
        Q_targets = self.compute_q_targets(
            next_states, rewards
        )
        # Compute Q predictions
        Q_predictions = self.compute_q_predictions(
            states, actions
        )

        # Compute loss (= MSE(predictions, targets))
        loss = self.criterion(Q_predictions, Q_targets)

        # Set gradients to zero
        self.optimizer.zero_grad()

        # Perform backwards pass and calculate gradients
        loss.backward()

        # Step optimizer
        self.optimizer.step()

        # Update target network (lagging weights)
        for target_param, param in zip(
                self.target_net.parameters(),
                self.prediction_net.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

    def compute_q_targets(self, next_states, rewards):
        """Computes the Q targets for given next_states and rewards batches"""

        # Compute output of the target net for next states
        # Keep the gradients for backwards loss pass
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
        # Keep the gradients for backwards loss pass
        policy_output = self.prediction_net(states)

        # Select appropriate Q values from output by indexing with
        # the actual actions
        Q_predictions = torch.gather(
            policy_output, dim=-1, index=actions
        )

        return Q_predictions

    def update_epsilon(self):
        """Update epsilon (called at the end of an episode)"""

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Save model and optimizer states to file"""

        torch.save({
            'prediction': self.prediction_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load model and optimizer states from file"""

        # Load stored dict
        pretrained_dict = torch.load(path)

        # Set states
        self.prediction_net.load_state_dict(
            pretrained_dict["prediction"]
        )
        self.target_net.load_state_dict(
            pretrained_dict["target"]
        )
        self.optimizer.load_state_dict(
            pretrained_dict["optimizer"]
        )


class DDPGAgent(object):
    """Class to represent a Deep Deterministic Policy Gradient agent"""

    def __init__(self):
        """Initializes and builds attributes for this class

        Parameters
        ----------
        """


class RDPGAgent(object):
    """Class to represent a Recurrent Deterministic Policy Gradient agent"""

    def __init__(self):
        """Initializes and builds attributes for this class

        Parameters
        ----------
        """

        raise NotImplementedError()

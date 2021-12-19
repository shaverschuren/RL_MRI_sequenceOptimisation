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
from optimizer import environments, models


class DQNAgent(object):
    """Class to represent a Deep Q-Network agent"""

    def __init__(
            self,
            n_states: int = 4,
            n_actions: int = 4,
            gamma: float = 0.99,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 5e-3,
            alpha: float = 0.005,
            tau: float = 1e-2,
            hidden_layers: list[int] = [8, 8],
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            n_states : int
                Number of values passed in "state" (amount of input neurons)
            n_actions : int
                Number of possible actions (amount of output neurons)
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
            hidden_layers : list[int]
                Number of neurons of each hidden layer
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
        self.hidden_layers = hidden_layers

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
        layers = [self.n_states] + self.hidden_layers + [self.n_actions]
        activation_funcs = ['relu'] * len(layers) + ['none']

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
            self.action_mode = "exploration"
            return torch.tensor(
                [np.random.choice(np.arange(
                    0, self.n_actions
                ))],
                device=self.device
            )
        else:
            # Exploitation (max expected reward)
            self.action_mode = "exploitation"
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
            rewards + self.gamma * Q_targets_next.unsqueeze(1)

        return Q_targets_current

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

    def __init__(
            self,
            action_space: environments.ActionSpace,
            n_states: int = 4,
            n_actions: int = 1,
            gamma: float = 0.99,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 2e-3,
            alpha_actor: float = 1e-4,
            alpha_critic: float = 1e-3,
            tau: float = 1e-2,
            hidden_layers: list[int] = [16, 64, 32],
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            action_space : environments.ActionSpace
                Action space object
            n_states : int
                Number of values passed in "state" (amount of input neurons)
            n_actions : int
                Number of possible actions (amount of output neurons)
            gamma : float
                Discount factor for future rewards
            epsilon : float
                Exploration/exploitation factor
            epsilon_min : float
                Minimal epsilon
            epsilon_decay : float
                Decay factor for epsilon
            alpha_actor : float
                Learning rate for Adam optimizer  for actor model
            alpha_critic : float
                Learning rate for Adam optimizer  for critic model
            tau : float
                Measure of "lag" between policy and target models
            hidden_layers : list[int]
                Number of neurons of each hidden layer
            device : torch.device | None
                The torch device. If None, assign one.
        """

        # Setup attributes
        self.action_space = action_space
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.tau = tau
        self.hidden_layers = hidden_layers

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model
        self.init_model()

    def init_model(self):
        """Constructs reinforcement learning model

        Neural nets: Fully connected 4-16-64-32-1 and 5-16-64-32-1
        Loss: L2 (MSE) Loss
        Optimizer: Adam with lr alpha
        """

        # Define model architecture
        actor_layers = [self.n_states] + self.hidden_layers + [self.n_actions]
        critic_layers = (
            [self.n_states + self.n_actions]
            + self.hidden_layers + [self.n_actions]
        )

        actor_activation_funcs, critic_activation_funcs = (
            ['relu'] * (len(self.hidden_layers) + 1) + ['tanh'],
            ['relu'] * (len(self.hidden_layers) + 1) + ['none']
        )

        # Construct actor models (network + target network)
        self.actor = models.FullyConnectedModel(
            actor_layers,
            actor_activation_funcs,
            self.device
        )
        self.actor_target = models.FullyConnectedModel(
            actor_layers,
            actor_activation_funcs,
            self.device
        )
        # Construct critic models (network + target network)
        self.critic = models.FullyConnectedModel(
            critic_layers,
            critic_activation_funcs,
            self.device
        )
        self.critic_target = models.FullyConnectedModel(
            critic_layers,
            critic_activation_funcs,
            self.device
        )

        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Setup optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.alpha_actor, weight_decay=1e-2
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.alpha_critic, weight_decay=1e-2
        )

        # Setup criterion
        self.critic_criterion = torch.nn.MSELoss()

    def select_action(self, state, train=True):
        """Select action

        Determine the action for this step
        This is determined by the agent model and the
        noise function passed earlier. This way, we
        balance exploration/exploitation.
        """

        # Get action from actor model
        pure_action = self.actor(state).detach().numpy()
        # Add noise (if training)
        noise = (
            np.random.normal(0., 1.0 * self.epsilon, np.shape(pure_action))
            if train else 0.
        )
        noisy_action = np.clip(
            pure_action + noise,
            self.action_space.low,
            self.action_space.high
        )

        return torch.FloatTensor(noisy_action, device=self.device)

    def update(self, batch):
        """Updates the models based on a given batch"""

        # Extract states, actions, rewards, next_states
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

        # Determine critic loss
        Qvals = self.critic(torch.cat([states, actions], 1))
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(
            torch.cat([next_states, next_actions.detach()], 1)
        )
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Determine actor loss
        policy_loss = -self.critic(
            torch.cat([states, self.actor(states)], 1)
        ).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target networks (lagging weights)
        for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )
        for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        return float(policy_loss.detach()), float(critic_loss.detach())

    def update_epsilon(self):
        """Update epsilon (called at the end of an episode)"""

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        """Loads models from a file"""

        # Load stored dict
        pretrained_dict = torch.load(path)

        # Set model states
        self.actor.load_state_dict(
            pretrained_dict["actor"]
        )
        self.critic.load_state_dict(
            pretrained_dict["critic"]
        )
        self.actor_target.load_state_dict(
            pretrained_dict["actor_target"]
        )
        self.critic_target.load_state_dict(
            pretrained_dict["critic_target"]
        )
        # Set optimizer states
        self.actor_optimizer.load_state_dict(
            pretrained_dict["actor_optimizer"]
        )
        self.critic_optimizer.load_state_dict(
            pretrained_dict["critic_optimizer"]
        )

    def save(self, path):
        """Saves models to a file"""

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)


class RDPGAgent(object):
    """Class to represent a Recurrent Deterministic Policy Gradient agent"""

    def __init__(self):
        """Initializes and builds attributes for this class

        Parameters
        ----------
        """

        raise NotImplementedError()

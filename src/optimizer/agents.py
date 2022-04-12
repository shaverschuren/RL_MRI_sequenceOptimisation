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
from torch.autograd import Variable
from optimizer import environments, models
from torchviz import make_dot


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
            epsilon_decay: float = 1. - 1e-3,
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

    def __init__(
            self,
            action_space: environments.ActionSpace,
            n_states: int = 2,
            n_actions: int = 2,
            gamma: float = 0.99,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 2e-3,
            tbptt_k1: int = 5,
            tbptt_k2: int = 5,
            alpha_actor: float = 1e-4,
            alpha_critic: float = 1e-3,
            tau: float = 1e-2,
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
            tbptt_k1 : int
                Amount of forward steps between model updates
            tbptt_k2 : int
                Amount of steps over which to perform backpropagation
            alpha_actor : float
                Learning rate for Adam optimizer  for actor model
            alpha_critic : float
                Learning rate for Adam optimizer  for critic model
            tau : float
                Measure of "lag" between policy and target models
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
        self.tbptt_k1 = tbptt_k1
        self.tbptt_k2 = tbptt_k2
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
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
        """Constructs reinforcement learning model

        Neural nets: LSTM-RNN
        Loss: L2 (MSE) Loss
        Optimizer: Adam with lr alpha
        """

        # Define hidden size
        hidden_size = 64

        # Construct actor models (network + target network)
        self.actor = models.RecurrentModel_LSTM(
            input_size=self.n_states,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            fully_connected_architecture=[
                self.n_states, 64, 128, hidden_size
            ],
            output_activation="tanh",
            device=self.device
        )
        self.critic = models.RecurrentModel_LSTM(
            input_size=self.n_states + self.n_actions,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            fully_connected_architecture=[
                self.n_states + self.n_actions, 64, 128, hidden_size
            ],
            output_activation="none",
            device=self.device
        )
        # Construct critic models (network + target network)
        self.actor_target = models.RecurrentModel_LSTM(
            input_size=self.n_states,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            fully_connected_architecture=[
                self.n_states, 64, 128, hidden_size
            ],
            output_activation="tanh",
            device=self.device
        )
        self.critic_target = models.RecurrentModel_LSTM(
            input_size=self.n_states + self.n_actions,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            fully_connected_architecture=[
                self.n_states + self.n_actions, 64, 128, hidden_size
            ],
            output_activation="none",
            device=self.device
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
        self.critic_criterion = torch.nn.L1Loss()

    def select_action(self, state, train=True):
        """Select action based on current state

        Determine the action for this step
        This is determined by the agent model and the
        noise function passed earlier. This way, we
        balance exploration/exploitation.
        """

        # Get action from actor model
        with torch.no_grad():
            pure_action, _ = self.actor(torch.unsqueeze(state, 0))
        pure_action = torch.squeeze(pure_action, 0).detach().numpy()
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
        """Updates the models based on a given batch

        Here, we use trucated backpropogation through time (TBPTT)
        TODO: source
        TODO: Explanation of k1, k2 etc.
        """

        # TODO: Debugging
        torch.autograd.set_detect_anomaly(True)
        addresses = []
        graphs = []

        # Check for validity of tbptt parameters
        if self.tbptt_k1 > len(batch) or self.tbptt_k2 > len(batch):
            raise ValueError(
                "For TBPTT, k1 and k2 should both be smaller or equal"
                " to the sequence length."
                f"\nGot n={len(batch)}, k1={self.tbptt_k1}, k2={self.tbptt_k2}"
            )

        # Reset hidden states
        self.reset(batch_size=len(batch[0]))

        # Extract initial hidden states
        # We manually keep track of hidden states because we'll
        # need to do several passes through the same model per timestep
        hidden_critic = [(self.critic.hx, self.critic.cx)]
        hidden_actor = [(self.actor.hx, self.actor.cx)]
        hidden_critic_target = [(self.critic_target.hx, self.critic_target.cx)]
        hidden_actor_target = [(self.actor_target.hx, self.actor_target.cx)]

        # Create lists for policy and critic loss
        policy_loss_sums = [
            torch.tensor(0., dtype=torch.float32, device=self.device)
        ]
        critic_loss_sums = [
            torch.tensor(0., dtype=torch.float32, device=self.device)
        ]

        # Define initial update count
        update_count = 0

        # Loop over timesteps of the trajectories
        # Process all trajectories in parallel, however
        for t in range(len(batch)):

            # Extract states, actions, rewards, next_states
            states = [transition.state for transition in batch[t]]
            actions = [transition.action for transition in batch[t]]
            rewards = [transition.reward for transition in batch[t]]
            next_states = [transition.next_state for transition in batch[t]]

            # Cast to tensors
            states = torch.cat(
                [state.unsqueeze(0) for state in states]
            ).to(self.device)
            actions = torch.cat(
                [action.unsqueeze(0) for action in actions]
            ).to(self.device)
            rewards = torch.unsqueeze(torch.cat(
                [reward.unsqueeze(0) for reward in rewards]
            ).to(self.device), 1)
            next_states = torch.cat(
                [next_state.unsqueeze(0) for next_state in next_states]
            ).to(self.device)

            # Determine target value (Q-prime)
            with torch.no_grad():
                # Compute next_Q and update target hidden states
                next_actions, hidden_actor_target_1 = self.actor_target(
                    next_states, hidden_actor_target[t]
                )
                next_Q, hidden_critic_target_1 = self.critic_target(
                    torch.cat([next_states, next_actions], 1),
                    hidden_critic_target[t]
                )
                # Compute Q_target (R + discount * Qnext)
                Q_target = rewards + self.gamma * next_Q

            # Compute actual Q-values and policy for this timestep
            Q, hidden_critic_1 = self.critic(
                torch.cat([states, actions], 1),
                hidden_critic[t]
            )
            policy, hidden_actor_1 = self.actor(states, hidden_actor[t])

            # Compute critic loss
            critic_loss = self.critic_criterion(Q, Q_target)

            # Compute actor loss
            policy_loss = -torch.mean(
                self.critic(
                    torch.cat([states, policy], 1),
                    hidden_critic[t]
                )[0]
            )

            g_critic = make_dot(
                critic_loss, params=dict(self.critic.named_parameters())  # ,
                # show_attrs=True, show_saved=True
            )
            g_policy = make_dot(
                policy_loss, params=dict(self.actor.named_parameters())  # ,
                # show_attrs=True, show_saved=True
            )

            graphs.append((g_policy, g_critic))

            # Store losses
            policy_loss_sums[update_count] += policy_loss
            critic_loss_sums[update_count] += critic_loss

            # Update hidden states
            address1 = (
                f"{hex(id(hidden_actor_1[0]))}; {hex(id(hidden_actor_1[1]))}; "
                f"{hex(id(hidden_critic_1[0]))}; {hex(id(hidden_critic_1[1]))}"
            )

            hidden_critic.append(hidden_critic_1)
            hidden_actor.append(hidden_actor_1)
            hidden_critic_target.append(hidden_critic_target_1)
            hidden_actor_target.append(hidden_actor_target_1)

            address2 = (
                f"{hex(id(hidden_actor_1[0]))}; {hex(id(hidden_actor_1[1]))}; "
                f"{hex(id(hidden_critic_1[0]))}; {hex(id(hidden_critic_1[1]))}"
            )
            address3 = (
                f"{hex(id(hidden_actor[-1][0]))}; "
                f"{hex(id(hidden_actor[-1][1]))}; "
                f"{hex(id(hidden_critic[-1][0]))}; "
                f"{hex(id(hidden_critic[-1][1]))}"
            )

            addresses.append(address1)

            if address1 != address2 or address1 != address3:
                raise UserWarning(f"\n{address1}\n{address2}\n{address3}")

            # Update the network periodically
            if (t + 1) % self.tbptt_k1 == 0:

                addresses_check_act = [
                    f"{hex(id(act[0]))}; {hex(id(act[1]))}"
                    for act in hidden_actor
                ]
                addresses_check_cri = [
                    f"{hex(id(cri[0]))}; {hex(id(cri[1]))}"
                    for cri in hidden_critic
                ]

                for i in range(1, len(hidden_actor)):
                    check_act = addresses_check_act[i] == addresses[i - 1][:30]
                    check_cri = addresses_check_cri[i] == addresses[i - 1][32:]

                    if not (check_act and check_cri):
                        raise UserWarning(
                            "Memory addresses changed!"
                            f"\nError found at hidden state index {i}"
                            f"\nExpected:\t{addresses_check_act[i]}; "
                            f"{addresses_check_cri[i]}"
                            f"\nFound:\t\t{addresses[i - 1]}"
                        )

                # print(f"Len: {len(hidden_critic)}")

                # Normalize losses
                policy_loss_sums[update_count] /= float(self.tbptt_k1)
                critic_loss_sums[update_count] /= float(self.tbptt_k1)

                # Detach hidden states that are too far back in history (k2)
                for i in range(t - self.tbptt_k2 + 2):
                    # print(f"{i}:{len(hidden_critic)}")
                    hidden_critic[i] = (
                        Variable(hidden_critic[i][0].data),
                        Variable(hidden_critic[i][1].data)
                    )
                    hidden_actor[i] = (
                        Variable(hidden_actor[i][0].data),
                        Variable(hidden_actor[i][1].data)
                    )
                    hidden_critic_target[i] = (
                        Variable(hidden_critic_target[i][0].data),
                        Variable(hidden_critic_target[i][1].data)
                    )
                    hidden_actor_target[i] = (
                        Variable(hidden_actor_target[i][0].data),
                        Variable(hidden_actor_target[i][1].data)
                    )

                # Update actor
                # Here, retain the graph since the gradients of the policy loss
                # flow through the critic model, which is needed later for
                # the critic backpropagation.
                self.actor_optimizer.zero_grad()
                policy_loss_sums[update_count].backward(retain_graph=True)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss_sums[update_count].backward()
                self.critic_optimizer.step()

                # Detach losses
                policy_loss_sums[update_count].detach()
                critic_loss_sums[update_count].detach()

                # Update target networks (lagging weights)
                for target_param, param in zip(
                        self.actor_target.parameters(),
                        self.actor.parameters()
                ):
                    target_param.data.copy_(
                        param.data * self.tau
                        + target_param.data * (1.0 - self.tau)
                    )
                for target_param, param in zip(
                        self.critic_target.parameters(),
                        self.critic.parameters()
                ):
                    target_param.data.copy_(
                        param.data * self.tau
                        + target_param.data * (1.0 - self.tau)
                    )

                # Create new loss term in the summation list
                policy_loss_sums.append(
                    torch.tensor(0., dtype=torch.float32, device=self.device)
                )
                critic_loss_sums.append(
                    torch.tensor(0., dtype=torch.float32, device=self.device)
                )

                # Increase update counter
                update_count += 1

        # Return losses
        if update_count > 0:
            return (
                float(sum(policy_loss_sums) / float(update_count)),
                float(sum(critic_loss_sums) / float(update_count))
            )
        else:
            raise UserWarning("Updating failed")

    def update_epsilon(self):
        """Update epsilon (called at the end of an episode)"""

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def detach_hidden(self):
        """Detach hidden states"""

        self.actor.hx = Variable(self.actor.hx.data)
        self.actor.cx = Variable(self.actor.cx.data)
        self.critic.hx = Variable(self.critic.hx.data)
        self.critic.cx = Variable(self.critic.cx.data)
        self.actor_target.hx = Variable(self.actor_target.hx.data)
        self.actor_target.cx = Variable(self.actor_target.cx.data)
        self.critic_target.hx = Variable(self.critic_target.hx.data)
        self.critic_target.cx = Variable(self.critic_target.cx.data)

    def reset(self, batch_size=1):
        """Reset hidden states of the models, ready for a new episode"""

        for model in (
            self.actor, self.actor_target, self.critic, self.critic_target
        ):
            model.reset_hidden_state(batch_size=batch_size)

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

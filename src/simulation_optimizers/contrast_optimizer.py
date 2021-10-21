"""Contrast Optimizer

This module implements a reinforcement model that
optimizes the flip angle for the signal contrast between
two (simulated) tissues. We implement this so that the T1/T2
is changed for each tissue each episode."""

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


class ContrastOptimizer():
    """
    A class to represent a reinforcement model that
    optimizes the flip angle for the signal contrast between
    two (simulated) tissues. We implement this so that the T1/T2
    is changed for each tissue each episode.
    """

    def __init__(
            self,
            n_episodes: int = 50,
            n_ticks: int = 100,
            batch_size: int = 32,
            epochs_per_episode: int = 10,
            memory_done_criterion: int = 15,
            n_done_criterion: int = 3,
            fa_initial_min: float = 0.,
            fa_initial_max: float = 90.,
            fa_delta: float = 1.0,
            Nfa: int = 100,
            T1_range_1: list[float] = [0.100, 1.500],
            T1_range_2: list[float] = [0.100, 1.500],
            T2_range_1: list[float] = [0.010, 0.100],
            T2_range_2: list[float] = [0.010, 0.100],
            tr: float = 0.050,
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

            Parameters
            ----------
                n_episodes : int
                    Number of training episodes
                n_ticks : int
                    Number of training steps
                batch_size : int
                    Batch size for training
                epochs_per_episode : int
                    Number of training epochs after each episode
                memory_done_criterion : int
                    Max length of "recent memory", used in "done" criterion
                n_done_criterion : int
                    Number of same flip angles in recent memory needed to end
                fa_initial_min : float
                    Minimal initial flip angle [deg]
                fa_initial_max : float
                    Maximal initial flip angle [deg]
                fa_delta : float
                    Amount of change in flip angle done by the model [deg]
                Nfa : int
                    Number of pulses in epg simulation
                T1_range_1 : list[float]
                    Range for T1 relaxation for tissue 1 in epg simulation [s]
                T1_range_2 : list[float]
                    Range for T1 relaxation for tissue 2 in epg simulation [s]
                T2_range_1 : list[float]
                    Range for T2 relaxation for tissue 1 in epg simulation [s]
                T2_range_2 : list[float]
                    Range for T2 relaxation for tissue 2 in epg simulation [s]
                tr : float
                    Repetition time in epg simulation [ms]
                gamma : float
                    Discount factor for Q value calculation
                epsilon : float
                    Initial epsilon (factor used for exploration regulation)
                epsilon_min : float
                    Minimal epsilon
                epsilon_decay : float
                    Epsilon decay factor
                alpha : float
                    Learning rate for Adam optimizer
                target_update_period : int
                    Periods between target net updates
                log_dir : str | bytes | os.PathLike
                    Path to log directory
                verbose : bool
                    Whether to print info
                device : torch.device
                    The PyTorch device. If None, assign one.
        """

        # Setup attributes
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size
        self.epochs_per_episode = epochs_per_episode
        self.memory_done_criterion = memory_done_criterion
        self.n_done_criterion = n_done_criterion
        self.fa_initial_min = fa_initial_min
        self.fa_initial_max = fa_initial_max
        self.fa = (fa_initial_min + fa_initial_max) / 2
        self.fa_delta = fa_delta
        self.Nfa = Nfa
        self.T1_range_1 = T1_range_1
        self.T1_range_2 = T1_range_2
        self.T2_range_1 = T2_range_1
        self.T2_range_2 = T2_range_2
        self.tr = tr
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

        Includes action space and deltas.
        The actual simulation "environment" is already implemented
        in the epg code.
        """

        # Define action space
        # 0 - Decrease 1xdelta
        # 1 - Increase 1xdelta
        # 2 - Decrease 5xdelta
        # 3 - Increase 5xdelta
        self.action_space = np.array([0, 1, 2, 3])
        self.deltas = np.array([
            -1. * self.fa_delta,
            +1. * self.fa_delta,
            -5. * self.fa_delta,
            +5. * self.fa_delta
        ])

    def init_model(self):
        """Constructs reinforcement learning model

        Neural nets: Fully connected 2-8-8-4
        Loss: L2 (MSE) Loss
        Optimizer: Adam with lr alpha
        """

        # Construct policy net
        self.prediction_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, 4)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4, 8)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(8, 8)),
            ('relu3', nn.ReLU()),
            ('output', nn.Linear(8, 4))
        ])).to(self.device)
        # Construct target net
        self.target_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, 4)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4, 8)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(8, 8)),
            ('relu3', nn.ReLU()),
            ('output', nn.Linear(8, 4))
        ])).to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.prediction_net.parameters(), lr=self.alpha
        )

    def calculate_contrast(self, fa=None):
        """Calculates the contrast using parameters stored in self"""

        # Select flip angle
        if not fa:
            fa = self.fa
        else:
            fa = float(fa)

        # Run simulations
        F0_1, _, _ = epg.epg_as_torch(
            self.Nfa, fa, self.tr,
            self.T1_1, self.T2_1, device=self.device
        )
        F0_2, _, _ = epg.epg_as_torch(
            self.Nfa, fa, self.tr,
            self.T1_2, self.T2_2, device=self.device
        )

        # Determine contrast
        contrast = abs(np.abs(F0_1.cpu()[-1]) - np.abs(F0_2.cpu()[-1]))

        return float(contrast)

    def calculate_exact_optimum(self):
        """Analytically determine the exact optimum for comparison."""

        # Determine E1 for both tissues
        E1a = np.exp(-self.tr / self.T1_1)
        E1b = np.exp(-self.tr / self.T1_2)

        # Calculate optimal flip angle analytically. Formula retrieved from:
        # Haselhoff EH. Optimization of flip angle for T1 dependent contrast: a
        # closed form solution. Magn Reson Med 1997;38:518 – 9.
        optimal_fa = float(np.arccos(
            (
                -2 * E1a * E1b + E1a + E1b - 2 + np.sqrt(
                    -3 * (E1a ** 2) - 3 * (E1b ** 2)
                    + 4 * (E1a ** 2) * (E1b ** 2) - 2 * E1a * E1b + 4
                )
            )
            / (
                2 * (E1a * E1b - E1a - E1b)
            )
        ) * 180. / np.pi)

        # REturn optimal flip angle and optimal contrast
        return optimal_fa, self.calculate_contrast(optimal_fa)

    def step(self, old_state, action, step_i):
        """Run step of the environment simulation

        - Perform selected action
        - Run EPG simulation
        - Update state
        - Update reward
        - Update done
        """

        # Adjust flip angle according to action
        if int(action) in self.action_space:
            # Adjust flip angle
            delta = float(self.deltas[int(action)])
            self.fa += delta
            # Correct for flip angle out of bounds
            if self.fa < 0.0: self.fa = 0.0
            if self.fa > 180.0: self.fa = 180.0
        else:
            raise ValueError("Action not in action space")

        # Run simulations and uppdate state
        state = torch.tensor(
            [self.calculate_contrast(), self.fa],
            device=self.device
        )

        # Define reward as either +/- 1 for increase or decrease in signal
        if state[0] > old_state[0]:
            reward_float = 1.0
        else:
            reward_float = -1.0

        # Scale reward with signal difference
        if float(old_state[0]) < 1e-2:
            # If old_state signal is too small, set reward gain to 20
            reward_gain = 30.
        else:
            # Calculate relative signal difference and derive reward gain
            signal_diff = abs(state[0] - old_state[0]) / old_state[0]
            reward_gain = signal_diff * 50.

            # If reward gain is lower than 0.5, use 0.5
            # We do this to prevent disappearing rewards near the optimum
            if reward_gain < 0.5: reward_gain = 0.5
            # If reward gain is higher than 20, use 20
            # We do this to prevent blowing up rewards near the edges
            if reward_gain > 30.: reward_gain = 30.

        reward_float *= reward_gain

        # Scale reward with step_i (faster improvement yields bigger rewards)
        # Only scale the positives, though.
        if reward_float > 0.:
            reward_float *= np.exp(-step_i / self.n_ticks)

        # Store reward in tensor
        reward = torch.tensor(
            [float(reward_float)], device=self.device
        )

        # Set done
        if step_i + 1 >= self.n_done_criterion:
            # Check whether the "done" criterion is met

            # Calculate recent memory length (max memory_done_criterion)
            memory_len = min(step_i + 1, self.memory_done_criterion)
            # Retrieve recent memory
            recent_memory = list(self.memory)[-memory_len + 1:]
            # Store as transitions
            recent_transitions = self.Transition(*zip(*recent_memory))
            # Extract flip angles
            recent_states = np.array(torch.cat(recent_transitions.state).cpu())
            recent_fa = np.delete(
                recent_states,
                np.arange(0, recent_states.size, 2)
            )
            # Append current/last flip angle
            recent_fa = np.append(recent_fa, float(old_state[1]))

            # Check for returning flip angles in recent memory
            _, counts = np.unique(recent_fa, return_counts=True)

            if (counts >= self.n_done_criterion).any():
                done = torch.tensor(1, device=self.device)
            else:
                done = torch.tensor(0, device=self.device)

        else:
            # Recent memory too short: We're not done yet
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
            if self.verbose: print("\033[93mR\033[0m", end="", flush=True)
            return torch.tensor(
                [np.random.choice(self.action_space)],
                device=self.device
            )
        else:
            # Exploitation (max expected reward) -> *P*olicy
            if self.verbose: print("\033[94mP\033[0m", end="", flush=True)
            with torch.no_grad():
                return torch.tensor(
                    [torch.argmax(self.prediction_net(state))],
                    device=self.device
                )

    def update_target(self):
        """Updates the target model weights to match the prediction model"""

        # Print some info
        if self.verbose:
            print("Updating target model...\t", end="", flush=True)

        # Loop over the layers of the prediction and target nets
        for layer in range(len(self.prediction_net)):
            # Check whether we have a layer that stores weights
            if hasattr(self.target_net[layer], 'weight'):
                # Update the target net weights to match the prediction net
                self.target_net[layer].weight = \
                    self.prediction_net[layer].weight

        # Print some info
        if self.verbose:
            print("\033[92mOK\033[0m")

    def optimize_model(self, batch_size):
        """Optimize model based on previous episode"""

        # Check whether memory is long enough
        if len(self.memory) < batch_size:
            return

        # Print some info
        if self.verbose:
            print("Training prediction model...\t", end="", flush=True)

        # Loop over number of epochs
        for _ in range(self.epochs_per_episode):
            # Copy memory to local list
            local_memory = list(self.memory)

            while len(local_memory) >= batch_size:
                # Create selection from remaining local memory
                indices = list(range(len(local_memory)))
                selection = list(random.sample(indices, batch_size))
                selection.sort(reverse=True)
                # Create transitions and remove items from local memory
                transitions = []
                for index in selection:
                    transitions.append(local_memory[index])
                    local_memory.pop(index)
                # Create batch
                batch = self.Transition(*zip(*transitions))

                # Split state, action and reward batches
                # States
                state_batch_list = []
                for tensor_i in range(len(batch.state)):
                    state_batch_list.append(
                        np.array(batch.state[tensor_i].cpu())
                    )
                state_batch_np = np.array(state_batch_list)
                state_batch = torch.as_tensor(
                    state_batch_np, device=self.device
                )
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
                Q_targets = self.compute_q_targets(
                    next_state_batch, reward_batch
                )
                # Compute Q predictions
                Q_predictions = self.compute_q_predictions(
                    state_batch, action_batch
                )
                # Compute loss (= MSE(predictions, targets))
                loss = F.mse_loss(Q_predictions, Q_targets)

                # Set gradients to zero
                self.optimizer.zero_grad()

                # Perform backwards pass and calculate gradients
                loss.backward()

                # Step optimizer
                self.optimizer.step()

        # Print some info
        if self.verbose:
            print("\033[92mOK\033[0m")

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
            policy_output, dim=-1, index=actions.unsqueeze(1)
        )

        return Q_predictions

    def run(self, train=True):
        """Run the training loop

        Here, if train=False is passed, we simply test
        the performance of a previously trained model
        """

        # Print some info
        if train:
            print("\n===== Running training loop =====\n")
        else:
            print("\n======= Running test loop =======\n")

        # Create lists of initial flip angles
        # (uniformly distributed in range)
        initial_fa_low = list(np.linspace(
            self.fa_initial_min,
            (self.fa_initial_min + self.fa_initial_max) / 2,
            self.n_episodes // 2
        ))
        initial_fa_high = list(np.linspace(
            (self.fa_initial_min + self.fa_initial_max) / 2,
            self.fa_initial_max,
            self.n_episodes // 2 if self.n_episodes % 2 == 0
            else self.n_episodes // 2 + 1
        ))
        # Create lists of T1 and T2s for both tissues
        # (uniformly distributed in range)
        T1_list_1 = list(np.linspace(
            self.T1_range_1[0], self.T1_range_1[1], self.n_episodes
        ))
        T2_list_1 = list(np.linspace(
            self.T2_range_1[0], self.T2_range_1[1], self.n_episodes
        ))
        T1_list_2 = list(np.linspace(
            self.T1_range_2[0], self.T1_range_2[1], self.n_episodes
        ))
        T2_list_2 = list(np.linspace(
            self.T2_range_2[0], self.T2_range_2[1], self.n_episodes
        ))

        # Loop over episodes
        for episode in range(self.n_episodes) if train else range(10):
            # Print some info
            if self.verbose:
                print(
                    f"\n=== Episode {episode + 1:3d}/"
                    f"{self.n_episodes if train else 10:3d} ==="
                )
            # Reset done and tick counter
            done = False
            tick = 0

            # Set initial flip angle, T1 and T2 for this episode
            # If train, take them from the uniform distributions
            # If test (not train), take them randomly in a range
            if train:
                # Set initial flip angle. Here, we randomly sample from the
                # uniformly distributed lists we created earlier. We alternate
                # between high and low flip angles to aid the training process.
                if episode % 2 == 0:
                    self.fa = float(initial_fa_high.pop(
                        random.randint(0, len(initial_fa_high) - 1)
                    ))
                else:
                    self.fa = float(initial_fa_low.pop(
                        random.randint(0, len(initial_fa_low) - 1)
                    ))
                # Set T1 and T2 for this episode. We randomly sample these
                # from the previously definded uniform distributions for both
                # tissues.
                self.T1_1 = float(T1_list_1.pop(
                    random.randint(0, len(T1_list_1) - 1)
                ))
                self.T2_1 = float(T2_list_1.pop(
                    random.randint(0, len(T2_list_1) - 1)
                ))
                self.T1_2 = float(T1_list_2.pop(
                    random.randint(0, len(T1_list_2) - 1)
                ))
                self.T2_2 = float(T2_list_2.pop(
                    random.randint(0, len(T2_list_2) - 1)
                ))
            else:
                # If in test mode, take flip angle, T1, T2 randomly.
                # We do this to provide a novel testing environment.
                self.fa = random.uniform(
                    self.fa_initial_min, self.fa_initial_max
                )
                self.T1_1 = random.uniform(
                    self.T1_range_1[0], self.T1_range_1[1])
                self.T2_1 = random.uniform(
                    self.T2_range_1[0], self.T2_range_1[1])
                self.T1_2 = random.uniform(
                    self.T1_range_2[0], self.T1_range_2[1])
                self.T2_2 = random.uniform(
                    self.T2_range_2[0], self.T2_range_2[1])

            # Run initial simulations
            contrast = self.calculate_contrast()
            # Set initial state
            state = torch.tensor(
                [contrast, self.fa],
                device=self.device
            )

            # # Print some info on the specific environment used this episode.
            optimal_angle, optimal_contrast = self.calculate_exact_optimum()
            print(
                "\n-----------------------------------"
                f"\nT1a={self.T1_1:.4f}s; T2a={self.T2_1:.4f}s; "
                f"T1b={self.T1_2:.4f}s; T2b={self.T2_2:.4f}s"
                f"\nInitial alpha:\t\t{self.fa:4.1f} [deg]"
                f"\nOptimal alpha:\t\t{optimal_angle:4.1f} [deg]"
                f"\nOptimal contrast:\t{optimal_contrast:4.3f} [-]"
                "\n-----------------------------------\n"
            )

            # Loop over steps/ticks
            while tick < self.n_ticks and not bool(done):
                # Print some info
                print(f"Step {tick + 1:3d}/{self.n_ticks:3d} - ", end="")

                # Choose action
                action = self.choose_action(
                    state, self.epsilon if train else 0.
                )
                # Simulate step
                next_state, reward, done = self.step(state, action, tick)
                # Add to memory
                self.remember(state, action, reward, next_state, done)
                # Update state
                state = next_state
                # Update tick counter
                tick += 1

                # Print some info
                color_str = "\033[92m" if reward > 0. else "\033[91m"
                end_str = "\033[0m"
                print(
                    f" - Action: {int(action):1d}"
                    f" - FA: {float(state[1]):4.1f}"
                    f" - Contrast: {float(state[0]):5.3f}"
                    " - Reward: "
                    "" + color_str + f"{float(reward):5.1f}" + end_str
                )
                if bool(done):
                    print("Stopping criterion met")

            # Print some info on theoretical optimum
            actual_contrast = float(state[0])
            if actual_contrast == 0.:
                relative_contrast_error = 1000.
            else:
                relative_contrast_error = abs(
                    optimal_contrast - actual_contrast
                ) * 100. / actual_contrast

            print(
                f"Actual error: (fa) {abs(self.fa - optimal_angle):4.1f} deg",
                f"; (signal) {relative_contrast_error:5.2f}%"
            )

            if train:
                # Optimize prediction/policy model
                self.optimize_model(self.batch_size)

                # Update target model
                if episode % self.target_update_period == 0:
                    self.update_target()

                # Update epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    optimizer = ContrastOptimizer()
    optimizer.run()
    optimizer.run(train=False)

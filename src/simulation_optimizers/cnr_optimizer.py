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
import warnings                                             # noqa: E402
from datetime import datetime                               # noqa: E402
from collections import namedtuple, OrderedDict, deque      # noqa: E402
import random                                               # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn as nn                                       # noqa: E402
import torch.optim as optim                                 # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from epg_simulator.python import epg                        # noqa: E402
from util import loggers                                    # noqa: E402


class ContrastOptimizer():
    """
    A class to represent a reinforcement model that
    optimizes the flip angle for the signal contrast between
    two (simulated) tissues. We implement this so that the T1/T2
    is changed for each tissue each episode.
    """

    def __init__(
            self,
            n_episodes: int = 250,
            n_ticks: int = 100,
            batch_size: int = 32,
            epochs_per_episode: int = 5,
            memory_done_criterion: int = 15,
            n_done_criterion: int = 3,
            fa_range: list[float] = [20., 60.],
            fa_delta: float = 1.0,
            Nfa: int = 100,
            T1_range_1: list[float] = [0.100, 2.500],
            T1_range_2: list[float] = [0.100, 2.500],
            T2_range_1: list[float] = [0.005, 0.100],
            T2_range_2: list[float] = [0.005, 0.100],
            tr: float = 0.050,
            noise_level: float = 0.05,
            gamma: float = 1.,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 2e-2,
            alpha: float = 0.005,
            target_update_period: int = 3,
            log_dir=os.path.join(root, "logs", "epg_cnr_optimizer"),
            config_path=os.path.join(root, "config.json"),
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
                fa_range : list[float]
                    Range of optimal and initial flip angles
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
                noise_level : float
                    Noise level for CNR calculation
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
                config_path : str | bytes | os.PathLike
                    Path to config file
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
        self.fa_range = fa_range
        self.fa = float(np.mean(fa_range))
        self.fa_delta = fa_delta
        self.Nfa = Nfa
        self.T1_range_1 = T1_range_1
        self.T1_range_2 = T1_range_2
        self.T2_range_1 = T2_range_1
        self.T2_range_2 = T2_range_2
        self.tr = tr
        self.noise_level = noise_level
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.target_update_period = target_update_period
        self.log_dir = log_dir
        self.config_path = config_path
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

        # Setup logger
        self.setup_logger()

    def init_model(self):
        """Constructs reinforcement learning model

        Neural nets: Fully connected 4-8-8-4
        Loss: L2 (MSE) Loss
        Optimizer: Adam with lr alpha
        """

        # Construct policy net
        self.prediction_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4, 4)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4, 8)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(8, 8)),
            ('relu3', nn.ReLU()),
            ('output', nn.Linear(8, 4))
        ])).to(self.device)
        # Construct target net
        self.target_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4, 4)),
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
        self.logs_fields = ["fa", "cnr", "error"]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def calculate_cnr(self, fa=None):
        """Calculates the CNR using parameters stored in self"""

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

        # Determine CNR
        cnr = (
            abs(np.abs(F0_1.cpu()[-1]) - np.abs(F0_2.cpu()[-1]))
            / self.noise_level
        )

        return float(cnr)

    def calculate_exact_optimum(self):
        """Analytically determine the exact optimum for comparison."""

        # Determine E1 for both tissues
        E1a = np.exp(-self.tr / self.T1_1)
        E1b = np.exp(-self.tr / self.T1_2)

        # Calculate optimal flip angle analytically. Formula retrieved from:
        # Haselhoff EH. Optimization of flip angle for T1 dependent cnr: a
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

        # Return optimal flip angle and optimal cnr
        return optimal_fa, self.calculate_cnr(optimal_fa)

    def set_t1s_from_distributions(self, optimal_fa_list):
        """Find values for T1 of both tissues based on fa_optimal"""

        # Sample an optimal_fa from the list
        fa_idx = random.randint(0, len(optimal_fa_list) - 1)
        optimal_fa = optimal_fa_list[fa_idx]

        # Loop until we find a proper match
        loop = 0
        done = False
        while not done:
            # Check whether we have surpassed the max count
            if loop >= 9999:
                # Display warning
                warnings.warn(
                    "\nT1a/T1b combination for flip angle "
                    f"of {optimal_fa:.2f} [deg] not found!"
                    "\nWe're skipping this flip angle."
                )
                # Replace non-viable flip angle
                optimal_fa_list.pop(fa_idx)
                optimal_fa_list.append(optimal_fa_list[0])
                # Break loop
                break

            # Set T1_1
            T1_1 = random.uniform(
                self.T1_range_1[0], self.T1_range_1[1])

            # Calculate T1_2 based on these parameters.
            T1_2 = self.calculate_2nd_T1(optimal_fa, T1_1)

            # If T1_2 calculation was succesful, remove T1_1 and optimal_fa
            # from the lists and stop loop
            if T1_2 and T1_2 != float('NaN'):
                # Remove value at chosen index from list
                optimal_fa_list.pop(fa_idx)
                # Set T1_1 and T1_2
                self.T1_1 = T1_1
                self.T1_2 = T1_2
                # Set done
                done = True

            # Update loop counter
            loop += 1

        return optimal_fa_list

    def calculate_2nd_T1(self, optimal_fa, T1_1):
        """Calculates T1 of 2nd tissue based on optimal fa and T1_1"""

        # This function is based on some algebra performed on the formula
        # given in calculate_exact_optimum()

        # Calculate optimal_fa in radians
        alpha = optimal_fa * np.pi / 180.

        # Define E1a
        E1a = np.exp(-self.tr / T1_1)

        # Define terms of quadratic formula
        a = (
            4 * np.cos(alpha) ** 2 * E1a ** 2
            + 8 * np.cos(alpha) * E1a ** 2
            - 4 * E1a
            - 8 * np.cos(alpha) ** 2 * E1a
            - 12 * np.cos(alpha) * E1a
            + 4
            + 4 * np.cos(alpha) ** 2
            + 4 * np.cos(alpha)
        )
        b = (
            -4 * E1a ** 2
            - 8 * np.cos(alpha) ** 2 * E1a ** 2
            - 12 * np.cos(alpha) * E1a ** 2
            + 12 * E1a
            + 8 * np.cos(alpha) ** 2 * E1a
            + 16 * np.cos(alpha) * E1a
            - 4
            - 8 * np.cos(alpha)
        )
        c = (
            4 * np.cos(alpha) ** 2 * E1a ** 2
            + 4 * np.cos(alpha) * E1a ** 2
            + 4 * E1a ** 2
            - 8 * np.cos(alpha) * E1a
            - 4 * E1a
        )

        # Define E1b (using quadratic formula)
        if (b ** 2 - 4 * a * c) > 0.:
            E1b = np.array([
                (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
                (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            ])
        else:
            return None

        # Calculate T1_2 if E1b is valid
        if (E1b > 0.).all():
            T1_2 = -self.tr / np.log(E1b)
        elif (E1b > 0.).any() and not (E1b > 0.).all():
            E1b = E1b[E1b > 0.]
            T1_2 = -self.tr / np.log(E1b)
        else:
            return None

        # Return T1_2 if in proper range
        if (self.T1_range_2[0] < T1_2 < self.T1_range_2[1]).any():
            # Remove non-valid T1_2 values
            T1_2 = T1_2[self.T1_range_2[0] < T1_2 < self.T1_range_2[1]]
            T1_2 = float(T1_2[0])
            # REturn T1_2
            return T1_2
        else:
            return None

    def find_best_output(self, step, n_memory=3):
        """Find best solution provided by model in final n steps"""

        # Calculate recent memory length
        memory_len = min(step + 1, n_memory)
        # Retrieve recent memory
        recent_memory = list(self.memory)[-memory_len:]
        # Store as transitions
        recent_transitions = self.Transition(*zip(*recent_memory))
        # Extract states
        recent_states = np.array(torch.cat(recent_transitions.state).cpu())

        # Extract flip angles
        recent_fa = np.delete(
            np.delete(
                recent_states,
                np.arange(0, recent_states.size, 2)
            ),
            np.arange(1, recent_states.size // 2, 2)
        )
        # Extract cnr
        recent_cnr = np.delete(
            np.delete(
                recent_states,
                np.arange(1, recent_states.size, 2)
            ),
            np.arange(1, recent_states.size // 2, 2)
        )

        # Find max cnr and respective flip angle
        max_idx = np.argmax(recent_cnr)
        best_fa = recent_fa[max_idx]
        best_cnr = recent_cnr[max_idx]

        # Find step number that gave the best cnr
        best_step = step - memory_len + max_idx

        # Return best fa and cnr + number of best step
        return float(best_fa), float(best_cnr), int(best_step)

    def step(self, old_state, action, episode_i, step_i):
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

        # Run simulations and update state
        state = torch.tensor(
            [
                self.calculate_cnr(), self.fa,            # New cnr, fa
                float(old_state[0]), float(old_state[1])  # Old cnr, fa
            ],
            device=self.device
        )

        # Define reward as either +/- 1 for increase or decrease in cnr
        if state[0] > old_state[0]:
            reward_float = 1.0
        else:
            reward_float = -1.0

        # Scale reward with cnr difference
        if float(old_state[0]) == 0.:
            # If old_state signal is 0, set reward gain to 30
            reward_gain = 30.
        else:
            # Calculate relative cnr difference and derive reward gain
            cnr_diff = abs(state[0] - old_state[0]) / old_state[0]
            reward_gain = cnr_diff * 100.

            # If reward gain is lower than 0.5, use 0.5
            # We do this to prevent disappearing rewards near the optimum
            if reward_gain < 0.5: reward_gain = 0.5
            # If reward gain is higher than 30, use 30
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
                np.delete(
                    recent_states,
                    np.arange(0, recent_states.size, 2)
                ),
                np.arange(0, recent_states.size // 2, 2)
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

        # Log this step (scalars)
        loop_type = 'train' if self.train else 'test'

        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_{loop_type}_episode_{episode_i + 1}",
            value=float(state[1]),
            step=step_i
        )
        self.logger.log_scalar(
            field="cnr",
            tag=f"{self.logs_tag}_{loop_type}_episode_{episode_i + 1}",
            value=float(state[0]),
            step=step_i
        )

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

        # Set 'train' attribute
        self.train = train

        # Print some info
        if train:
            print("\n===== Running training loop =====\n")
        else:
            print("\n======= Running test loop =======\n")

        # Create list of initial and optimal flip angles
        # (uniformly distributed in range)
        initial_fa = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

        optimal_fa = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

        # Create lists of T2s for both tissues
        # (uniformly distributed in range)
        T2_list_1 = list(np.linspace(
            self.T2_range_1[0], self.T2_range_1[1], self.n_episodes
        ))
        T2_list_2 = list(np.linspace(
            self.T2_range_2[0], self.T2_range_2[1], self.n_episodes
        ))

        # Loop over episodes
        for episode in range(self.n_episodes) if train else range(20):
            # Print some info
            if self.verbose:
                print(
                    f"\n=== Episode {episode + 1:3d}/"
                    f"{self.n_episodes if train else 20:3d} ==="
                )
            # Reset done and tick counter
            done = False
            tick = 0

            # Set initial flip angle, T1 and T2 for this episode
            # If train, take them from the uniform distributions
            # If test (not train), take them randomly in a range
            if train:
                # Set initial flip angle. Here, we randomly sample from the
                # uniformly distributed list we created earlier.
                self.fa = float(initial_fa.pop(
                    random.randint(0, len(initial_fa) - 1)
                ))
                # Set the T1s for this episode. Here, we randomly sample
                # T1_1 from the uniform distribution and then calculate T1_2
                # based on the desired optimum flip angle
                optimal_fa = \
                    self.set_t1s_from_distributions(optimal_fa)

                # Set T2s for this episode. We randomly sample these
                # from the previously definded uniform distributions for both
                # tissues.
                self.T2_1 = float(T2_list_1.pop(
                    random.randint(0, len(T2_list_1) - 1)
                ))
                self.T2_2 = float(T2_list_2.pop(
                    random.randint(0, len(T2_list_2) - 1)
                ))
            else:
                # If in test mode, take flip angle, T1, T2 randomly.
                # We do this to provide a novel testing environment.
                self.fa = random.uniform(
                    self.fa_range[0], self.fa_range[1]
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
            cnr = self.calculate_cnr()
            # Set initial state (cnr, fa, previous cnr, previous fa)
            state = torch.tensor(
                [cnr, self.fa, 0., 0.],
                device=self.device
            )

            # # Print some info on the specific environment used this episode.
            optimal_angle, optimal_cnr = self.calculate_exact_optimum()
            print(
                "\n-----------------------------------"
                f"\nT1a={self.T1_1:.4f}s; T2a={self.T2_1:.4f}s; "
                f"T1b={self.T1_2:.4f}s; T2b={self.T2_2:.4f}s"
                f"\nInitial alpha:\t\t{self.fa:4.1f} [deg]"
                f"\nOptimal alpha:\t\t{optimal_angle:4.1f} [deg]"
                f"\nOptimal CNR:\t\t{optimal_cnr:4.2f} [-]"
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
                next_state, reward, done = self.step(
                    state, action, episode, tick
                )
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
                    f" - CNR: {float(state[0]):5.3f}"
                    " - Reward: "
                    "" + color_str + f"{float(reward):5.1f}" + end_str
                )
                if bool(done):
                    print("Stopping criterion met")

            # Print some info on error relative to theoretical optimum
            found_fa, found_cnr, best_step = self.find_best_output(tick)

            if found_cnr == 0.:
                relative_cnr_error = 100.
            else:
                relative_cnr_error = abs(
                    optimal_cnr - found_cnr
                ) * 100. / found_cnr

            print(
                f"Actual error (step {best_step:2d}): "
                f"(fa) {abs(found_fa - optimal_angle):4.1f} deg",
                f"; (cnr) {relative_cnr_error:5.2f}%"
            )

            if train:
                # Log episode
                self.logger.log_scalar(
                    field="fa",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(found_fa),
                    step=episode
                )
                self.logger.log_scalar(
                    field="cnr",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(found_cnr),
                    step=episode
                )
                self.logger.log_scalar(
                    field="error",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=min(float(relative_cnr_error) / 100., 1.),
                    step=episode
                )

                # Optimize prediction/policy model
                self.optimize_model(self.batch_size)

                # Update target model
                if episode % self.target_update_period == 0:
                    self.update_target()

                # Backup model
                torch.save({
                    'prediction_state_dict': self.prediction_net.state_dict(),
                    'target_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, self.model_path)

                # Update epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    optimizer = ContrastOptimizer()
    optimizer.run()
    optimizer.run(train=False)

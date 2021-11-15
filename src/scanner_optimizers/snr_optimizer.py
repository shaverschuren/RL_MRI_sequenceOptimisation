"""SNR Optimizer

This module implements a reinforcement model that
optimizes the flip angle for SNR in a single phantom."""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
from typing import Union                                    # noqa: E402
import json                                                 # noqa: E402
import time                                                 # noqa: E402
from datetime import datetime                               # noqa: E402
import h5py                                                 # noqa: E402
from collections import namedtuple, OrderedDict, deque      # noqa: E402
import random                                               # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn as nn                                       # noqa: E402
import torch.optim as optim                                 # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from util import loggers                                    # noqa: E402


class SNROptimizer():
    """
    A class to represent a reinforcement model that
    optimizes the flip angle for SNR in a single phantom.
    """

    def __init__(
            self,
            n_episodes: int = 50,
            n_ticks: int = 30,
            batch_size: int = 32,
            epochs_per_episode: int = 10,
            memory_done_criterion: int = 15,
            n_done_criterion: int = 3,
            fa_range: list[float] = [20., 60.],
            fa_delta: float = 1.0,
            gamma: float = 1.,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 5e-2,
            alpha: float = 0.005,
            target_update_period: int = 3,
            log_dir=os.path.join(root, "logs", "snr_optimizer"),
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
        Also, provide paths to file locations on the server
        we use to communicate to the scanner.
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

        # Read config file
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Define communication paths
        self.txt_path = self.config["param_loc"]
        self.lck_path = self.txt_path + ".lck"
        self.data_path = self.config["data_loc"]

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

        # Define datafields
        self.logs_fields = ["fa", "snr", "img"]

        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

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
        # Extract snr
        recent_snr = np.delete(
            np.delete(
                recent_states,
                np.arange(1, recent_states.size, 2)
            ),
            np.arange(1, recent_states.size // 2, 2)
        )

        # Find max snr and respective flip angle
        max_idx = np.argmax(recent_snr)
        best_fa = recent_fa[max_idx]
        best_snr = recent_snr[max_idx]

        # Find step number that gave the best snr
        best_step = step - memory_len + max_idx

        # Return best fa and snr + number of best step
        return float(best_fa), float(best_snr), int(best_step)

    def perform_scan(self, fa=None):
        """Perform scan by passing parameters to scanner"""

        # Set flip angle we'll communicate to the scanner
        if not fa:
            fa = self.fa

        # Write new flip angle to appropriate location
        with open(self.lck_path, 'w') as txt_file:
            txt_file.write(f"{int(fa)}")
        os.system(f"mv {self.lck_path} {self.txt_path}")

        # Wait for image to come back by checking the data file
        while not os.path.isfile(self.data_path):
            time.sleep(0.1)

        # When the image is returned, load it and store the results
        with h5py.File(self.data_path, "r") as f:
            img = np.asarray(f['/img'])

        # Remove the data file
        os.remove(self.data_path)

        return img

    def step(self, old_state, action, episode_i, step_i):
        """Run step of the environment simulation

        - Perform selected action
        - Wait for image to come back
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

        # Scan and read image
        img = self.perform_scan()

        # Calculate SNR
        snr = np.mean(img) / np.std(img)

        # Update state
        state = torch.tensor(
            [
                float(snr), float(self.fa),               # New snr, fa
                float(old_state[0]), float(old_state[1])  # Old snr, fa
            ],
            device=self.device
        )

        # Define reward as either +/- 1 for increase or decrease in snr
        if state[0] > old_state[0]:
            reward_float = 1.0
        else:
            reward_float = -1.0

        # Scale reward with snr difference
        if float(old_state[0]) == 0.:
            # If old_state signal is 0, set reward gain to 30
            reward_gain = 30.
        else:
            # Calculate relative snr difference and derive reward gain
            snr_diff = abs(state[0] - old_state[0]) / old_state[0]
            reward_gain = snr_diff * 100.

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

        # Log this step (scalars + image)
        loop_type = 'train' if self.train else 'test'

        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_{loop_type}_episode_{episode_i + 1}",
            value=float(self.fa),
            step=step_i
        )
        self.logger.log_scalar(
            field="snr",
            tag=f"{self.logs_tag}_{loop_type}_episode_{episode_i + 1}",
            value=float(snr),
            step=step_i
        )
        self.logger.log_image(
            field="img",
            tag=f"{self.logs_tag}_{loop_type}_episode_{episode_i + 1}",
            image=np.array(img),
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

        # Create list of initial flip angles
        # (uniformly distributed in range)
        initial_fa = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
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
                # uniformly distributed list we created earlier.
                self.fa = float(initial_fa.pop(
                    random.randint(0, len(initial_fa) - 1)
                ))
            else:
                # If in test mode, take flip angle randomly.
                # We do this to provide a novel testing environment.
                self.fa = random.uniform(
                    self.fa_range[0], self.fa_range[1]
                )

            # Scan and read initial image
            img = self.perform_scan()

            # Calculate SNR
            snr = np.mean(img) / np.std(img)

            # Update state
            state = torch.tensor(
                [
                    float(snr), float(self.fa), 0., 0.
                ],
                device=self.device
            )

            # Print some info on the specific environment used this episode.
            print(
                "\n-----------------------------------"
                f"\nInitial alpha:\t\t{self.fa:4.1f} [deg]"
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
                    f" - snr: {float(state[0]):5.2f}"
                    " - Reward: "
                    "" + color_str + f"{float(reward):5.1f}" + end_str
                )
                if bool(done):
                    print("Stopping criterion met")

            # Print some info on error relative to theoretical optimum
            found_fa, found_snr, best_step = self.find_best_output(tick)

            print(
                f"Optimal results (step {best_step:2d}): "
                f"(fa) {found_fa:4.1f} deg",
                f"; (snr) {found_snr:5.2f}"
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
                    field="snr",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(found_snr),
                    step=episode
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
    optimizer = SNROptimizer()
    optimizer.run()
    optimizer.run(train=False)

"""SNR Optimizer (DDPG)

This module implements a reinforcement model that
optimizes the flip angle for multiple (simulated) tissues
to maximize signal. For this purpose, we change the T1/T2
for each episode. A DDPG-based method is used.
"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
from typing import Union                                    # noqa: E402
from datetime import datetime                               # noqa: E402
from collections import namedtuple                          # noqa: E402
import random                                               # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.optim as optim                                 # noqa: E402
from epg_simulator.python import epg                        # noqa: E402
from util import model, training, loggers                   # noqa: E402


class SNROptimizer():
    """
    A class to represent an optimizer model for
    the flip angle to maximize the SNR of multiple
    simulated tissues. For this purpose, we change the T1/T2
    for each episode. A DDPG method is used for the RL part.
    """

    def __init__(
            self,
            n_episodes: int = 2000,
            n_ticks: int = 100,
            batch_size: int = 128,
            n_done_criterion: int = 5,
            fa_range: list[float] = [20., 60.],
            Nfa: int = 100,
            T1_range: list[float] = [0.100, 2.500],
            T2_range: list[float] = [0.005, 0.100],
            tr: float = 0.050,
            noise_level: float = 0.05,
            gamma: float = 0.99,  # 1.,
            epsilon: float = 1.,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1. - 1e-2,  # 1. - 2e-3,
            actor_alpha: float = 1e-4,  # 1e-4,
            critic_alpha: float = 1e-3,  # 1e-3,
            tau: float = 1e-2,
            log_dir=os.path.join(root, "logs", "epg_snr_optimizer_ddpg"),
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
                    Number of minimal steps before 'done' verdict is accepted
                fa_range : list[float]
                    Range of optimal and initial flip angles
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
                    Noise level for snr calculation
                gamma : float
                    Discount factor for Q value calculation
                epsilon : float
                    Initial epsilon (factor used for exploration regulation)
                epsilon_min : float
                    Minimal epsilon
                epsilon_decay : float
                    Epsilon decay factor
                actor_alpha : float
                    Learning rate for Adam optimizer for actor model
                critic_alpha : float
                    Learning rate for Adam optimizer for critic model
                tau : float
                    Delay factor for target net. Should be in [0-1]
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
        self.n_done_criterion = n_done_criterion
        self.fa_range = fa_range
        self.fa = float(np.mean(fa_range))
        self.Nfa = Nfa
        self.T1_range = T1_range
        self.T2_range = T2_range
        self.tr = tr
        self.noise_level = noise_level
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self.tau = tau
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
        self.transition_contents = (
            'state', 'action', 'reward', 'next_state', 'done'
        )
        self.Transition = namedtuple(
            'Transition', self.transition_contents
        )
        self.memory = training.LongTermMemory(
            capacity=10000,
            transition_contents=self.transition_contents
        )

        # Setup environment
        self.init_env()
        # Setup logger
        self.init_logger()
        # Setup model
        self.init_model()

    def init_env(self):
        """Constructs the environment

        Includes action space and noise function.
        The actual simulation "environment" is already implemented
        in the epg code.
        """

        # Define action space
        # (change flip angle), with:
        # . min=-1.0 : (decrease fa with -fa)
        # . max=+1.0 : (increase fa with +fa)
        # (done), with:
        # . min=-1.0 : (not done)
        # . max=+1.0 : (done)
        self.action_space = np.array([
            [-1.0, 1.0], [-1.0, 1.0]
        ])

    def init_model(self):
        """Constructs reinforcement learning model
        TODO: Try in-256-1
        Neural nets: Fully connected 4-16-64-32-1 and 5-16-64-32-1
        Loss: L2 (MSE) Loss
        Optimizer: Adam with lr alpha
        """

        # Define model architecture
        actor_layers, critic_layers = ([4, 16, 64, 32, 2], [6, 16, 64, 32, 2])
        actor_activation_funcs, critic_activation_funcs = (
            ['relu', 'relu', 'relu', 'relu', 'tanh'],
            ['relu', 'relu', 'relu', 'relu', 'none']
        )

        # Construct actor models (network + target network)
        self.actor = model.FullyConnectedModel(
            actor_layers,
            actor_activation_funcs,
            self.device
        )
        self.actor_target = model.FullyConnectedModel(
            actor_layers,
            actor_activation_funcs,
            self.device
        )
        # Construct critic models (network + target network)
        self.critic = model.FullyConnectedModel(
            critic_layers,
            critic_activation_funcs,
            self.device
        )
        self.critic_target = model.FullyConnectedModel(
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
            self.actor.parameters(), lr=self.actor_alpha, weight_decay=1e-2
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_alpha, weight_decay=1e-2
        )

        # Setup criterion
        self.critic_criterion = torch.nn.MSELoss()

    def init_logger(self):
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
        self.logs_fields = [
            "fa", "fa_norm", "snr", "error", "done", "epsilon",
            "actor_fa", "action_fa", "critic_loss", "policy_loss"
        ]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def norm_parameters(self):
        """Update normalized scan parameters"""

        # Perform normalisation for all parameters
        # Only fa for now
        self.fa_norm = float(
            (self.fa - 0.)
            / (180. - 0.)
        )

    def calculate_snr(self, fa=None):
        """Calculates the SNR using parameters stored in self"""

        # Select flip angle
        if not fa:
            fa = self.fa
        else:
            fa = float(fa)

        # Run simulations
        F0, _, _ = epg.epg_as_torch(
            self.Nfa, fa, self.tr,
            self.T1, self.T2, device=self.device
        )

        # Determine snr
        snr = (
            np.abs(F0.cpu()[-1])
            / self.noise_level
        )

        return float(snr)

    def calculate_exact_optimum(self):
        """Analytically determine the exact optimum for comparison."""

        # Calculate Ernst angle
        ernst_angle = np.arccos(np.exp(-self.tr / self.T1)) * 180. / np.pi

        # Calculate and return SNR
        return ernst_angle, self.calculate_snr(ernst_angle)

    def set_t1_from_distribution(self, optimal_fa_list):
        """Find values for T1 of tissue based on ernst angle"""

        # Sample an optimal_fa from the list
        fa_idx = random.randint(0, len(optimal_fa_list) - 1)
        optimal_fa = optimal_fa_list.pop(fa_idx)

        # Convert to radians
        fa_rad = optimal_fa * np.pi / 180.

        # Calculate tissue T1
        self.T1 = -self.tr / np.log(np.cos(fa_rad))

        return optimal_fa_list

    def find_best_output(self, n_memory=5):
        """Find best solution provided by model in final n steps"""

        # Calculate recent memory length
        memory_len = min(self.tick + 1, n_memory)
        # Retrieve recent memory
        recent_memory = self.memory.get_recent_memory(memory_len)
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
        best_step = self.tick - memory_len + max_idx

        # Return best fa and snr + number of best step
        return float(best_fa), float(best_snr), int(best_step)

    def step(self, old_state, action):
        """Run step of the environment simulation

        - Perform selected action
        - Run EPG simulation
        - Update state
        - Update reward
        - Update done
        """

        # Check if action in action space
        action_min = np.min(self.action_space, axis=1)
        action_max = np.max(self.action_space, axis=1)
        action_np = action.detach().numpy()
        if (
            (action_min <= action_np).all() and (action_np <= action_max).all()
        ):
            # Adjust flip angle
            delta = float(action[0]) * self.fa
            self.fa += delta / 2  # TODO: Think about this ...
            # Correct for flip angle out of bounds
            if self.fa < 0.0: self.fa = 0.0
            if self.fa > 180.0: self.fa = 180.0

            # Set done
            done = torch.tensor(int(float(action[1]) < 0.), device=self.device)

            # Update normalized scan parameters
            self.norm_parameters()
        else:
            raise ValueError("Action not in action space")

        # Run simulations and update state
        state = torch.tensor(
            [
                self.calculate_snr(), self.fa_norm,       # New snr, fa
                float(old_state[0]), float(old_state[1])  # Old snr, fa
            ],
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
            reward_gain = 20.
        else:
            # Calculate relative signal difference and derive reward gain
            snr_diff = abs(state[0] - old_state[0]) / old_state[0]
            reward_gain = snr_diff * 100.

            # If reward is lower than 0.01, penalise
            # the system for taking steps that are too small.
            if reward_gain < 0.01:
                reward_float = -1.0
                reward_gain = 0.05
            # If reward gain is higher than 20, use 20
            # We do this to prevent blowing up rewards near the edges
            if reward_gain > 20.: reward_gain = 20.

        # If reward is negative, increase gain
        if reward_float < 0.:
            reward_gain *= 2.0

        # Define reward
        reward_float *= reward_gain

        # Scale reward with step_i (faster improvement yields bigger rewards)
        # Only scale the positives, though.
        if reward_float > 0.:
            reward_float *= np.exp(-self.tick / (self.n_ticks / 5.))

        # Update reward based on 'done' verdict. Here, giving 'done' too
        # early will drastically decrease the given reward.
        if bool(done):
            reward_float -= max(0, 1 - self.tick / self.n_done_criterion) * 20.

        # Store reward in tensor
        reward = torch.tensor(
            [float(reward_float)], device=self.device
        )

        # # Set done
        # if self.tick >= self.n_done_criterion:
        #     # Retrieve recent memory
        #     recent_memory = \
        #         self.memory.get_recent_memory(self.n_done_criterion)
        #     # Store as transitions
        #     recent_transitions = self.Transition(*zip(*recent_memory))
        #     # Extract flip angles
        #     recent_states = np.array(torch.cat(recent_transitions.state).cpu())
        #     recent_snr = np.delete(
        #         np.delete(
        #             recent_states,
        #             np.arange(1, recent_states.size, 2)
        #         ),
        #         np.arange(0, recent_states.size // 2, 2)
        #     )
        #     # Append current/last SNR
        #     recent_snr = np.append(recent_snr, float(old_state[0]))

        #     # Check for improvement in SNR over recent records
        #     if not (recent_snr[1:] > recent_snr[0]).any():
        #         done = torch.tensor(1, device=self.device)
        #     else:
        #         done = torch.tensor(0, device=self.device)

        # else:
        #     # Recent memory too short: We're not done yet
        #     done = torch.tensor(0, device=self.device)

        # Log this step (scalars)
        loop_type = 'train' if self.train else 'test'

        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_{loop_type}_episode_{self.episode + 1}",
            value=float(self.fa),
            step=self.tick
        )
        self.logger.log_scalar(
            field="fa_norm",
            tag=f"{self.logs_tag}_{loop_type}_episode_{self.episode + 1}",
            value=float(self.fa_norm),
            step=self.tick
        )
        self.logger.log_scalar(
            field="snr",
            tag=f"{self.logs_tag}_{loop_type}_episode_{self.episode + 1}",
            value=float(state[0]),
            step=self.tick
        )

        return state, reward, done

    def get_action(self, state):
        """Get action

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
            if self.train else 0.
        )
        noisy_action = np.clip(
            pure_action + noise,
            np.min(self.action_space, axis=1),
            np.max(self.action_space, axis=1)
        )
        # Print some info
        if self.verbose:
            print(f"Model: {float(pure_action[0]):5.2f}", end="")

        # Log actor output and eventual action
        loop_type = 'train' if self.train else 'test'

        self.logger.log_scalar(
            field="actor_fa",
            tag=f"{self.logs_tag}_{loop_type}_episode_{self.episode + 1}",
            value=float(pure_action[0]),
            step=self.tick
        )
        self.logger.log_scalar(
            field="action_fa",
            tag=f"{self.logs_tag}_{loop_type}_episode_{self.episode + 1}",
            value=float(noisy_action[0]),
            step=self.tick
        )

        return torch.FloatTensor(noisy_action, device=self.device)

    def update(self):
        """Update model"""

        # Sample batch from memory
        states, actions, rewards, next_states, _ = \
            self.memory.sample(self.batch_size)
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

        # Log losses for this step (if train)
        if self.train:
            self.logger.log_scalar(
                field="critic_loss",
                tag=f"{self.logs_tag}_train_losses",
                value=float(critic_loss),
                step=self.train_tick
            )
            self.logger.log_scalar(
                field="policy_loss",
                tag=f"{self.logs_tag}_train_losses",
                value=float(policy_loss),
                step=self.train_tick
            )

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

        # Create lists of T2s
        # (uniformly distributed in range)
        T2_list = list(np.linspace(
            self.T2_range[0], self.T2_range[1], self.n_episodes
        ))

        # Set training step counter
        if self.train: self.train_tick = 0

        # If test, tighten done criterion
        if not self.train: self.n_done_criterion = 3

        # Loop over episodes
        for self.episode in range(self.n_episodes) if train else range(20):
            # Print some info
            if self.verbose:
                print(
                    f"\n=== Episode {self.episode + 1:3d}/"
                    f"{self.n_episodes if train else 20:3d} ==="
                )
            # Reset done and tick counter
            done = False
            self.tick = 0

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
                    self.set_t1_from_distribution(optimal_fa)

                # Set T2 for this episode. We randomly sample this
                # from the previously definded uniform distribution.
                self.T2 = float(T2_list.pop(
                    random.randint(0, len(T2_list) - 1)
                ))
            else:
                # If in test mode, take flip angle, T1, T2 randomly.
                # We do this to provide a novel testing environment.
                self.fa = random.uniform(
                    self.fa_range[0], self.fa_range[1]
                )
                self.T1 = random.uniform(
                    self.T1_range[0], self.T1_range[1])
                self.T2 = random.uniform(
                    self.T2_range[0], self.T2_range[1])

            # TODO: Remove (debugging)
            self.T1 = 0.19
            self.T2 = 0.1

            # Normalize parameters
            self.norm_parameters()

            # Run initial simulation
            snr = self.calculate_snr()
            # Set initial state (snr, fa, previous snr, previous fa)
            state = torch.tensor(
                [snr, self.fa_norm, 0., 0.],
                device=self.device
            )

            # Print some info on the specific environment used this episode.
            ernst_angle, optimal_snr = self.calculate_exact_optimum()
            print(
                "\n-----------------------------------"
                f"\nRunning episode with T1={self.T1:.4f}s & T2={self.T2:.4f}s"
                f"\nInitial alpha:\t\t{self.fa:4.1f} [deg]"
                f"\nErnst angle:\t\t{ernst_angle:4.1f} [deg]"
                f"\nOptimal snr:\t\t{optimal_snr:4.2f} [-]"
                "\n-----------------------------------"
            )

            # Loop over steps/ticks
            while self.tick < self.n_ticks and not bool(done):
                # Print some info
                print(f"Step {self.tick + 1:3d}/{self.n_ticks:3d} - ", end="")

                # Get action
                action = self.get_action(state)

                # Simulate step
                next_state, reward, done = self.step(state, action)
                # Add to memory
                self.memory.push(state, action, reward, next_state, done)
                # Update state
                state = next_state
                # Update tick counter
                self.tick += 1
                if self.train: self.train_tick += 1

                # Update model
                if train and len(self.memory) >= self.batch_size:
                    self.update()

                # Print some info
                color_str = "\033[92m" if reward > 0. else "\033[91m"
                end_str = "\033[0m"
                print(
                    f" - Action: {float(action[0]):5.2f}"
                    f" - FA: {float(self.fa):5.1f}"
                    f" - SNR: {float(state[0]):5.2f}"
                    " - Reward: "
                    "" + color_str + f"{float(reward):5.1f}" + end_str
                )
                if bool(done):
                    print("Stopping criterion met")

            # Print some info on error relative to theoretical optimum
            found_fa, found_snr, best_step = self.find_best_output()

            if found_snr == 0.:
                relative_snr_error = 100.
            else:
                relative_snr_error = abs(
                    optimal_snr - found_snr
                ) * 100. / found_snr

            print(
                f"Actual error (step {best_step:2d}): "
                f"(fa) {abs(found_fa - ernst_angle):4.1f} deg",
                f"; (snr) {relative_snr_error:5.2f}%"
            )

            if train:
                # Log episode
                self.logger.log_scalar(
                    field="fa",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(found_fa),
                    step=self.episode
                )
                self.logger.log_scalar(
                    field="snr",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(found_snr),
                    step=self.episode
                )
                self.logger.log_scalar(
                    field="error",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=min(float(relative_snr_error) / 100., 1.),
                    step=self.episode
                )
                self.logger.log_scalar(
                    field="done",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(self.tick + 1),
                    step=self.episode
                )
                self.logger.log_scalar(
                    field="epsilon",
                    tag=f"{self.logs_tag}_train_episodes",
                    value=float(self.epsilon),
                    step=self.episode
                )

                # Backup model
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'actor_target': self.actor_target.state_dict(),
                    'critic_target': self.critic_target.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict()
                }, self.model_path)

                # Update epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    optimizer = SNROptimizer()
    optimizer.run()
    optimizer.run(train=False)

"""Module implementing several RL algorithms

Here, we give implementations of several algorithms, namely:

- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- RDPG (Recurrent Deterministic Policy Gradient)
"""

import os
from typing import Union
import numpy as np
from datetime import datetime
import torch
import json
from optimizer import agents, environments
from util import training, loggers


class DQN(object):
    """Class to represent a DQN optimizer algorithm"""

    def __init__(
            self,
            env,
            log_dir: Union[str, os.PathLike],
            n_episodes: int = 750,
            n_ticks: int = 30,
            batch_size: int = 64,
            pretrained_path: Union[str, os.PathLike, None] = None,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            env : optimizer.environments.* object
                Environment to be optimized
            log_dir : str | os.PathLike
                Directory in which we store the logs
            n_episodes : int
                Number of episodes we'll run
            n_ticks : int
                Maximum number of ticks in an episode
            batch_size: int
                Batch size used for optimization
            Pretrained_path : str | os.PathLike | None
                Path to pretrained model
        """

        # Build attributes
        self.env = env
        self.metric = self.env.metric
        self.log_dir = log_dir
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size
        self.pretrained_path = pretrained_path

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Setup agent
        self.agent = agents.DQNAgent(
            env.n_states, env.n_actions,
            hidden_layers=[8, 8] if self.metric == "snr" else [64, 256, 32],
            epsilon_decay=1. - (4. / float(self.n_episodes))
        )
        if pretrained_path: self.agent.load(pretrained_path)

        # Setup memory
        self.memory = training.LongTermMemory(10000)
        # Setup logger
        self.setup_logger()

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
        self.logs_fields = [
            "img", "fa", "fa_norm", self.metric, "error", "done", "epsilon"
        ]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def log_step(self, state, action, reward, next_state, done):
        """Log a single step"""

        # Print some info regarding this step
        reward_color = "\033[92m" if reward > 0. else "\033[91m"
        if self.agent.action_mode == "exploration":
            action_mode = "\033[93mR\033[0m"
        elif self.agent.action_mode == "exploitation":
            action_mode = "\033[94mP\033[0m"
        else:
            raise RuntimeError()
        end_str = "\033[0m"

        print(
            f"Step {self.tick + 1:3d}/{self.n_ticks:3d} - "
            f"{action_mode} - "
            f"Action: {int(action):2d} - "
            f"FA: {float(next_state[1]) * 90.:5.1f} - "
            f"{self.metric.upper()}: "
            f"{float(next_state[0]) * self.env.metric_calibration:5.2f} -"
            " Reward: "
            "" + reward_color + f"{float(reward):6.2f}" + end_str
        )

        # Log this step to tensorboard
        run_type = "train" if self.train else "test"

        if (
            isinstance(self.env, environments.ScannerEnv)
            and hasattr(self.env, 'recent_img')
        ):
            self.logger.log_image(
                field="img",
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                image=np.array(self.env.recent_img),
                step=self.tick
            )

        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.fa),
            step=self.tick
        )
        self.logger.log_scalar(
            field="fa_norm",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.fa_norm),
            step=self.tick
        )
        self.logger.log_scalar(
            field=self.metric,
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(getattr(self.env, self.metric)),
            step=self.tick
        )
        self.logger.log_scalar(
            field="done",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.done),
            step=self.tick
        )

    def log_episode(self):
        """Log a single episode"""

        # Extract recent memory
        recent_memory = self.memory.get_recent_memory(5)
        recent_states = [transition.next_state for transition in recent_memory]
        recent_metrics = [float(state[0]) for state in recent_states]
        recent_fa = [float(state[1]) for state in recent_states]

        # Find "best" fa/metric in recent memory
        best_idx = np.argmax(recent_metrics)
        best_metric = float(recent_metrics[best_idx])
        best_fa = float(recent_fa[best_idx]) * 90.

        # Log scalars
        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_train_episodes",
            value=best_fa,
            step=self.episode
        )
        self.logger.log_scalar(
            field=self.metric,
            tag=f"{self.logs_tag}_train_episodes",
            value=best_metric,
            step=self.episode
        )
        self.logger.log_scalar(
            field="epsilon",
            tag=f"{self.logs_tag}_train_episodes",
            value=self.agent.epsilon,
            step=self.episode
        )

        # If theoretical optimum is known, log the error
        if isinstance(self.env, environments.SimulationEnv):
            # Find optimal fa and snr/cnr
            optimal_fa = self.env.optimal_fa
            optimal_metric = getattr(self.env, f"optimal_{self.metric}")
            # Calculate relative SNR/CNR error
            if best_metric == 0.:
                relative_error = 1.
            else:
                relative_error = abs(
                    optimal_metric - best_metric
                ) / best_metric

            self.logger.log_scalar(
                field="error",
                tag=f"{self.logs_tag}_train_episodes",
                value=min(relative_error, 1.),
                step=self.episode
            )

    def verbose_episode(self):
        """Prints some info about the current episode"""

        # Assemble print string
        if isinstance(self.env, environments.SimulationEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 10:3d}"
                " ==========\n"
                "\n-----------------------------------"
                "\nRunning episode with "
            )

            if self.metric == "snr":
                print_str += f"T1={self.env.T1:.4f}s & T2={self.env.T2:.4f}s"
            elif self.metric == "cnr":
                print_str += (
                    f"T1a={self.env.T1_1:.4f}s; T2a={self.env.T2_1:.4f}s; "
                    f"T1b={self.env.T1_2:.4f}s; T2b={self.env.T2_2:.4f}s"
                )
            else:
                RuntimeError()

            print_str += (
                f"\nInitial FA:\t\t{self.env.fa:4.1f} [deg]"
                f"\nOptimal FA:\t\t{self.env.optimal_fa:4.1f} [deg]"
                f"\nOptimal {self.metric.upper()}:\t\t"
                f"{getattr(self.env, f'optimal_{self.metric}'):4.2f} [-]"
                "\n-----------------------------------"
            )
        elif isinstance(self.env, environments.ScannerEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 10:3d}"
                " ==========\n"
                "\n-----------------------------------"
                f"\nInitial FA:\t{self.env.fa:4.1f} [deg]"
                "\n-----------------------------------"
            )
        else:
            print(self.env.__class__)
            raise RuntimeError()

        # Print the string
        print(print_str)

    def run(self, train=True):
        """Run either training or testing loop"""

        # Set training variable
        self.train = train

        # Print some info
        print(
            "\n\n==================== "
            f"Running {'training' if train else 'test'} loop"
            " ====================\n\n"
        )

        # Create training initial condition distributions
        if train:
            self.env.n_episodes = self.n_episodes
            self.env.homogeneous_initialization = True
            self.env.set_homogeneous_dists()
        else:
            self.env.n_episodes = None
            self.env.homogeneous_initialization = False

        # Episode loop
        for self.episode in range(self.n_episodes) if train else range(10):

            # Reset environment
            self.env.reset()

            # Print some info
            self.verbose_episode()

            # Loop over ticks/steps
            for self.tick in range(self.n_ticks):

                # Extract current state
                state = self.env.state

                # Choose action
                action = self.agent.select_action(state, train)

                # Simulate step
                next_state, reward, done = self.env.step(action)

                # Add to memory
                self.memory.push(
                    state, action, reward, next_state, done
                )

                # If training, update model
                if train and self.batch_size <= len(self.memory):
                    batch = self.memory.sample(self.batch_size)
                    self.agent.update(batch)

                # Log step results
                self.log_step(state, action, reward, next_state, done)

                # Check if done
                if done:
                    break

            # Log episode results
            if train:
                self.log_episode()

            # Update epsilon
            self.agent.update_epsilon()

            # Backup model
            self.agent.save(self.model_path)


class DDPG(object):
    """Class to represent a DDPG optimizer"""

    def __init__(
            self,
            env,
            log_dir: Union[str, os.PathLike],
            n_episodes: int = 2000,
            n_ticks: int = 30,
            batch_size: int = 64,
            pretrained_path: Union[str, os.PathLike, None] = None,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            env : optimizer.environments.* object
                Environment to be optimized
            log_dir : str | os.PathLike
                Directory in which we store the logs
            n_episodes : int
                Number of episodes we'll run
            n_ticks : int
                Maximum number of ticks in an episode
            batch_size: int
                Batch size used for optimization
            pretrained_path : str | os.PathLike | None
                Path to pretrained model
        """

        # Build attributes
        self.env = env
        self.metric = self.env.metric
        self.log_dir = log_dir
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size
        self.pretrained_path = pretrained_path

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Setup agent
        self.agent = agents.DDPGAgent(
            env.action_space, env.n_states, env.n_actions,
            epsilon_decay=1. - (4. / float(self.n_episodes))
        )
        if self.pretrained_path: self.agent.load(pretrained_path)

        # Setup memory
        self.memory = training.LongTermMemory(10000)
        # Setup logger
        self.setup_logger()

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
        self.logs_fields = [
            "img", "fa", "fa_norm", self.metric, "error", "done", "epsilon",
            "critic_loss", "policy_loss"
        ]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def log_step(self, state, action, reward, next_state, done):
        """Log a single step"""

        # Print some info regarding this step
        reward_color = "\033[92m" if reward > 0. else "\033[91m"
        end_str = "\033[0m"

        print(
            f"Step {self.tick + 1:3d}/{self.n_ticks:3d} - "
            f"Action: {float(action):5.2f} - "
            f"FA: {float(next_state[1]) * 90.:5.1f} - "
            f"{self.metric.upper()}: "
            f"{float(next_state[0]) * self.env.metric_calibration:5.2f} -"
            " Reward: "
            "" + reward_color + f"{float(reward):6.2f}" + end_str
        )

        # Log this step to tensorboard
        run_type = "train" if self.train else "test"

        if (
            isinstance(self.env, environments.ScannerEnv)
            and hasattr(self.env, 'recent_img')
        ):
            self.logger.log_image(
                field="img",
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                image=np.array(self.env.recent_img),
                step=self.tick
            )

        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.fa),
            step=self.tick
        )
        self.logger.log_scalar(
            field="fa_norm",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.fa_norm),
            step=self.tick
        )
        self.logger.log_scalar(
            field=self.metric,
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(getattr(self.env, self.metric)),
            step=self.tick
        )
        self.logger.log_scalar(
            field="done",
            tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            value=float(self.env.done),
            step=self.tick
        )

        # Log losses (if applicable)
        if (
            self.train
            and hasattr(self, "policy_loss")
            and hasattr(self, "critic_loss")
        ):
            # Create or update training tick
            if not hasattr(self, "train_tick"):
                self.train_tick = 0
            else:
                self.train_tick += 1
            # Log losses
            self.logger.log_scalar(
                field="policy_loss",
                tag=f"{self.logs_tag}_train_losses",
                value=self.policy_loss,
                step=self.train_tick
            )
            self.logger.log_scalar(
                field="critic_loss",
                tag=f"{self.logs_tag}_train_losses",
                value=self.critic_loss,
                step=self.train_tick
            )

    def log_episode(self):
        """Log a single episode"""

        # Extract recent memory
        recent_memory = self.memory.get_recent_memory(5)
        recent_states = [transition.next_state for transition in recent_memory]
        recent_metrics = [float(state[0]) for state in recent_states]
        recent_fa = [float(state[1]) for state in recent_states]

        # Find "best" fa/metric in recent memory
        best_idx = np.argmax(recent_metrics)
        best_metric = (
            float(recent_metrics[best_idx])
            * self.env.metric_calibration
        )
        best_fa = float(recent_fa[best_idx]) * 90.

        # Log scalars
        self.logger.log_scalar(
            field="fa",
            tag=f"{self.logs_tag}_train_episodes",
            value=best_fa,
            step=self.episode
        )
        self.logger.log_scalar(
            field=self.metric,
            tag=f"{self.logs_tag}_train_episodes",
            value=best_metric,
            step=self.episode
        )
        self.logger.log_scalar(
            field="epsilon",
            tag=f"{self.logs_tag}_train_episodes",
            value=self.agent.epsilon,
            step=self.episode
        )

        # If theoretical optimum is known, log the error
        if isinstance(self.env, environments.SimulationEnv):
            # Find optimal fa and snr/cnr
            optimal_fa = self.env.optimal_fa
            optimal_metric = getattr(self.env, f"optimal_{self.metric}")
            # Calculate relative SNR/CNR error
            if best_metric == 0.:
                relative_error = 1.
            else:
                relative_error = abs(
                    optimal_metric - best_metric
                ) / best_metric

            self.logger.log_scalar(
                field="error",
                tag=f"{self.logs_tag}_train_episodes",
                value=min(relative_error, 1.),
                step=self.episode
            )

    def verbose_episode(self):
        """Prints some info about the current episode"""

        # Assemble print string
        if isinstance(self.env, environments.SimulationEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 10:3d}"
                " ==========\n"
                "\n-----------------------------------"
                "\nRunning episode with "
            )

            if self.metric == "snr":
                print_str += f"T1={self.env.T1:.4f}s & T2={self.env.T2:.4f}s"
            elif self.metric == "cnr":
                print_str += (
                    f"T1a={self.env.T1_1:.4f}s; T2a={self.env.T2_1:.4f}s; "
                    f"T1b={self.env.T1_2:.4f}s; T2b={self.env.T2_2:.4f}s"
                )
            else:
                RuntimeError()

            print_str += (
                f"\nInitial FA:\t\t{self.env.fa:4.1f} [deg]"
                f"\nOptimal FA:\t\t{self.env.optimal_fa:4.1f} [deg]"
                f"\nOptimal {self.metric.upper()}:\t\t"
                f"{getattr(self.env, f'optimal_{self.metric}'):4.2f} [-]"
                "\n-----------------------------------"
            )
        elif isinstance(self.env, environments.ScannerEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 10:3d}"
                " ==========\n"
                "\n-----------------------------------"
                f"\nInitial FA:\t{self.env.fa:4.1f} [deg]"
                "\n-----------------------------------"
            )
        else:
            print(self.env.__class__)
            raise RuntimeError()

        # Print the string
        print(print_str)

    def run(self, train=True):
        """Run either training or testing loop"""

        # Set training variable
        self.train = train

        # Print some info
        print(
            "\n\n==================== "
            f"Running {'training' if train else 'test'} loop"
            " ====================\n\n"
        )

        # Create training initial condition distributions
        if train:
            self.env.n_episodes = self.n_episodes
            self.env.homogeneous_initialization = True
            self.env.set_homogeneous_dists()
        else:
            self.env.n_episodes = None
            self.env.homogeneous_initialization = False

        # Episode loop
        for self.episode in range(self.n_episodes) if train else range(10):

            # Reset environment
            self.env.reset()

            # Print some info
            self.verbose_episode()

            # Loop over ticks/steps
            for self.tick in range(self.n_ticks):

                # Extract current state
                state = self.env.state

                # Choose action
                action = self.agent.select_action(state, train)

                # Simulate step
                next_state, reward, done = self.env.step(action)

                # Add to memory
                self.memory.push(
                    state, action, reward, next_state, done
                )

                # If training, update model
                if train and self.batch_size <= len(self.memory):
                    batch = self.memory.sample(self.batch_size)
                    self.policy_loss, self.critic_loss = \
                        self.agent.update(batch)

                # Log step results
                self.log_step(state, action, reward, next_state, done)

                # Check if done
                if done:
                    break

            # Log episode results
            if train:
                self.log_episode()

            # Update epsilon
            self.agent.update_epsilon()

            # Backup model
            self.agent.save(self.model_path)


class RDPG(object):
    """Class to represent an RDPG optimizer

    Some inspiration was taken from Heess et al. (2015)
    : https://arxiv.org/pdf/1512.04455.pdf
    """

    def __init__(
            self,
            env,
            log_dir: Union[str, os.PathLike],
            n_episodes: int = 100000,
            n_ticks: int = 10,  # TODO: 30
            batch_size: int = 64,
            model_done: bool = True,
            single_fa: bool = False,
            pretrained_path: Union[str, os.PathLike, None] = None,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            env : optimizer.environments.* object
                Environment to be optimized
            log_dir : str | os.PathLike
                Directory in which we store the logs
            n_episodes : int
                Number of episodes we'll run
            n_ticks : int
                Maximum number of ticks in an episode
            batch_size: int
                Batch size used for optimization
            model_done: bool
                Whether model should give "done" command
            single_fa : bool
                If True, we only optimize a single flip angle.
                If False, we optimize the entirety of the echo train.
            pretrained_path : str | os.PathLike | None
                Path to pretrained model
        """

        # Build attributes
        self.env = env
        self.metric = self.env.metric
        self.log_dir = log_dir
        self.n_episodes = n_episodes
        self.n_ticks = n_ticks
        self.batch_size = batch_size
        self.model_done = model_done
        self.single_fa = single_fa
        self.pretrained_path = pretrained_path

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Determine n_actions
        if single_fa: n_actions = 2 if model_done else 1
        else: n_actions = env.n_actions + 1 if model_done else env.n_actions
        # Determine n_states
        if single_fa: n_states = 2
        else: n_states = [env.img_shape, env.n_pulses]

        # Setup agent
        self.agent = agents.RDPGAgent(
            env.action_space,
            n_actions=n_actions,
            n_states=n_states,
            single_fa=single_fa,
            epsilon_decay=1. - (10. / float(self.n_episodes))
        )
        if self.pretrained_path: self.agent.load(pretrained_path)

        # Setup memory
        self.memory = training.EpisodicMemory(
            self.n_episodes // 4,
            ('state', 'action', 'reward', 'next_state') if self.single_fa
            else (
                'state_img', 'state_fa', 'action',
                'reward', 'next_state_img', 'next_state_fa'
            )
        )
        # Setup logger
        self.setup_logger()

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

        # Setup dynamic variable checkpoint path
        self.variable_path = os.path.join(self.logs_path, "dynamic_vars.json")

        # Define datafields
        self.logs_fields = [
            "img", self.metric, f"{self.metric}_norm",
            "error", "done", "epsilon", "critic_loss", "policy_loss",
            "n_scans", "reward", "performance"
        ]
        if self.single_fa: self.logs_fields.extend(["fa", "fa_norm"])
        else: self.logs_fields.extend(["theta", "theta_norm"])

        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

        # Setup loss history and "best" model backup feature
        self.loss_history = []
        self.best_loss = None
        self.backup_best = False
        self.best_episode = 0

    def log_initiation(self):
        """Log episode initialisation to tensorboard"""

        # Determine run type
        run_type = "train" if self.train else "test"

        # If in test mode or 1/100 of a train run, log this step
        if (
            run_type == "test"
            or self.episode in list(
                range(0, self.n_episodes, self.n_episodes // 100)
            )
        ):
            # Image
            if (
                isinstance(self.env, environments.ScannerEnv)
                and hasattr(self.env, 'recent_img')
            ):
                self.logger.log_image(
                    field="img",
                    tag=(
                        f"{self.logs_tag}_{run_type}_"
                        f"episode_{self.episode + 1}"
                    ),
                    image=np.array(self.env.recent_img) / 5.,
                    step=-1
                )

            # Scalars (first state if applicable -> state0)
            if self.single_fa:
                self.logger.log_scalar(
                    field="fa",
                    tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                    value=float(self.env.fa),
                    step=-1
                )
                self.logger.log_scalar(
                    field="fa_norm",
                    tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                    value=float(self.env.fa_norm),
                    step=-1
                )
            self.logger.log_scalar(
                field=self.metric,
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                value=float(getattr(self.env, self.metric)),
                step=-1
            )
            self.logger.log_scalar(
                field=f"{self.metric}_norm",
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                value=float(getattr(self.env, f"{self.metric}_norm")),
                step=-1
            )

            # Log accuracy (if applicable)
            if isinstance(self.env, environments.SimulationEnv):
                # Extract optimal metric
                optimal_metric = getattr(self.env, f"optimal_{self.metric}")
                # Calculate performance (how much of optimal SNR/CNR is there)
                performance = getattr(self.env, self.metric) / optimal_metric

                # Log the error
                self.logger.log_scalar(
                    field="performance",
                    tag=(
                        f"{self.logs_tag}_{run_type}_"
                        f"episode_{self.episode + 1}"
                    ),
                    value=performance,
                    step=-1
                )

    def log_step(self, state, action, reward, next_state, done):
        """Log a single step"""

        # Print some info regarding this step
        reward_color = "\033[92m" if reward > 0. else "\033[91m"
        end_str = "\033[0m"

        print(
            f"Step {self.tick + 1:3d}/{self.n_ticks:3d} - "
            # f"Action: {float(action[0]):5.2f} - "
            # f"FA: {float(self.env.fa):5.1f} - "
            f"{self.metric.upper()}: "
            f"{float(getattr(self.env, self.metric)):5.2f} -"
            " Reward: "
            "" + reward_color + f"{float(reward):6.3f}" + end_str
        )

        # Log this step to tensorboard
        # (but only for 100 training episodes because of speed issues)
        run_type = "train" if self.train else "test"

        # TODO: Implement proper logging for theta

        if (
            run_type == "test"
            or self.episode in list(
                range(0, self.n_episodes, self.n_episodes // 100)
            )
        ):

            # Image
            if (
                (
                    isinstance(self.env, environments.ScannerEnv)
                    or isinstance(self.env, environments.KspaceEnv)
                ) and hasattr(self.env, 'recent_img')
            ):
                self.logger.log_image(
                    field="img",
                    tag=(
                        f"{self.logs_tag}_{run_type}_"
                        f"episode_{self.episode + 1}"
                    ),
                    image=np.array(self.env.recent_img.cpu()) / 5.,
                    step=self.tick + 1
                )

            # Scalars (current state -> state1)
            # self.logger.log_scalar(
            #     field="fa",
            #     tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            #     value=float(self.env.fa),
            #     step=self.tick
            # )
            # self.logger.log_scalar(
            #     field="fa_norm",
            #     tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
            #     value=float(self.env.fa_norm),
            #     step=self.tick
            # )
            self.logger.log_scalar(
                field=self.metric,
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                value=float(getattr(self.env, self.metric)),
                step=self.tick
            )
            self.logger.log_scalar(
                field=f"{self.metric}_norm",
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                value=float(getattr(self.env, f"{self.metric}_norm")),
                step=self.tick
            )
            self.logger.log_scalar(
                field="reward",
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                value=float(reward),
                step=self.tick
            )
            self.logger.log_scalar(
                field="done",
                tag=f"{self.logs_tag}_{run_type}_episode_{self.episode + 1}",
                value=float(self.env.done),
                step=self.tick
            )

            # Log accuracy (if applicable)
            if isinstance(self.env, environments.SimulationEnv):
                # Extract optimal metric
                optimal_metric = getattr(self.env, f"optimal_{self.metric}")
                # Calculate performance (how much of optimal SNR/CNR is there)
                performance = getattr(self.env, self.metric) / optimal_metric

                # Log the error
                self.logger.log_scalar(
                    field="performance",
                    tag=(
                        f"{self.logs_tag}_{run_type}_"
                        f"episode_{self.episode + 1}"
                    ),
                    value=performance,
                    step=self.tick
                )

    def log_episode(self):
        """Log a single episode"""

        # Extract recent memory
        recent_memory = self.memory.get_recent_memory(6)
        # recent_states = [transition.state for transition in recent_memory]
        # recent_next_states = [
        #     transition.next_state for transition in recent_memory
        # ]
        # recent_metrics = [float(state[0]) for state in recent_next_states]
        # recent_fa = [float(state[1]) for state in recent_next_states]

        # # Find "best" fa/metric in recent memory
        # best_idx = np.argmax(recent_metrics)
        # best_metric = (
        #     float(recent_metrics[best_idx])
        #     * self.env.metric_calibration
        # )
        # best_fa = float(recent_fa[best_idx]) * 90.

        # Find cumulative reward
        previous_trajectory = self.memory.memory[-1]
        rewards = [
            float(transition.reward) for transition in previous_trajectory
        ]
        cumulative_reward = sum(rewards)

        # Log scalars
        # self.logger.log_scalar(
        #     field="fa",
        #     tag=f"{self.logs_tag}_train_episodes",
        #     value=best_fa,
        #     step=self.episode
        # )
        # self.logger.log_scalar(
        #     field=self.metric,
        #     tag=f"{self.logs_tag}_train_episodes",
        #     value=best_metric,
        #     step=self.episode
        # )
        self.logger.log_scalar(
            field="epsilon",
            tag=f"{self.logs_tag}_train_episodes",
            value=self.agent.epsilon,
            step=self.episode
        )
        self.logger.log_scalar(
            field="n_scans",
            tag=f"{self.logs_tag}_train_episodes",
            value=self.tick + 1,
            step=self.episode
        )
        self.logger.log_scalar(
            field="reward",
            tag=f"{self.logs_tag}_train_episodes",
            value=cumulative_reward,
            step=self.episode
        )

        # Log losses (if applicable)
        if (
            self.train
            and hasattr(self, "policy_loss")
            and hasattr(self, "critic_loss")
        ):
            # Log losses in tensorboard
            self.logger.log_scalar(
                field="policy_loss",
                tag=f"{self.logs_tag}_train_episodes",
                value=self.policy_loss,
                step=self.episode
            )
            self.logger.log_scalar(
                field="critic_loss",
                tag=f"{self.logs_tag}_train_episodes",
                value=self.critic_loss,
                step=self.episode
            )
            # Log losses internally
            self.loss_history.append({
                "episode": self.episode,
                "policy_loss": float(self.policy_loss),
                "critic_loss": float(self.critic_loss)
            })
            # Check whether we're currently on the "best" episode yet
            if len(self.loss_history) < 5:
                # Skip this if there isn't enough episodes yet for mean filter
                pass
            else:
                # Calculate mean loss over last five episodes
                mean_loss = 0.

                for i in range(-5, 0):
                    loss_sum = (
                        float(self.loss_history[i]["policy_loss"])
                        + float(self.loss_history[i]["critic_loss"])
                    )
                    mean_loss += loss_sum

                mean_loss /= 5.

                # Update "best" loss value and check whether we're
                # on the "best" episode yet (if applicable)
                if self.best_loss is not None:
                    # Check if better if data is available
                    if mean_loss < self.best_loss:
                        self.best_loss = mean_loss
                        self.backup_best = True
                        self.best_episode = self.episode
                    else:
                        self.backup_best = False
                else:
                    # If no data available yet, just use the most recent one
                    self.best_loss = mean_loss
                    self.backup_best = True

        # If theoretical optimum is known, log the error
        if isinstance(self.env, environments.SimulationEnv):
            # Find optimal fa and snr/cnr
            optimal_fa = self.env.optimal_fa
            optimal_metric = getattr(self.env, f"optimal_{self.metric}")
            # Calculate relative SNR/CNR error
            if best_metric == 0.:
                relative_error = 1.
            else:
                relative_error = max(
                    0., optimal_metric - best_metric
                ) / best_metric

            # Log the error
            self.logger.log_scalar(
                field="error",
                tag=f"{self.logs_tag}_train_episodes",
                value=min(relative_error, 1.),
                step=self.episode
            )
            # Print the error
            print(
                f"(Step {self.tick + 2 + (best_idx - len(recent_memory))}) "
                "Error relative to actual optimum: "
                f"{relative_error * 100.:.2f}%"
            )

    def verbose_episode(self):
        """Prints some info about the current episode"""

        # Assemble print string
        if isinstance(self.env, environments.SimulationEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 20:3d}"
                " ==========\n"
                "\n-----------------------------------"
                "\nRunning episode with "
            )

            if self.metric == "snr":
                print_str += f"T1={self.env.T1:.4f}s & T2={self.env.T2:.4f}s"
            elif self.metric == "cnr":
                print_str += (
                    f"T1a={self.env.T1_1:.4f}s; T2a={self.env.T2_1:.4f}s; "
                    f"T1b={self.env.T1_2:.4f}s; T2b={self.env.T2_2:.4f}s"
                )
            else:
                RuntimeError()

            print_str += (
                f"\nInitial FA:\t\t{self.env.fa:4.1f} [deg]"
                f"\nOptimal FA:\t\t{self.env.optimal_fa:4.1f} [deg]"
                f"\nOptimal {self.metric.upper()}:\t\t"
                f"{getattr(self.env, f'optimal_{self.metric}'):4.2f} [-]"
                "\n-----------------------------------"
            )
        elif isinstance(self.env, environments.ScannerEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 20:3d}"
                " ==========\n"
                "\n-----------------------------------"
                f"\nInitial FA:\t{self.env.fa:4.1f} [deg]"
                "\n-----------------------------------"
            )
        elif isinstance(self.env, environments.KspaceEnv):
            print_str = (
                "\n========== "
                f"Episode {self.episode + 1:3d}/"
                f"{self.n_episodes if self.train else 20:3d}"
                " ==========\n"
                "\n-----------------------------------"
                f"\nInitial FA:\t{self.env.fa_init:4.1f} [deg]"
                "\n-----------------------------------"
            )
        else:
            raise NotImplementedError(
                f"The passed environment class ({self.env.__class__}) "
                "is not supported in algorithms.py"
            )

        # Print the string
        print(print_str)

    def run(self, train=True):
        """Run either training or testing loop"""

        # Set training variable
        self.train = train

        # Print some info
        print(
            "\n\n==================== "
            f"Running {'training' if train else 'test'} loop"
            " ====================\n\n"
        )

        # Create training initial condition distributions
        if train:
            self.env.n_episodes = self.n_episodes
        else:
            self.env.n_episodes = 20

        self.env.homogeneous_initialization = True
        self.env.set_homogeneous_dists()

        # Episode loop
        for self.episode in range(self.n_episodes) if train else range(20):

            # Reset environment and log start
            self.env.reset()
            self.log_initiation()

            # Reset agent
            self.agent.reset()

            # Create episodic memory element (history)
            # for states and next_states
            if self.single_fa:
                states = torch.zeros(
                    (1, self.env.n_states), device=self.device,
                    dtype=torch.float
                )
                next_states = torch.zeros(
                    (1, self.env.n_states), device=self.device,
                    dtype=torch.float
                )
            else:
                states = [
                    torch.zeros(
                        (1, *self.env.img_shape), device=self.device,
                        dtype=torch.float
                    ),
                    torch.zeros(
                        (1, self.env.n_actions), device=self.device,
                        dtype=torch.complex64
                    )
                ]
                next_states = [
                    torch.zeros(
                        (1, *self.env.img_shape), device=self.device,
                        dtype=torch.float
                    ),
                    torch.zeros(
                        (1, self.env.n_actions), device=self.device,
                        dtype=torch.complex64
                    )
                ]

            # Define actions/rewards tensors
            actions = torch.zeros(
                (1, self.env.n_actions), device=self.device)
            rewards = torch.zeros((1), device=self.device)

            # Print some info
            self.verbose_episode()

            # Loop over ticks/steps
            for self.tick in range(self.n_ticks):

                # Extract current state
                state = self.env.state

                # Choose action
                action = self.agent.select_action(state, train)

                # Simulate step
                next_state, reward, done = self.env.step(action)

                # Add states and next_states to history
                if self.single_fa:
                    states = torch.cat(
                        (states, torch.unsqueeze(state, 0))  # type: ignore
                    )
                    next_states = torch.cat(
                        (
                            next_states,
                            torch.unsqueeze(next_state, 0)  # type: ignore
                        )
                    )
                else:
                    states[0] = torch.cat(
                        (states[0], torch.unsqueeze(state[0], 0))
                    )
                    states[1] = torch.cat(
                        (states[1], torch.unsqueeze(state[1], 0))
                    )
                    next_states[0] = torch.cat(
                        (next_states[0], torch.unsqueeze(next_state[0], 0))
                    )
                    next_states[1] = torch.cat(
                        (next_states[1], torch.unsqueeze(next_state[1], 0))
                    )
                # Add action/reward to history
                actions = torch.cat((actions, torch.unsqueeze(action, 0)))
                rewards = torch.cat((rewards, reward))

                # Log step results
                self.log_step(state, action, reward, next_state, done)

                # Check if done and actually stop only if
                # we're far enough into the episode.
                done_threshold = (
                    float(self.tick + 1) / float(self.n_ticks)
                    > 1. - 3 * float(self.episode + 1) / float(self.n_episodes)
                )
                if (
                    done and (
                        done_threshold or not self.train
                    ) or (
                        not self.model_done and done
                    )
                ):
                    print("Stopping criterion met!")
                    break

            # Update memory with previous episode (remove first step)
            if self.single_fa:
                self.memory.push(
                    states[1:], actions[1:], rewards[1:], next_states[1:]
                )
            else:
                self.memory.push(
                    states[0][1:], states[1][1:],
                    actions[1:], rewards[1:],
                    next_states[0][1:], next_states[1][1:]
                )

            # If training, update model
            if train:
                # Generate batch
                batch = self.memory.sample(self.batch_size)
                # Run training
                self.policy_loss, self.critic_loss = \
                    self.agent.update(batch)

            # Log episode results
            if train:
                self.log_episode()

            # Update epsilon
            self.agent.update_epsilon()

            # Backup model once in a while
            if (
                self.episode % (self.n_episodes // 100) == 0
                or self.episode + 1 == self.n_episodes
            ):
                # Update most recent backup
                self.agent.save(self.model_path)
                # Save dynamic variables for this training session
                with open(self.variable_path, "w") as outfile:
                    json.dump({
                        "episode": self.episode,
                        "epsilon": self.agent.epsilon
                    }, outfile)
                # Update "best" backup
                if self.backup_best:
                    self.agent.save(self.model_path.replace(".pt", "_best.pt"))


class Validator(object):
    """Class to represent a validator algorithm"""

    def __init__(
            self,
            env,
            log_dir: Union[str, os.PathLike],
            fa_range: list[int] = [1, 50],
            n_ticks: int = 50,
            device: Union[torch.device, None] = None):
        """Initializes and builds the attributes for this class

        Parameters
        ----------
            env : optimizer.environments.* object
                Environment to be optimized
            log_dir : str | os.PathLike
                Directory in which we store the logs
            fa_range : list[int]
                Range over which to vary the flip angle
            n_ticks : int
                Number of steps we'll vary over
            device : torch.device | None
                The torch device. If None, assign one.
        """

        # Build attributes
        self.env = env
        self.metric = self.env.metric
        self.log_dir = log_dir
        self.fa_range = fa_range
        self.n_ticks = n_ticks

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Setup logger
        self.setup_logger()

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
        self.logs_fields = [
            "img", "fa", "fa_norm", self.metric
        ]
        # Setup logger object
        self.logger = loggers.TensorBoardLogger(
            self.logs_path, self.logs_fields
        )

    def run(self, train=True):
        """Run validation loop"""

        # Create list of flip angles we wish to try
        fa_list = np.linspace(
            self.fa_range[0], self.fa_range[1], self.n_ticks
        )

        # Lock parameters if environment is simulation
        if isinstance(self.env, environments.SimulationEnv):
            self.env.lock_material_params = True
            self.env.reset()
        # Set initial flip angle if in scanner environment
        elif isinstance(self.env, environments.ScannerEnv):
            self.env.reset(fa=self.fa_range[0], run_scan=False)

        # Print start info
        print(
            f"\n======= Running {self.env.metric.upper()} validation ======="
            "\n\n--------------------------------------"
            f"\nFA Range [deg]: [{self.fa_range[0]}, {self.fa_range[1]}]"
            f"\nN_scans:        {self.n_ticks}"
            "\n--------------------------------------\n"
        )

        # Loop over flip angles, create scans and log results
        step = 0
        for fa in fa_list:
            # If applicable, print some info
            print(
                f"Scan #{step+1:2d}: fa = {fa:4.1f} [deg] -> ",
                end="", flush=True
            )

            # Set flip angle
            self.env.fa = float(fa)
            # self.env.norm_parameters()

            # Perform scan
            if isinstance(self.env, environments.SimulationEnv):
                self.env.run_simulation()
            elif isinstance(self.env, environments.ScannerEnv):
                self.env.run_scan_and_update()
            else:
                raise RuntimeError("This shouldn't happen")

            # Log this step (scalars + image)
            self.logger.log_scalar(
                field="fa",
                tag=f"{self.logs_tag}_validation",
                value=float(self.env.fa),
                step=step
            )
            self.logger.log_scalar(
                field="fa_norm",
                tag=f"{self.logs_tag}_validation",
                value=float(self.env.fa_norm),
                step=step
            )
            self.logger.log_scalar(
                field=self.metric,
                tag=f"{self.logs_tag}_validation",
                value=float(getattr(self.env, self.metric)),
                step=step
            )
            if isinstance(self.env, environments.ScannerEnv):
                self.logger.log_image(
                    field="img",
                    tag=f"{self.logs_tag}_validation",
                    image=np.array(self.env.recent_img) / 5.,
                    step=step
                )

            # If applicable, print some info
            print(
                f"\rScan #{step+1:2d}: fa = {fa:4.1f} [deg] -> "
                f"{self.metric} = {getattr(self.env, self.metric):5.2f} [-]"
                "        "
            )

            # Update step counter
            step += 1

        # Print logs location
        print(
            "\nValidation complete! Logs stored at:"
            f"\n{self.logs_path}\n"
        )

"""Module implementing several environments used for RL optimization

We provide definitions for two environments, namely:
- SimulationEnv (An environment for optimizing EPG sequence parameters)
- ScannerEnv (An environment for optimizing actual scanner sequence parameters)

Additionally, these environments may be adjusted to accept
either continuous or discrete action spaces. Also, optimisations
may be performed over both snr and cnr.
"""

import os
from typing import Union
import json
import time
import h5py
import warnings
import numpy as np
import random
import torch
from epg_simulator.python import epg
from util import roi


class ActionSpace(object):
    """Class to represent an action space"""

    def __init__(
            self,
            action_names: list[str],
            action_ranges: Union[np.ndarray, None] = None,
            action_deltas: Union[np.ndarray, None] = None,
            _type: str = "continuous"):
        """Initializes and builds attributes for this class

        Parameters
        ----------
            action_names : list[str]
                Descriptive names for each action
            action_ranges : np.ndarray | None
                Ranges of continuous actions (only if _type=continuous)
            action_deltas : np.ndarray | None
                Deltas of discrete actions (only if _type=discrete)
            _type : str
                Type of action range ("continuous" or "discrete")
        """

        # Build attributes
        self._type = _type
        self._info = []

        # Build actual action space
        if _type == "continuous":
            # Check if action_ranges is defined
            if action_ranges is None:
                raise UserWarning(
                    "Action_ranges must be defined in continuous mode"
                )

            # Define ranges
            self.ranges = action_ranges
            self.low = np.min(action_ranges, axis=1)
            self.high = np.max(action_ranges, axis=1)

            # Define _info
            for action_i in range(len(action_names)):
                self._info.append({
                    "neuron": action_i,
                    "name": action_names[action_i],
                    "min": self.low[action_i],
                    "max": self.high[action_i]
                })

        elif _type == "discrete":
            # Check if action_deltas is defined
            if action_deltas is None:
                raise UserWarning(
                    "Action_deltas must be defined in discrete mode"
                )

            # Define deltas
            self.deltas = action_deltas

            # Define _info
            for action_i in range(len(action_names)):
                self._info.append({
                    "neuron": action_i,
                    "name": action_names[action_i],
                    "delta": action_deltas[action_i]
                })

        else:
            raise ValueError(
                "_type should be either 'continuous' or 'discrete', "
                f"but got '{_type}'"
            )


class SimulationEnv(object):
    """Class to represent a reinforcement learning environment
    for scan parameter optimization in Extended Phase Graph (EPG)
    simulations.
    """

    def __init__(
            self,
            mode: str = "snr",
            action_space_type: Union[str, None] = "continuous",
            recurrent_model: Union[bool, None] = False,
            homogeneous_initialization: bool = False,
            n_episodes: Union[None, int] = None,
            fa_range: list[float] = [20., 60.],
            Nfa: int = 100,
            T1_range: list[float] = [0.100, 2.500],
            T2_range: list[float] = [0.005, 0.100],
            tr: float = 0.050,
            noise_level: float = 0.05,
            lock_material_params: bool = False,
            validation_mode: bool = False,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

            Parameters
            ----------
                mode : str
                    Optimization mode (either snr or cnr)
                action_space_type : str
                    Type of action space. Either discrete or continuous
                recurrent_model : bool
                    Whether we use a recurrent optimizer. This influences
                    the state we'll pass to the model.
                homogeneous_initialization : bool
                    Whether to keep track of homogeneous initialization
                    for training purposes
                n_episodes : int | None
                    The emount of episodes we'll create a distribution over
                    if homogeneous_initialization=True
                fa_range : list[float]
                    Range of optimal and initial flip angles
                Nfa : int
                    Number of pulses in epg simulation
                T1_range : list[float]
                    Range for T1 relaxation in epg simulation [s]
                T2_range : list[float]
                    Range for T2 relaxation in epg simulation [s]
                tr : float
                    Repetition time in epg simulation [ms]
                noise_level : float
                    Noise level for snr calculation
                lock_material_params : bool
                    If True, don't vary material parameters over episodes
                validation_mode : bool
                    If True, use validation mode
                device : None | torch.device
                    Torch device
            """

        # Setup attributes
        self.metric = mode
        self.action_space_type = action_space_type
        self.recurrent_model = recurrent_model
        self.homogeneous_initialization = homogeneous_initialization
        self.n_episodes = n_episodes
        self.fa_range = fa_range
        self.fa = float(np.mean(fa_range))
        self.Nfa = Nfa
        self.T1_range = T1_range
        self.T2_range = T2_range
        self.tr = tr
        self.noise_level = noise_level
        self.lock_material_params = lock_material_params
        self.validation_mode = validation_mode
        self.episode = 0
        self.tick = 0

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize action space
        self.init_actionspace()
        # Set homogeneous initialization distributions
        if self.homogeneous_initialization: self.set_homogeneous_dists()
        # Set environment to starting state
        self.reset()
        # Set n_actions and n_states
        if not validation_mode:
            self.n_actions = len(self.action_space._info)
            self.n_states = len(self.state)

    def init_actionspace(self):
        """Initialize action space

        For continuous mode, use 1 output (fa)
        For discrete mode, use 4 outputs (fa up down, small big)
        """

        if not self.validation_mode:
            if self.action_space_type == "continuous":
                self.action_space = ActionSpace(
                    ["Change FA"],
                    action_ranges=np.array([[-1., 1.]]),
                    _type="continuous"
                )
            elif self.action_space_type == "discrete":
                self.action_space = ActionSpace(
                    [
                        "Decrease FA by 1 [deg]",
                        "Increase FA by 1 [deg]",
                        "Decrease FA by 5 [deg]",
                        "Increase FA by 5 [deg]"
                    ],
                    action_deltas=np.array([-1., 1., 5., -5.]),
                    _type="discrete"
                )
            else:
                raise UserWarning(
                    "action_space_type should be either 'continuous'"
                    " or 'discrete'"
                )

    def set_homogeneous_dists(self):
        """Determine a set of uniformly distributed lists
        for the initializations per episode.
        """

        # Check whether self.n_episodes is defined
        if not self.n_episodes:
            raise UserWarning(
                "If homogeneous_initialization=True, "
                "n_episodes MUST be defined"
            )

        # Create list of initial and optimal flip angles
        # (uniformly distributed in range)
        self.initial_fa_list = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

        self.optimal_fa_list = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

        if self.metric == "snr":
            # Create lists of T2s for single tissue
            # (uniformly distributed in range)
            self.T2_list = list(np.linspace(
                self.T2_range[0], self.T2_range[1],
                self.n_episodes
            ))
        elif self.metric == "cnr":
            # Create lists of T2s for both tissues
            # (uniformly distributed in range)
            self.T2_list_1 = list(np.linspace(
                self.T2_range[0], self.T2_range[1], self.n_episodes
            ))
            self.T2_list_2 = list(np.linspace(
                self.T2_range[0], self.T2_range[1], self.n_episodes
            ))

    def set_t1_from_distribution(self, optimal_fa_list):
        """Find values for T1 for single tissue (snr) or two (cnr)"""

        if self.metric == "snr":
            # Sample an optimal_fa from the list
            fa_idx = random.randint(0, len(optimal_fa_list) - 1)
            optimal_fa = optimal_fa_list.pop(fa_idx)

            # Convert to radians
            fa_rad = optimal_fa * np.pi / 180.

            # Calculate tissue T1
            self.T1 = -self.tr / np.log(np.cos(fa_rad))
        elif self.metric == "cnr":
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
                    self.T1_range[0], self.T1_range[1])

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
        if (self.T1_range[0] < T1_2 < self.T1_range[1]).any():
            # Remove non-valid T1_2 values
            T1_2 = T1_2[self.T1_range[0] < T1_2 < self.T1_range[1]]
            T1_2 = float(T1_2[0])
            # REturn T1_2
            return T1_2
        else:
            return None

    def norm_parameters(self):
        """Update normalized scan parameters"""

        # Perform normalisation for all parameters
        # Only fa for now
        self.fa_norm = float(
            (self.fa - 0.)
            / (180. - 0.)
        )

    def calculate_theoretical_optimum(self):
        """Determine the theoretical optimum for a set of parameters"""

        # If metric=SNR, calculate Ernst angle
        if self.metric == "snr":
            # Calculate Ernst angle
            self.optimal_fa = \
                np.arccos(np.exp(-self.tr / self.T1)) * 180. / np.pi
            # Simulate SNR for Ernst angle
            self.optimal_snr = self.run_simulation(
                fa=self.optimal_fa, pass_to_self=False)

        # If metric=CNR, calculate the optimum algebraically
        elif self.metric == "cnr":
            # Determine E1 for both tissues
            E1a = np.exp(-self.tr / self.T1_1)
            E1b = np.exp(-self.tr / self.T1_2)

            # Calculate best flip angle analytically. Formula retrieved from:
            # Haselhoff EH. Optimization of flip angle for T1 dependent cnr: a
            # closed form solution. Magn Reson Med 1997;38:518 – 9.
            self.optimal_fa = float(np.arccos(
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

            # Simulate CNR for this FA
            self.optimal_cnr = self.run_simulation(
                fa=self.optimal_fa, pass_to_self=False
            )

    def run_simulation(self, fa=None, pass_to_self=True):
        """Run a simulation for scan parameters stored in self"""

        # Select flip angle
        if not fa:
            fa = self.fa
        else:
            fa = float(fa)

        # Determine SNR (if mode="snr")
        if self.metric == "snr":
            # Run simulations
            F0, _, _ = epg.epg_as_numpy(
                self.Nfa, fa, self.tr,
                self.T1, self.T2
            )

            # Determine snr
            snr = float(
                np.abs(F0[-1])
                / self.noise_level
            )
            if pass_to_self: self.snr = snr

            # Return snr
            return snr

        # Determine CNR (if mode="cnr")
        elif self.metric == "cnr":
            # Run simulations
            F0_1, _, _ = epg.epg_as_numpy(
                self.Nfa, fa, self.tr,
                self.T1_1, self.T2_1
            )
            F0_2, _, _ = epg.epg_as_numpy(
                self.Nfa, fa, self.tr,
                self.T1_2, self.T2_2
            )

            # Determine CNR
            cnr = float(
                np.abs(np.abs(F0_1[-1]) - np.abs(F0_2[-1]))
                / self.noise_level
            )
            if pass_to_self: self.cnr = cnr

            # Return cnr
            return cnr

        else:
            raise RuntimeError(
                "mode should be 'snr' or 'cnr'"
            )

    def define_reward(self):
        """Define reward for last step"""

        # Define reward as either +/- 1 for increase or decrease in signal
        if self.state[0] > self.old_state[0]:
            reward_float = 1.0
        else:
            reward_float = -1.0

        # Scale reward with signal difference
        if float(self.old_state[0]) < 1e-2:
            # If old_state signal is too small, set reward gain to 20
            reward_gain = 20.
        else:
            # Calculate relative signal difference and derive reward gain
            snr_diff = (
                abs(self.state[0] - self.old_state[0])
                / self.old_state[0]
            )
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
            reward_float *= np.exp(-self.tick / 20.)

        # Store reward in tensor
        self.reward = torch.tensor(
            [float(reward_float)], device=self.device
        )

    def define_done(self):
        """Define done for last step"""

        # Would like to do this through the agent in the future
        self.done = torch.tensor(0, device=self.device)

    def step(self, action):
        """Run a single step of the RL loop

        - Perform action
        - Run EPG simulation
        - Determine state
        - Determine reward
        - Determine done
        """

        # Update counter
        self.tick += 1

        # Check action validity and perform action
        action_np = action.detach().numpy()
        if self.action_space_type == "continuous":
            if (
                (self.action_space.low <= action_np).all()
                and (action_np <= self.action_space.high).all()
            ):
                # Adjust flip angle
                delta = float(action[0]) * self.fa / 2
                self.fa += delta
            else:
                raise RuntimeError(
                    "Action not in action space. Expected something "
                    f"in range {self.action_space.ranges} but got "
                    f"{action_np}"
                )
        elif self.action_space_type == "discrete":
            if action_np < len(self.action_space.deltas):
                # Adjust flip angle
                delta = float(self.action_space.deltas[int(action)])
                self.fa += delta
            else:
                raise RuntimeError(
                    "Action not in action space. Expected something "
                    f"in range [0, {len(self.action_space.deltas) - 1}] "
                    f"but got {action_np}."
                )

        # Correct for flip angle out of bounds
        if self.fa < 0.0: self.fa = 0.0
        if self.fa > 180.0: self.fa = 180.0
        # Update normalized scan parameters
        self.norm_parameters()

        # Run EPG
        self.run_simulation()

        # Define new state
        self.old_state = self.state
        if not self.recurrent_model:
            self.state = torch.tensor(
                [
                    getattr(self, self.metric), self.fa_norm,
                    float(self.old_state[0]), float(self.old_state[1])
                ],
                device=self.device
            )
        else:
            self.state = torch.tensor(
                [
                    getattr(self, self.metric), self.fa_norm
                ],
                device=self.device
            )

        # Define reward
        self.define_reward()

        # Define done
        self.define_done()

        return self.state, self.reward, self.done

    def reset(self):
        """Reset environment for next episode"""

        # Update counters and reset done
        self.episode += 1
        self.tick = 0
        self.done = False

        # Set new T1, T2, fa, fa_initial
        if self.homogeneous_initialization:
            # Set initial flip angle. Here, we randomly sample from the
            # uniformly distributed list we created earlier.
            self.fa = float(self.initial_fa_list.pop(
                random.randint(0, len(self.initial_fa_list) - 1)
            ))
            # Set the T1s for this episode. Here, we randomly sample
            # T1_1 from the uniform distribution and then calculate T1_2
            # based on the desired optimum flip angle
            self.optimal_fa_list = \
                self.set_t1_from_distribution(self.optimal_fa_list)

            if self.metric == "snr":
                # Set T2 for this episode. We randomly sample this
                # from the previously definded uniform distribution.
                self.T2 = float(self.T2_list.pop(
                    random.randint(0, len(self.T2_list) - 1)
                ))
            elif self.metric == "cnr":
                # Set T2s for this episode. We randomly sample these
                # from the previously definded uniform distributions for both
                # tissues.
                self.T2_1 = float(self.T2_list_1.pop(
                    random.randint(0, len(self.T2_list_1) - 1)
                ))
                self.T2_2 = float(self.T2_list_2.pop(
                    random.randint(0, len(self.T2_list_2) - 1)
                ))

        else:
            # Randomly set fa
            self.fa = random.uniform(
                self.fa_range[0], self.fa_range[1]
            )
            # Randomly set T1/T2 (either for single tissue of two tissues)
            if self.metric == "snr":
                self.T1 = random.uniform(
                    self.T1_range[0], self.T1_range[1])
                self.T2 = random.uniform(
                    self.T2_range[0], self.T2_range[1])
            elif self.metric == "cnr":
                self.T1_1 = random.uniform(
                    self.T1_range[0], self.T1_range[1])
                self.T1_2 = random.uniform(
                    self.T1_range[0], self.T1_range[1])
                self.T2_1 = random.uniform(
                    self.T2_range[0], self.T2_range[1])
                self.T2_2 = random.uniform(
                    self.T2_range[0], self.T2_range[1])

        # If lock_material_params, we simply don't vary T1/T2s
        if self.lock_material_params:
            if self.metric == "snr":
                self.T1 = float(np.mean(self.T1_range))
                self.T2 = float(np.mean(self.T2_range))
            elif self.metric == "cnr":
                self.T1_1 = float(np.percentile(self.T1_range, 25))
                self.T1_2 = float(np.percentile(self.T1_range, 75))
                self.T2_1 = float(np.mean(self.T2_range))
                self.T2_2 = float(np.mean(self.T2_range))

        # Normalize parameters
        self.norm_parameters()

        # Determine theoretical optimum
        self.calculate_theoretical_optimum()

        # Run single simulation step and define initial state
        self.run_simulation()
        self.state = torch.tensor(
            [getattr(self, self.metric), self.fa_norm, 0., 0.]
            if not self.recurrent_model else
            [getattr(self, self.metric), self.fa_norm],
            device=self.device
        )


class ScannerEnv(object):
    """Class to represent a reinforcement learning environment
    for scan parameter optimization in an actual MRI scanner
    """

    def __init__(
            self,
            config_path: Union[str, os.PathLike],
            log_dir: Union[str, os.PathLike],
            metric: str = "snr",
            action_space_type: Union[str, None] = "continuous",
            recurrent_model: Union[bool, None] = False,
            homogeneous_initialization: bool = False,
            n_episodes: Union[int, None] = None,
            fa_range: list[float] = [20., 60.],
            overwrite_roi: bool = False,
            validation_mode: bool = False,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

            Parameters
            ----------
                metric : str
                    Optimization metric (either snr or cnr)
                config_path : str | bytes | os.PathLike
                    Path to config file (mostly for scanner interaction)
                log_dir : str | bytes | os.PathLike
                    Path to log directory
                action_space_type : str
                    Type of action space. Either discrete or continuous
                recurrent_model : bool
                    Whether we use a recurrent optimizer. This influences
                    the state we'll pass to the model.
                homogeneous_initialization : bool
                    Whether to draw the initial flip angles from a
                    homogeneous distribution (for training)
                n_episodes : int | None
                    If homogeneous_initialization=True, n_episodes
                    gives the length of the to-be-initialized homogeneous lists
                fa_range : list[float]
                    Range of initial flip angles
                overwrite_roi : bool
                    Whether to overwrite an existing ROI file
                validation_mode : bool
                    If True, use validation mode
                device : None | torch.device
                    Torch device
            """

        # Setup attributes
        self.config_path = config_path
        self.log_dir = log_dir
        self.metric = metric
        self.action_space_type = action_space_type
        self.recurrent_model = recurrent_model
        self.homogeneous_initialization = homogeneous_initialization
        self.n_episodes = n_episodes
        self.fa_range = fa_range
        self.overwrite_roi = overwrite_roi
        self.validation_mode = validation_mode
        self.episode = 0
        self.tick = 0

        # Setup device
        if not device:
            self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Read config
        self.read_config()
        # Initialize action space
        self.init_actionspace()
        # Setup ROI
        self.setup_roi()
        # Set homogeneous initialization distributions
        if self.homogeneous_initialization: self.set_homogeneous_dists()
        # Set environment to starting state
        self.reset()
        # Set n_actions and n_states
        if not validation_mode:
            self.n_actions = len(self.action_space._info)
            self.n_states = len(self.state)

    def read_config(self):
        """Read info from config file for scanner interaction"""

        # Read config file
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Define communication paths
        self.txt_path = self.config["param_loc"]
        self.lck_path = self.txt_path + ".lck"
        self.data_path = self.config["data_loc"]

    def init_actionspace(self):
        """Initialize action space

        For continuous mode, use 1 output (fa)
        For discrete mode, use 4 outputs (fa up down, small big)
        """

        if not self.validation_mode:
            if self.action_space_type == "continuous":
                self.action_space = ActionSpace(
                    ["Change FA"],
                    action_ranges=np.array([[-1., 1.]]),
                    _type="continuous"
                )
            elif self.action_space_type == "discrete":
                self.action_space = ActionSpace(
                    [
                        "Decrease FA by 1 [deg]",
                        "Increase FA by 1 [deg]",
                        "Decrease FA by 5 [deg]",
                        "Increase FA by 5 [deg]"
                    ],
                    action_deltas=np.array([-1., 1., 5., -5.]),
                    _type="discrete"
                )
            else:
                raise UserWarning(
                    "action_space_type should be either 'continuous'"
                    " or 'discrete'"
                )

    def setup_roi(self):
        """Setup ROI for this environment"""

        # Generate ROI path and calibration image
        self.roi_path = os.path.join(self.log_dir, "roi.npy")
        calibration_image = self.perform_scan(pass_fa=False)

        # Check for existence of ROI file
        if not self.overwrite_roi and os.path.exists(self.roi_path):
            # Load ROI data
            self.roi = np.load(self.roi_path)
        else:
            # Check whether appropriate directory exists
            if not os.path.exists(os.path.dirname(self.roi_path)):
                os.mkdir(os.path.dirname(self.roi_path))
            # Generate new ROI data
            self.roi = roi.generate_rois(calibration_image, self.roi_path)

        # Check whether number of ROIs is appropriate
        if self.metric == "snr":
            if not np.shape(self.roi)[0] == 1:
                raise UserWarning(
                    f"Expected a single ROI to be selected but got "
                    f"{np.shape(self.roi)[0]}.\n"
                    f"ROIs are stored in {self.roi_path}"
                )
        elif self.metric == "cnr":
            if not np.shape(self.roi)[0] == 2:
                raise UserWarning(
                    f"Expected two ROIs to be selected but got "
                    f"{np.shape(self.roi)[0]}.\n"
                    f"ROIs are stored in {self.roi_path}"
                )

    def set_homogeneous_dists(self):
        """Determine a set of uniformly distributed lists
        for the initializations per episode.
        """

        # Check whether self.n_episodes is defined
        if not self.n_episodes:
            raise UserWarning(
                "If homogeneous_initialization=True, "
                "n_episodes MUST be defined"
            )

        # Create list of initial and optimal flip angles
        # (uniformly distributed in range)
        self.initial_fa_list = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

    def norm_parameters(self):
        """Update normalized scan parameters"""

        # Perform normalisation for all parameters
        # Only fa for now
        self.fa_norm = float(
            (self.fa - 0.)
            / (180. - 0.)
        )

    def perform_scan(self, fa=None, pass_fa=True):
        """Perform scan by passing parameters to scanner"""

        # If pass_fa, generate a flip angle file
        if pass_fa:
            # Set flip angle we'll communicate to the scanner
            if not fa:
                fa = self.fa

            # Write new flip angle to appropriate location
            with open(self.lck_path, 'w') as txt_file:
                txt_file.write(f"{fa:.2f}")
            os.system(f"mv {self.lck_path} {self.txt_path}")

        # Wait for image to come back by checking the data file
        while not os.path.exists(self.data_path):
            # Refresh file table
            os.system(f"ls {os.path.dirname(self.data_path)} > /dev/null")
            # Wait for a while
            time.sleep(0.05)

        # When the image is returned, load it and store the results
        with h5py.File(self.data_path, "r") as f:
            self.recent_img = np.asarray(f['/img'])

        # Remove the data file
        os.remove(self.data_path)

        return self.recent_img

    def run_scan_and_update(self):
        """Runs a scan and updates snr/cnr parameter accordingly"""

        # Perform scan and extract image
        img = self.perform_scan()

        # Extract roi
        img_roi = roi.extract_rois(img, self.roi)

        # Determine either snr or cnr (based on mode)
        if self.metric == "snr":
            # Check ROI validity
            if not np.shape(img_roi)[0] == 1:
                raise UserWarning(
                    "In snr mode, only one ROI should be selected."
                )
            # Calculate SNR
            self.snr = float(np.mean(img_roi) / np.std(img_roi))
        elif self.metric == "cnr":
            # Check ROI validity
            if not np.shape(img_roi)[0] == 2:
                raise UserWarning(
                    "In cnr mode, two ROIs should be selected."
                )
            # Calculate CNR
            self.cnr = float(
                np.abs(
                    np.mean(img_roi[0]) - np.mean(img_roi[1])
                ) / np.std(np.concatenate(
                    (img_roi[0].flatten(), img_roi[1].flatten())
                ))
            )

    def define_reward(self):
        """Define reward for last step"""

        # Define reward as either +/- 1 for increase or decrease in signal
        if self.state[0] > self.old_state[0]:
            reward_float = 1.0
        else:
            reward_float = -1.0

        # Scale reward with signal difference
        if float(self.old_state[0]) < 1e-2:
            # If old_state signal is too small, set reward gain to 20
            reward_gain = 20.
        else:
            # Calculate relative signal difference and derive reward gain
            snr_diff = (
                abs(self.state[0] - self.old_state[0])
                / self.old_state[0]
            )
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
            reward_float *= np.exp(-self.tick / 20.)

        # Store reward in tensor
        self.reward = torch.tensor(
            [float(reward_float)], device=self.device
        )

    def define_done(self):
        """Define done for last step"""

        # Would like to do this through the agent in the future
        self.done = torch.tensor(0, device=self.device)

    def step(self, action):
        """Run a single step of the RL loop

        - Perform action
        - Run EPG simulation
        - Determine state
        - Determine reward
        - Determine done
        """

        # Update counter
        self.tick += 1

        # Check action validity and perform action
        action_np = action.detach().numpy()
        if self.action_space_type == "continuous":
            if (
                (self.action_space.low <= action_np).all()
                and (action_np <= self.action_space.high).all()
            ):
                # Adjust flip angle
                delta = float(action[0]) * self.fa / 2
                self.fa += delta
            else:
                raise RuntimeError(
                    "Action not in action space. Expected something "
                    f"in range {self.action_space.ranges} but got "
                    f"{action_np}"
                )
        elif self.action_space_type == "discrete":
            if action_np < len(self.action_space.deltas):
                # Adjust flip angle
                delta = float(self.action_space.deltas[int(action)])
                self.fa += delta
            else:
                raise RuntimeError(
                    "Action not in action space. Expected something "
                    f"in range [0, {len(self.action_space.deltas) - 1}] "
                    f"but got {action_np}."
                )

        # Correct for flip angle out of bounds
        if self.fa < 0.0: self.fa = 0.0
        if self.fa > 180.0: self.fa = 180.0
        # Update normalized scan parameters
        self.norm_parameters()

        # Run EPG
        self.run_scan_and_update()

        # Define new state
        self.old_state = self.state
        if not self.recurrent_model:
            self.state = torch.tensor(
                [
                    getattr(self, self.metric), self.fa_norm,
                    float(self.old_state[0]), float(self.old_state[1])
                ],
                device=self.device
            )
        else:
            self.state = torch.tensor(
                [
                    getattr(self, self.metric), self.fa_norm
                ],
                device=self.device
            )

        # Define reward
        self.define_reward()

        # Define done
        self.define_done()

        return self.state, self.reward, self.done

    def reset(self):
        """Reset environment for next episode"""

        # Update counters and reset done
        self.episode += 1
        self.tick = 0
        self.done = False

        # Set new fa_initial
        if self.homogeneous_initialization:
            # Set initial flip angle. Here, we randomly sample from the
            # uniformly distributed list we created earlier.
            self.fa = float(self.initial_fa_list.pop(
                random.randint(0, len(self.initial_fa_list) - 1)
            ))
        else:
            # Randomly set fa
            self.fa = random.uniform(
                self.fa_range[0], self.fa_range[1]
            )

        # Normalize parameters
        self.norm_parameters()

        # Run single simulation step and define initial state
        self.run_scan_and_update()
        self.state = torch.tensor(
            [getattr(self, self.metric), self.fa_norm, 0., 0.]
            if not self.recurrent_model else
            [getattr(self, self.metric), self.fa_norm],
            device=self.device
        )
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
import glob
import time
import h5py
import warnings
import numpy as np
import random
import torch
from scipy import interpolate
from copy import deepcopy
import epg_simulator.python.epg_numba as epg
import kspace_simulator.simulator as kspace_sim
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
            model_done: bool = True,
            recurrent_model: Union[bool, None] = False,
            homogeneous_initialization: bool = False,
            n_episodes: Union[None, int] = None,
            fa_range: list[float] = [5., 45.],
            Nfa: int = 1000,
            T1_range: list[float] = [0.100, 2.000],
            T2_range: list[float] = [0.025, 0.150],
            tr: float = 0.010,
            noise_level: float = 0.010,
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
                model_done : bool
                    Whether the model may give the "done" command as part
                    of the action space
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
        self.model_done = model_done
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

        If model_done, add the "done" command to the action space
        """

        if not self.validation_mode:
            # Set base action space variables for continuous or discrete mode
            if self.action_space_type == "continuous":
                _type = "continuous"
                action_names = ["Change FA"]
                action_ranges = np.array([[-1., 1.]])
                action_deltas = None
            elif self.action_space_type == "discrete":
                _type = "discrete"
                action_names = [
                    "Decrease FA by 1 [deg]",
                    "Increase FA by 1 [deg]",
                    "Decrease FA by 5 [deg]",
                    "Increase FA by 5 [deg]"
                ]
                action_deltas = np.array([-1., 1., 5., -5.])
                action_ranges = None
            else:
                raise UserWarning(
                    "action_space_type should be either 'continuous'"
                    " or 'discrete'"
                )

            # Append action space with "done" if applicable
            if self.model_done:
                if _type == "continuous":
                    action_names.append("Done")
                    action_ranges = np.concatenate(
                        (action_ranges, np.array([[-1., 1.]])), axis=0
                    )
                else:
                    raise UserWarning(
                        "DQN model isn't equipped to give a done signal"
                    )

            self.action_space = ActionSpace(
                action_names=action_names,
                action_ranges=action_ranges,
                action_deltas=action_deltas,
                _type=_type
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
            T1_1_list = list(np.linspace(
                self.T1_range[0], self.T1_range[1], 10000
            ))
            while not done:
                # Check whether we have surpassed the max count
                if loop >= 9999:
                    # Display warning
                    warnings.warn(
                        "\n\x1b[33;20mT1a/T1b combination for flip angle "
                        f"of {optimal_fa:.2f} [deg] not found!"
                        "\nWe're skipping this flip angle.\x1b[0m"
                    )
                    # Replace non-viable flip angle (if possible)
                    optimal_fa_list.pop(fa_idx)
                    if len(optimal_fa_list) > 0:
                        optimal_fa_list.append(
                            optimal_fa_list[
                                random.randint(0, len(optimal_fa_list) - 1)
                            ]
                        )
                        # Call upon this function to try with another fa
                        self.set_t1_from_distribution(optimal_fa_list)

                    # Break loop
                    break

                # Set T1_1
                T1_1 = T1_1_list.pop(random.randint(0, len(T1_1_list) - 1))

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

        else:
            raise RuntimeError()

        self.optimal_fa = optimal_fa

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
        # fa
        if hasattr(self, "fa"):
            self.fa_norm = float(
                (self.fa - 0.)
                / (90. - 0.)
            )
        # snr / cnr
        if hasattr(self, self.metric):
            setattr(
                self, f"{self.metric}_norm",
                getattr(self, self.metric) / self.metric_calibration
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
            # TODO: This only works in the T1-dependent FA range!!! Gives huge
            # errors when applied to T2-weighted cases. Will have to fix this
            # somehow.
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

    def calculate_steady_state(self, fa, tr, T1, T2):
        """Prototyping function.
        Calculates steady state instead of simulation.
        """

        # Determine flip angle in rads
        alpha = fa * np.pi / 180.

        # calculate signal
        F0 = np.array([
            np.exp((-tr / 2.) / T2)
            * (np.sin(alpha) * (1. - np.exp(-tr / T1)))
            / (1. - np.cos(alpha) * np.exp(-tr / T1))
        ])

        return F0

    def run_simulation(self, fa=None, pass_to_self=True):
        """Run a simulation for scan parameters stored in self"""

        # TODO: Substituted simulations for steady state equation for now

        # Select flip angle
        if not fa:
            fa = self.fa
        else:
            fa = float(fa)

        # Determine SNR (if mode="snr")
        if self.metric == "snr":
            # # Run simulations
            # F0, _, _ = epg.epg_as_numpy(
            #     self.Nfa, fa, self.tr,
            #     self.T1, self.T2
            # )
            F0 = self.calculate_steady_state(fa, self.tr, self.T1, self.T2)

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
            # F0_1, _, _ = epg.epg_as_numpy(
            #     self.Nfa, fa, self.tr,
            #     self.T1_1, self.T2_1
            # )
            # F0_2, _, _ = epg.epg_as_numpy(
            #     self.Nfa, fa, self.tr,
            #     self.T1_2, self.T2_2
            # )
            F0_1 = self.calculate_steady_state(
                fa, self.tr, self.T1_1, self.T2_1
            )
            F0_2 = self.calculate_steady_state(
                fa, self.tr, self.T1_2, self.T2_2
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

        # If the flip angle is changed less than 0.1 deg, penalize the model
        # for waiting too long without stopping
        if abs(self.state[1] - self.old_state[1]) < (0.1 / 180.):
            reward_float -= 0.5

        # If the "done" criterion is passed, tweak the reward based on
        # how close we are to the theretical optimum
        if self.done:
            # Check whether the theoretical optimum is available
            if hasattr(self, f"optimal_{self.metric}"):
                # Extract optimal metric and define error
                optimal_metric = getattr(self, f"optimal_{self.metric}")
                self.error = max(0., float(
                    (optimal_metric - self.state[0] * self.metric_calibration)
                    / optimal_metric)
                )
                # Tweak reward based on error
                if self.error > 0.:
                    if self.n_episodes is not None:
                        reward_delta = min(
                            40.,
                            ((
                                (
                                    4.80 * (
                                        float(self.episode)
                                        / float(self.n_episodes)) ** 2
                                    + 0.20
                                )
                                * min(1., self.error)) ** -1) * 2 - 40.
                        )
                    else:
                        reward_delta = min(
                            40., 2. / (0.2 * min(1., self.error)) - 40.
                        )
                else:
                    reward_delta = 40.

                reward_float += reward_delta

        # Store reward in tensor
        self.reward = torch.tensor(
            [float(reward_float)], device=self.device
        )

    def define_done(self, action):
        """Define done for last step"""

        # If not model_done, use hard-coded criterion. Else, use the action
        if self.model_done:
            self.done = torch.tensor(
                0 if action[1] < 0. else 1,
                device=self.device)
        else:
            # Extract history of this episode
            metric_history = [float(state[0]) for state in self.history]
            # Define patience
            patience = 10 if len(metric_history) > 9 else len(metric_history)

            # Determine whether snr/cnr has improved in our patience period
            done = 0
            max_idx = metric_history.index(max(metric_history))

            if max_idx >= len(metric_history) - patience:
                done = 0
            else:
                done = 1

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

        # Check action validity and perform action TODO:
        action_np = action.detach().numpy()
        self.recent_action = float(action_np[0])

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

        # Run EPG
        self.run_simulation()

        # Update normalized scan parameters
        self.norm_parameters()

        # Define new state
        self.old_state = self.state
        if not self.recurrent_model:
            self.state = torch.tensor(
                [
                    getattr(self, f"{self.metric}_norm"), self.fa_norm,
                    float(self.old_state[0]), float(self.old_state[1])
                ],
                device=self.device
            )
        else:
            self.state = torch.tensor(
                [
                    getattr(self, f"{self.metric}_norm"), self.fa_norm
                ],
                device=self.device
            )
        # Store in history
        self.history.append(self.state)

        # Define done
        self.define_done(action_np)

        # Define reward
        self.define_reward()

        return self.state, self.reward, self.done

    def reset(self):
        """Reset environment for next episode"""

        # Update counters and reset done
        self.episode += 1
        self.tick = 0
        self.done = False
        self.history = []

        # Set new T1, T2, fa, fa_initial
        if self.homogeneous_initialization:

            # Set initial flip angle. Here, we randomly sample from the
            # uniformly distributed list we created earlier.
            self.fa = float(self.initial_fa_list.pop(
                random.randint(0, len(self.initial_fa_list) - 1)
            ))

            # Set material params from distributions
            if not self.lock_material_params:
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
                    # from the previously definded uniform distributions for
                    # both tissues.
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

            # Randomly set material parameters
            if not self.lock_material_params:

                # Randomly set T1/T2 (either for single tissue of two tissues)
                if self.metric == "snr":
                    self.T1 = random.uniform(
                        self.T1_range[0], self.T1_range[1])
                    self.T2 = random.uniform(
                        self.T2_range[0], min(self.T2_range[1], self.T1))
                elif self.metric == "cnr":
                    self.T1_1 = random.uniform(
                        self.T1_range[0], self.T1_range[1])
                    self.T1_2 = random.uniform(
                        self.T1_range[0], self.T1_range[1])
                    self.T2_1 = random.uniform(
                        self.T2_range[0], min(self.T2_range[1], self.T1_1))
                    self.T2_2 = random.uniform(
                        self.T2_range[0], min(self.T2_range[1], self.T1_2))

        # If lock_material_params, we simply don't vary T1/T2s
        if self.lock_material_params:
            if self.metric == "snr":
                # Set to approximate phantom for testing purposes
                self.T1 = 0.600  # float(np.mean(self.T1_range))
                self.T2 = 0.100  # float(np.mean(self.T2_range))
            elif self.metric == "cnr":
                # Set to WM/GM for testing purposes
                self.T1_1 = 0.700   # float(np.percentile(self.T1_range, 25))
                self.T1_2 = 1.400   # float(np.percentile(self.T1_range, 75))
                self.T2_1 = 0.070   # float(np.mean(self.T2_range))
                self.T2_2 = 0.100   # float(np.mean(self.T2_range))

        # Determine theoretical optimum
        self.calculate_theoretical_optimum()

        # Run single simulation step
        self.run_simulation()

        # Perform metric calibration
        self.metric_calibration = getattr(self, f"{self.metric}")

        # Normalize parameters
        self.norm_parameters()
        # Define state
        self.state = torch.tensor(
            [getattr(self, f"{self.metric}_norm"), self.fa_norm, 0., 0.]
            if not self.recurrent_model else
            [getattr(self, f"{self.metric}_norm"), self.fa_norm],
            device=self.device
        )
        # Store in history
        self.history.append(self.state)


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
            model_done: bool = True,
            recurrent_model: Union[bool, None] = False,
            homogeneous_initialization: bool = False,
            n_episodes: Union[int, None] = None,
            fa_range: list[float] = [5., 45.],
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
                model_done : bool
                    Whether the model may give the "done" command as part
                    of the action space
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
        self.model_done = model_done
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
        # Setup calibration
        self.setup_calibration()
        # Set homogeneous initialization distributions
        if self.homogeneous_initialization: self.set_homogeneous_dists()
        # Set environment to starting state
        self.reset(run_scan=False)
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

        If model_done, add the "done" command to the action space
        """

        if not self.validation_mode:
            # Set base action space variables for continuous or discrete mode
            if self.action_space_type == "continuous":
                _type = "continuous"
                action_names = ["Change FA"]
                action_ranges = np.array([[-1., 1.]])
                action_deltas = None
            elif self.action_space_type == "discrete":
                _type = "discrete"
                action_names = [
                    "Decrease FA by 1 [deg]",
                    "Increase FA by 1 [deg]",
                    "Decrease FA by 5 [deg]",
                    "Increase FA by 5 [deg]"
                ]
                action_deltas = np.array([-1., 1., 5., -5.])
                action_ranges = None
            else:
                raise UserWarning(
                    "action_space_type should be either 'continuous'"
                    " or 'discrete'"
                )

            # Append action space with "done" if applicable
            if self.model_done:
                if _type == "continuous":
                    action_names.append("Done")
                    action_ranges = np.concatenate(
                        (action_ranges, np.array([[-1., 1.]])), axis=0
                    )
                else:
                    raise UserWarning(
                        "DQN model isn't equipped to give a done signal"
                    )

            self.action_space = ActionSpace(
                action_names=action_names,
                action_ranges=action_ranges,
                action_deltas=action_deltas,
                _type=_type
            )

    def setup_roi(self, verbose=True):
        """Setup ROI for this environment"""

        # If applicable, print line
        if verbose:
            print(
                "Setting up ROI... ",
                end="", flush=True
            )

        # Generate ROI path and calibration image
        self.roi_path = os.path.join(self.log_dir, "roi.npy")
        # TODO: Changed this during scan session
        self.calibration_image = self.perform_scan(
            fa=50., pass_fa=True
        )

        # Check for existence of ROI file
        if not self.overwrite_roi and os.path.exists(self.roi_path):
            # Load ROI data
            self.roi = np.load(self.roi_path)
            # Print "remove file"
            print(
                "Setting up ROI... "
                "Already there. Removed initial image"
            )
        else:
            # Check whether appropriate directory exists
            if not os.path.exists(os.path.dirname(self.roi_path)):
                os.mkdir(os.path.dirname(self.roi_path))
            # Generate new ROI data
            self.roi = roi.generate_rois(self.calibration_image, self.roi_path)

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

    def setup_calibration(self):
        """Setup calibration data for expected image size
        and intensity.
        """

        # Define expected size
        self.img_size = np.shape(self.calibration_image)

        # Define expected intensities
        self.intensity_95 = np.percentile(self.calibration_image, 95)
        self.intensity_scaling = 1 / float(self.intensity_95)

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
        # fa
        if hasattr(self, "fa"):
            self.fa_norm = float(
                (self.fa - 0.)
                / (90. - 0.)
            )
        # snr / cnr
        if hasattr(self, self.metric):
            setattr(
                self, f"{self.metric}_norm",
                getattr(self, self.metric) / self.metric_calibration
            )

    def perform_scan(self, fa=None, pass_fa=True, verbose=True):
        """Perform scan by passing parameters to scanner"""

        # Print line if verbose
        if verbose:
            print("Waiting for scanner...")

        # If pass_fa, generate a flip angle file
        if pass_fa:
            # Remove image if still there
            if os.path.exists(self.data_path): os.remove(self.data_path)
            if os.path.exists(self.lck_path): os.remove(self.lck_path)
            if os.path.exists(self.txt_path): os.remove(self.txt_path)

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

        # If applicable, scale the image with the appropriate factor
        if hasattr(self, "intensity_scaling"):
            self.recent_img *= self.intensity_scaling

        # Remove the data file
        os.remove(self.data_path)

        # Print line if verbose
        if verbose:
            print("\033[A                             \033[A")

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
            if not len(img_roi) == 1:
                raise UserWarning(
                    "In snr mode, only one ROI should be selected."
                )
            # Calculate SNR
            img_roi = np.array(img_roi)
            self.snr = float(np.mean(img_roi))
        elif self.metric == "cnr":
            # Check ROI validity
            if not len(img_roi) == 2:
                raise UserWarning(
                    "In cnr mode, two ROIs should be selected."
                )
            # Calculate CNR (signal difference / weighted avg. over variance)
            self.cnr = float(
                np.abs(
                    np.mean(img_roi[0]) - np.mean(img_roi[1])
                )
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

    def define_done(self, action):
        """Define done for last step"""

        # If not model_done, use hard-coded criterion. Else, use the action
        if self.model_done:
            self.done = torch.tensor(
                0 if action[1] < 0. else 1,
                device=self.device)
        else:
            # Extract history of this episode
            metric_history = [float(state[0]) for state in self.history]
            # Define patience
            patience = 10 if len(metric_history) > 9 else len(metric_history)

            # Determine whether snr/cnr has improved in our patience period
            done = 0
            max_idx = metric_history.index(max(metric_history))

            if max_idx >= len(metric_history) - patience:
                done = 0
            else:
                done = 1

        # TODO: Testing --> done also given at final tick
        if len(self.history) > 29:
            done = 1
            # else:
            #     done = 0

            # Define done
            self.done = torch.tensor(done, device=self.device)

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

        # Run EPG
        self.run_scan_and_update()

        # Update normalized scan parameters
        self.norm_parameters()

        # Define new state
        self.old_state = self.state
        if not self.recurrent_model:
            self.state = torch.tensor(
                [
                    getattr(self, f"{self.metric}_norm"), self.fa_norm,
                    float(self.old_state[0]), float(self.old_state[1])
                ],
                device=self.device
            )
        else:
            self.state = torch.tensor(
                [
                    getattr(self, f"{self.metric}_norm"), self.fa_norm
                ],
                device=self.device
            )
        # Store in history
        self.history.append(self.state)

        # Define reward
        self.define_reward()

        # Define done
        self.define_done(action_np)

        return self.state, self.reward, self.done

    def reset(self, fa=None, run_scan=True, verbose=True):
        """Reset environment for next episode"""

        # If applicable, print line
        if verbose:
            print(
                "Resetting environment... ",
                end="", flush=True
            )

        # Update counters and reset done
        self.episode += 1
        self.tick = 0
        self.done = False
        self.history = []

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

        # If fa is passed, overwrite it
        if fa: self.fa = fa

        # Normalize parameters
        self.norm_parameters()

        # Run single simulation step and define initial state (if applicable)
        if run_scan:
            # Run scan
            self.run_scan_and_update()
            # Perform metric calibration
            self.metric_calibration = getattr(self, f"{self.metric}")
            # Normalize parameters
            self.norm_parameters()
            # Set state
            self.state = torch.tensor(
                [getattr(self, f"{self.metric}_norm"), self.fa_norm, 0., 0.]
                if not self.recurrent_model else
                [getattr(self, f"{self.metric}_norm"), self.fa_norm],
                device=self.device
            )
        else:
            self.state = torch.tensor(
                [0., 0., 0., 0.]
                if not self.recurrent_model else
                [0., 0.],
                device=self.device
            )

        # Store in history
        self.history.append(self.state)

        # If applicable, print line
        if verbose:
            print(
                "\rResetting environment... Done"
                "                  "
            )


class KspaceEnv(object):
    """Class to represent a reinforcement learning environment
    for full echo train flip angle optimization on an MRI acquisition
    simulator.
    """

    def __init__(
            self,
            config_path: Union[str, os.PathLike],
            log_dir: Union[str, os.PathLike],
            metric: str = "cnr",
            model_done: bool = False,
            homogeneous_initialization: bool = True,
            n_episodes: Union[int, None] = None,
            n_prep_pulses: int = 3,
            parametrization_n_knots: int = 5,
            fa_range: list[float] = [5., 45.],
            validation_mode: bool = False,
            device: Union[torch.device, None] = None):
        """Initializes and builds attributes for this class

            Parameters
            ----------
                metric : str
                    Optimization metric (either snr or cnr)
                config_path : str | bytes | os.PathLike
                    Path to config file
                log_dir : str | bytes | os.PathLike
                    Path to log directory
                model_done : bool
                    Whether the model may give the "done" command as part
                    of the action space
                homogeneous_initialization : bool
                    Whether to draw the initial flip angles from a
                    homogeneous distribution (for training)
                n_episodes : int | None
                    Number of episodes
                n_prep_pulses : int
                    Number of preparation pulses used in simulation
                parametrization_n_knots : int
                    Number of splines used to sample pulse train
                fa_range : list[float]
                    Range of initial flip angles
                validation_mode : bool
                    If True, use validation mode
                device : None | torch.device
                    Torch device
            """

        # Setup attributes
        self.config_path = config_path
        self.log_dir = log_dir
        self.metric = metric
        self.model_done = model_done
        self.homogeneous_initialization = homogeneous_initialization
        self.n_episodes = n_episodes
        self.n_prep_pulses = n_prep_pulses
        self.parametrization_n_knots = parametrization_n_knots
        self.n_state_vector = parametrization_n_knots + 1
        self.fa_range = fa_range
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
        # Set homogeneous fa_initial dist
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
        self.data_dir = self.config["kspace_sim_data_loc"]

        # Get subjects from data directory
        subject_dirs = glob.glob(
            os.path.join(self.data_dir, "[0-9][0-9]_[0-999]")
        )

        # Check all these subjects for the appropriate files
        if len(subject_dirs) > 0:
            # Check whether all appropriate files are present
            incomplete_subjects = []
            for dir in subject_dirs:
                if not (
                    os.path.exists(os.path.join(dir, "T1.npy"))
                    and os.path.exists(os.path.join(dir, "T2.npy"))
                    and os.path.exists(os.path.join(dir, "PD.npy"))
                    and os.path.exists(os.path.join(dir, "mask_1.npy"))
                    and os.path.exists(os.path.join(dir, "mask_2.npy"))
                ):
                    incomplete_subjects.append(dir)
            # Remove incomplete subject folders from list and give warning
            if len(incomplete_subjects) > 0:
                # Remove incomplete subjects from the subject directory list
                subject_dirs = [
                    dir for dir in subject_dirs
                    if dir not in incomplete_subjects
                ]
                # Throw warning
                incomplete_subjects.sort()
                warnings.warn(
                    "\nWarning! Some of the data directories passed are "
                    "incomplete.\nEvery directory is required to contain the "
                    "following files: T1.npy; T2.npy; PD.npy; mask_1.npy; "
                    "mask_2.npy."
                    "\nThe directories that didn't match this criterium are:\n"
                    + "\n".join(incomplete_subjects) + "\n"
                )

                # Check whether any comlete subjects remain
                if len(subject_dirs) == 0:
                    raise UserWarning(
                        "No valid data directories remain. "
                        "Please consult te previous warning for instructions."
                    )

            # Read one file to determine the number of acquisition pulses.
            # We won't check whether these are the same for every image for
            # now, but if this is not the case, we'll get errors later on!
            tryout_map = np.load(os.path.join(subject_dirs[0], "T1.npy"))
            self.img_shape = np.shape(tryout_map)
            self.n_acq_pulses = self.img_shape[0]  # 0-axis is the PA direction

            # Define the number of total pulses
            self.n_pulses = self.n_prep_pulses + self.n_acq_pulses

            # Store subject directories in self
            self.subject_dirs = subject_dirs
            self.remaining_subject_dirs = deepcopy(subject_dirs)

        else:
            raise FileNotFoundError(
                f"The given data directory ({self.data_dir}) "
                "does not contain folders with the appropriate "
                "name convention ('[0-999]_[0-999]')"
            )

    def init_actionspace(self):
        """Initialize action space

        For continuous mode, use 1 output (fa)
        For discrete mode, use 4 outputs (fa up down, small big)

        If model_done, add the "done" command to the action space
        """

        if not self.validation_mode:
            # Set action space type to continuous
            _type = "continuous"

            # Add FA actions for prep pulses
            action_names = [
                f"Change FA (prep pulses)"
            ]
            action_ranges = np.array([[-1., 1.]])
            action_deltas = None
            # Add FA actions for acquisition pulses
            action_names.extend([
                f"Change FA node #{node_i} (acq pulses)"
                for node_i in range(self.parametrization_n_knots)
            ])
            action_ranges = np.concatenate(
                (
                    action_ranges,
                    np.array([[-1., 1.]] * self.parametrization_n_knots)
                ), axis=0
            )

            # Append action space with "done" if applicable
            if self.model_done:

                action_names.append("Done")
                action_ranges = np.concatenate(
                    (action_ranges, np.array([[-1., 1.]])), axis=0
                )

            self.action_space = ActionSpace(
                action_names=action_names,
                action_ranges=action_ranges,
                action_deltas=action_deltas,
                _type=_type
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
        # fa
        if hasattr(self, "fa_init"):
            self.fa_init_norm = float(
                (self.fa_init - 0.)
                / (90. - 0.)
            )
        # theta
        if hasattr(self, "theta"):
            self.theta_norm = (
                (self.theta - 0.)
                / (90. - 0.)
            )
        # Pulsetrain parameters d
        if hasattr(self, "pulsetrain_knots"):
            self.pulsetrain_knots_norm = (
                (self.pulsetrain_knots - 0.)
                / (90. - 0.)
            )
        if hasattr(self, "pulsetrain_param_vector"):
            self.pulsetrain_param_vector_norm = (
                (self.pulsetrain_param_vector - 0.)
                / (90. - 0.)
            )
        # snr / cnr
        if hasattr(self, self.metric):
            setattr(
                self, f"{self.metric}_norm",
                getattr(self, self.metric) / self.metric_calibration
            )

    def setup_rois(self, subject_dir):
        """Setup the ROIs for the current episode"""

        # Load masks
        mask_1 = np.load(os.path.join(subject_dir, "mask_1.npy"))
        mask_2 = np.load(os.path.join(subject_dir, "mask_2.npy"))

        # Store in self
        self.roi = np.array([mask_1, mask_2], dtype=bool)

    def select_subject_dir(self):
        """Select the subject directory for current episode"""

        # If applicable, renew the subject dir list
        if len(self.remaining_subject_dirs) == 0:
            self.remaining_subject_dirs = deepcopy(self.subject_dirs)

        # Randomly select index
        idx = random.randint(0, len(self.remaining_subject_dirs) - 1)
        # Retrieve directory
        self.current_dir = self.remaining_subject_dirs[idx]
        # Remove directory from list
        self.remaining_subject_dirs.pop(idx)

    def perform_simulation(self, theta=None, verbose=True):
        """Perform scan by passing parameters to scanner"""

        # Define flip angle train
        if theta is None: theta = self.theta
        # Cast to tensor
        if type(theta) != torch.Tensor:
            theta = torch.tensor(
                theta, dtype=torch.complex64, device=self.device
            )

        # Perform simulation
        self.recent_img, _ = self.simulator.forward(
            theta, tr=0.050, n_prep=self.n_prep_pulses
        )

        # # Cast to np array
        # self.recent_img = self.recent_img.detach().numpy()

        return self.recent_img

    def run_simulation_and_update(self):
        """Runs a scan and updates snr/cnr parameter accordingly"""

        # Perform scan and extract image
        img = self.perform_simulation()

        # Extract rois
        img_roi = [img[self.roi[0]], img[self.roi[1]]]

        # Determine the CNR metric, which we will use
        # for the 'seperability' of the two ROIs (WM/GM) or in other
        # applications the 'detectability' of lesions in a background tissue.
        # TODO: We might have to use another metric here later on!
        # This one really isn't perfect at all. Maybe Smith et al.
        # lesion detectability.

        if self.metric == "snr":
            raise NotImplementedError("Only CNR is implemented")
        else:
            # Store previous CNR
            if hasattr(self, "cnr"): self.old_cnr = self.cnr

            # Calculate new CNR (signal difference / variances)
            self.cnr = float(
                torch.abs(
                    torch.mean(img_roi[0]) - torch.mean(img_roi[1])
                ) /
                torch.sqrt(
                    torch.var(img_roi[0]) + torch.var(img_roi[1])
                )
            )

    def define_reward(self):
        """Define reward for last step"""

        # Retrieve old and new CNR values
        cnr_old = self.old_cnr
        cnr_new = self.cnr

        # Define reward as either +/- 1 for increase or decrease in signal
        if cnr_new > cnr_old:
            reward_float = 1.0
        else:
            reward_float = -1.0

        # Scale reward with signal difference
        if float(cnr_old) < 1e-2:
            # If old_state signal is too small, set reward gain to 20
            reward_gain = 20.
        else:
            # Calculate relative signal difference and derive reward gain
            cnr_diff = (
                abs(cnr_new - cnr_old)
                / cnr_old
            )
            reward_gain = cnr_diff * 100.

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

    def define_done(self, action):
        """Define done for last step"""

        # If not model_done, use hard-coded criterion. Else, use the action
        # if self.model_done:
        #     self.done = torch.tensor(
        #         0 if action[-1] < 0. else 1,
        #         device=self.device)
        # else:
        #     # Extract history of this episode
        #     metric_history = [float(state[0]) for state in self.history]
        #     # Define patience
        #     patience = 10 if len(metric_history) > 9 else len(metric_history)

        #     # Determine whether snr/cnr has improved in our patience period
        #     done = 0
        #     max_idx = metric_history.index(max(metric_history))

        #     if max_idx >= len(metric_history) - patience:
        #         done = 0
        #     else:
        #         done = 1

        done = 0

        # TODO: Testing --> done also given at final tick
        if len(self.history) > 29:
            done = 1
            # else:
            #     done = 0

            # Define done
            self.done = torch.tensor(done, device=self.device)

    # def update_theta(self, action):
    #     """Update theta (if applicable) using pulse train parametrization"""

    def set_pulsetrain_parametrization(self, parameters: torch.Tensor):
        """Set parametrization of pulse train

        Here, we define this parametrization as a n-order
        polynomial, defined at 0<x<1 and 0<y<1. Here, x=0
        is the first pulse while x=1 is the last pulse. Also,
        y=0 corresponds to a flip angle of 0 deg, while y=1 corresponds
        to a flip angle of 180 deg.

        To retrieve the individual flip angles from this interpolation,
        use interpolate.splev(x, self.spline_representation, der=0)
        """

        if len(parameters) != self.parametrization_n_knots:
            raise ValueError("Argument count doesn't match n_splines")

        # Set n
        n = len(parameters)

        # Set x and y for nodes
        y_nodes = deepcopy(parameters).cpu()
        x_nodes = torch.linspace(0, 1, n).cpu()

        # Create 2nd order spline representation
        self.spline_representation = \
            interpolate.splrep(x_nodes, y_nodes, s=0, k=2)

    def get_pulsetrain_parametrization(self):
        """Get individual flip angles from parametrization"""

        # Get x-values on spline x-axis
        x_theta = torch.linspace(0, 1, self.n_acq_pulses)

        # Get theta and clip to proper range
        theta_acq = np.array(interpolate.splev(
            x_theta, self.spline_representation, der=0
        )).clip(0., 180.)

        # Cast to tensor
        self.theta_acq = torch.tensor(
            theta_acq, dtype=torch.float32, device=self.device
        )

        # Combine preparation and accquisition pulses
        self.theta = torch.concat((self.theta_prep, self.theta_acq), 0)

        # Store the pulsetrain param vector. This is given by:
        # [fa prep, fa knot acq 1, fa knot acq 2, ..., fa knot acq n]
        self.pulsetrain_param_vector = torch.tensor(
            [self.fa_prep, *self.pulsetrain_knots],
            dtype=torch.float32, device=self.device
        )

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
        action_np = action.cpu().detach().numpy()

        if (
            (self.action_space.low <= action_np).all()
            and (action_np <= self.action_space.high).all()
        ):
            # Calculate deltas [-1, 1] -> [3/4, 6/4]
            deltas = (((action_np + 1.) / 2.) * 0.75 + 0.75)
            # Adjust prep pulses
            self.fa_prep *= deltas[0]
            self.theta_prep *= deltas[0]
            # Adjust acq pulses
            self.pulsetrain_knots *= deltas[1:]

            # Adjust theta
            self.set_pulsetrain_parametrization(self.pulsetrain_knots)
            self.get_pulsetrain_parametrization()
        else:
            raise RuntimeError(
                "Action not in action space. Expected something "
                f"in range {self.action_space.ranges} but got "
                f"{action_np}"
            )

        # Run EPG
        self.run_simulation_and_update()

        # Update normalized scan parameters
        self.norm_parameters()

        # Define new state
        self.old_state = self.state
        self.state = [
            self.recent_img,
            self.pulsetrain_param_vector_norm
        ]

        # Store in history
        self.history.append(self.state)

        # Define reward
        self.define_reward()

        # Define done
        self.define_done(action_np)

        return self.state, self.reward, self.done

    def reset(self, fa=None, verbose=True):
        """Reset environment for next episode"""

        # If applicable, print line
        if verbose:
            print(
                "Resetting environment... ",
                end="", flush=True
            )

        # Update counters and reset done
        self.episode += 1
        self.tick = 0
        self.done = False
        self.history = []

        # Set new fa_initial
        if self.homogeneous_initialization:
            # Set initial flip angle. Here, we randomly sample from the
            # uniformly distributed list we created earlier.
            self.fa_init = float(self.initial_fa_list.pop(
                random.randint(0, len(self.initial_fa_list) - 1)
            ))
        else:
            # Randomly set fa
            self.fa_init = random.uniform(
                self.fa_range[0], self.fa_range[1]
            )

        # If fa is passed, overwrite it
        if fa: self.fa_init = fa

        # Set pulse train parameters
        self.pulsetrain_knots = torch.tensor(
            [self.fa_init] * self.parametrization_n_knots,
            dtype=torch.float32, device=self.device
        )

        # Set preparation pulses
        self.fa_prep = self.fa_init
        self.theta_prep = torch.tensor(
            [self.fa_prep] * self.n_prep_pulses,
            dtype=torch.float32, device=self.device
        )

        # Set parametrization of pulse train
        self.set_pulsetrain_parametrization(self.pulsetrain_knots)

        # Setup pulse train
        self.get_pulsetrain_parametrization()

        # Normalize parameters
        self.norm_parameters()

        # Select subject directory to use for this episode
        self.select_subject_dir()

        # Define the simulator class we'll use
        # TODO: Still have to implement subject dir selection
        self.simulator = kspace_sim.SimulatorObject(
            self.current_dir, device=self.device
        )

        # Define the ROIs used for this episode
        self.setup_rois(self.current_dir)

        # Run single simulation step and define initial state (if applicable)
        # Run simulation
        self.run_simulation_and_update()

        # Perform metric calibration
        self.metric_calibration = getattr(self, f"{self.metric}")

        # Normalize parameters
        self.norm_parameters()

        # Set state ([tensor(2D image), tensor(1D FA knots vector)])
        self.state = [
            self.recent_img,
            self.pulsetrain_param_vector_norm
        ]

        # Store in history
        self.history.append(self.state)

        # If applicable, print line
        if verbose:
            print(
                "\rResetting environment... Done"
                "                  "
            )

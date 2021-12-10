"""Module implementing several environments used for RL optimization

We provide definitions for two environments, namely:
- SimulationEnv (An environment for optimizing EPG sequence parameters)
- ScannerEnv (An environment for optimizing actual scanner sequence parameters)

Additionally, these environments may be adjusted to accept
either continuous or discrete action spaces. Also, optimisations
may be performed over both snr and cnr.
"""

from typing import Union
import numpy as np
import random
import torch
from epg_simulator.python import epg


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
            if not action_ranges:
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
            if not action_deltas:
                raise UserWarning(
                    "Action_deltas must be defined in continuous mode"
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
            action_space_type: str = "continuous",
            recurrent_model: bool = False,
            homogeneous_initialization: bool = True,
            n_episodes=None,
            fa_range: list[float] = [20., 60.],
            Nfa: int = 100,
            T1_range: list[float] = [0.100, 2.500],
            T2_range: list[float] = [0.005, 0.100],
            tr: float = 0.050,
            noise_level: float = 0.05,
            lock_material_params: bool = False,
            gamma: float = 0.99,
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
                gamma : float
                    Discount factor for Q value calculation
                device : None | torch.device
                    Torch device
            """

        # Setup attributes
        self.mode = mode
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
        self.gamma = gamma
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

    def init_actionspace(self):
        """Initialize action space

        For continuous mode, use 1 output (fa)
        For discrete mode, use 4 outputs (fa up down, small big)
        """

        if self.action_space_type == "continuous":
            self.action_space = ActionSpace(
                ["Change FA"],
                action_ranges=np.array([[-1., 1.]]),
                _type="continuous"
            )
        elif self.action_space == "discrete":
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
        TODO: for CNR case
        """

        # Create list of initial and optimal flip angles
        # (uniformly distributed in range)
        self.initial_fa = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

        self.optimal_fa = list(np.linspace(
            self.fa_range[0], self.fa_range[1],
            self.n_episodes
        ))

        # Create lists of T2s
        # (uniformly distributed in range)
        self.T2_list = list(np.linspace(
            self.T2_range[0], self.T2_range[1], self.n_episodes
        ))

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

    def norm_parameters(self):
        """Update normalized scan parameters"""

        # Perform normalisation for all parameters
        # Only fa for now
        self.fa_norm = float(
            (self.fa - 0.)
            / (180. - 0.)
        )

    def run_simulation(self, fa=None):
        """Run a simulation for scan parameters stored in self"""

        # Select flip angle
        if not fa:
            fa = self.fa
        else:
            fa = float(fa)

        # Determine SNR (if mode="snr")
        if self.mode == "snr":
            # Run simulations
            F0, _, _ = epg.epg_as_numpy(
                self.Nfa, fa, self.tr,
                self.T1, self.T2
            )

            # Determine snr
            self.snr = (
                np.abs(F0[-1])
                / self.noise_level
            )

        # TODO: Determine CNR (if mode="cnr")
        elif self.mode == "cnr":
            raise NotImplementedError()
        else:
            raise RuntimeError(
                "mode should be 'snr' or 'cnr'"
            )

        # Return self.snr or self.cnr
        return getattr(self, self.mode)

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
                    getattr(self, self.mode), self.fa_norm,
                    float(self.old_state[0]), float(self.old_state[1])
                ],
                device=self.device
            )
        else:
            self.state = torch.tensor(
                [
                    getattr(self, self.mode), self.fa_norm
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
            self.fa = float(self.initial_fa.pop(
                random.randint(0, len(self.initial_fa) - 1)
            ))
            # Set the T1s for this episode. Here, we randomly sample
            # T1_1 from the uniform distribution and then calculate T1_2
            # based on the desired optimum flip angle
            self.optimal_fa = \
                self.set_t1_from_distribution(self.optimal_fa)

            # Set T2 for this episode. We randomly sample this
            # from the previously definded uniform distribution.
            self.T2 = float(self.T2_list.pop(
                random.randint(0, len(self.T2_list) - 1)
            ))
        else:
            self.fa = random.uniform(
                self.fa_range[0], self.fa_range[1]
            )
            self.T1 = random.uniform(
                self.T1_range[0], self.T1_range[1])
            self.T2 = random.uniform(
                self.T2_range[0], self.T2_range[1])

        # Normalize parameters
        self.norm_parameters()

        # Run single simulation step and define initial state
        self.run_simulation()
        self.state = torch.tensor(
            [getattr(self, self.mode), self.fa_norm, 0., 0.]
            if not self.recurrent_model else
            [getattr(self, self.mode), self.fa_norm],
            device=self.device
        )


class ScannerEnv(object):
    pass

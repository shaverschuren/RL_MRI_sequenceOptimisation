"""Module stored for some training-related general utilities

For now, only a short term memory class is implemented.
Might implement some other stuff here at a later stage.
"""

from typing import Union
from collections import namedtuple, OrderedDict, deque
import numpy as np
import random


class OUNoise(object):
    """Class representing Ornstein-Ulhenbeck process

    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    Adapted from:
    https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(
            self,
            action_space: np.ndarray,
            mu: float = 0.0,
            theta: float = 0.15,
            max_sigma: float = 0.3,
            min_sigma: Union[None, float] = None,
            decay_period: int = 10000):
        """Builds OUNoise attributes

        Parameters
        ----------
        action_space : np.ndarray
            Numpy array with action space - shape (n_actions by 2)
        mu : float
        theta : float
        max_sigma : float
        min_sigma : float
        decay_period : int
        """

        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma if min_sigma else max_sigma
        self.decay_period = decay_period
        self.action_dim = np.shape(action_space)[0]
        self.low = np.min(action_space, axis=1)
        self.high = np.max(action_space, axis=1)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = (
            self.theta * (self.mu - x)
            + np.random.normal(
                loc=self.mu, scale=self.sigma, size=self.action_dim
            )
        )
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        action = action.detach().numpy()
        ou_state = self.evolve_state()
        self.sigma = (
            self.max_sigma
            - (self.max_sigma - self.min_sigma)
            * min(1.0, t / float(self.decay_period))
        )
        return np.clip(action + ou_state, self.low, self.high)


class LongTermMemory(object):
    """Class implementing a long term memory object.

    We use this in the training of RL models. This object
    is also sometimes referred to as 'replay memory'.
    """

    def __init__(
            self,
            capacity: int,
            transition_contents: tuple[str, ...] =
            ('state', 'action', 'reward', 'next_state', 'done')):
        """Constructs this memory object

        Parameters
        ----------
            capacity : int
                Maximum capacity of the memory object.
                If full, overflow.
            transition_contents : tuple[str, ...]
                Contents of each stored transition state.
        """

        # Construct attributes
        self.capacity = capacity
        self.transition_contents = transition_contents

        # Construct memory
        self.memory = deque([], maxlen=capacity)
        # Construct transition template
        self.Transition = namedtuple(
            'Transition', transition_contents
        )

    def push(self, *args):
        """Save a transition"""

        # Check if the right amount of args are passed
        if not len(args) == len(self.transition_contents):
            raise ValueError(
                "The number of datapoints passed as a single transition"
                f" is incorrect. Got {len(args)} but expected"
                f" {len(self.transition_contents)}."
            )
        else:
            # If OK, append transition to memory
            self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of samples from the memory"""

        # Check whether batch size isn't larger than memory
        if self.__len__() < batch_size:
            raise IndexError(
                "Batch size is larger then memory size!\n"
                f"Batch size: {batch_size}\nMemory size: {self.__len__()}"
            )

        # Initialize batches
        batches = OrderedDict([
            (batch_name, []) for batch_name in self.transition_contents
        ])

        # Randomly sample batch from memory
        batch = random.sample(self.memory, batch_size)

        # Split batches
        for transition in batch:
            for batch_name, _ in batches.items():
                batches[batch_name].append(getattr(transition, batch_name))

        # Return seperate batches (so e.g. state_batch, action_batch etc.)
        return tuple([batch for _, batch in list(batches.items())])

    def get_recent_memory(self, length: int):
        """Extract recent memory"""

        return list(self.memory)[-length:]

    def __len__(self):
        """Return length"""

        return len(self.memory)

    def __iter__(self):
        """Return iterator"""

        return list(self.memory)


class EpisodicMemory(object):
    """Class implementing an episodic memory object used for RDPG.

    We use this in the training of RL models. This object
    is also sometimes referred to as 'replay memory'.
    """

    def __init__(
            self,
            capacity: int,
            transition_contents: tuple[str, ...] =
            ('state', 'action', 'reward', 'next_state')):
        """Constructs this memory object

        Parameters
        ----------
            capacity : int
                Maximum capacity of the memory object.
                If full, overflow.
            transition_contents : tuple[str, ...]
                Contents of each stored transition state.
        """

        # Construct attributes
        self.capacity = capacity
        self.transition_contents = transition_contents

        # Construct memory
        self.memory = deque([], maxlen=capacity)
        # Construct transition template
        self.Transition = namedtuple(
            'Transition', transition_contents
        )

    def push(self, *args):
        """Save an episode worth of transitions"""

        # Check whether right amount of args is passed
        if not len(args) == len(self.transition_contents):
            raise ValueError(
                f"The transition contents are: {self.transition_contents}"
                f", so {len(self.transition_contents)} arguments were "
                f"expected, but got {len(args)}"
            )

        # Define amount of transitions passed
        n_transitions = args[0].shape[0]

        # Check if the right amount of args are passed
        if not all(n_transitions == arg.shape[0] for arg in args):
            raise ValueError(
                "The number of datapoints passed"
                " is incorrect. All to-be-pushed tensors should "
                f"be the same size at axis 0."
            )

        # If OK, append history and rewards to memory
        trajectory = []
        for i in range(n_transitions):
            # Extract data fields
            transition_contents = [arg[i] for arg in args]
            # Append data to memory
            trajectory.append(
                self.Transition(*transition_contents)
            )
        self.memory.append(trajectory)

    def sample(self, batch_size):
        """Sample a random batch of samples from the memory"""

        # Sample random trajectories from memory
        # If smaller than batch size, just return the full memory (scrambled)
        if self.__len__() < batch_size:
            trajectories = random.sample(self.memory, len(self.memory))
        else:
            trajectories = random.sample(self.memory, batch_size)

        # Extract the minimal length of the sampled trajectories
        min_len = min([len(trajectory) for trajectory in trajectories])
        # Truncate trajectories to minimal size
        trajectories_trunc = []
        for trajectory in trajectories:
            start_idx = random.randint(0, len(trajectory) - min_len)
            trajectories_trunc.append(
                trajectory[start_idx: start_idx + min_len + 1]
            )

        # Zip trajectories so that timesteps are packed together
        # This allows us to perform training of the batch in parallel
        # (much quicker and more stable)
        batch = list(map(list, zip(*trajectories_trunc)))

        # Return batch
        return batch

    def get_recent_memory(self, length: int):
        """Extract recent memory (from last episode)"""

        return list(self.memory[-1])[-length:]

    def __len__(self):
        """Return length"""

        return len(self.memory)

    def __iter__(self):
        """Return iterator"""

        return list(self.memory)

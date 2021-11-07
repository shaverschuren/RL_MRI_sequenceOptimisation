"""Module stored for some training-related general utilities

For now, only a short term memory class is implemented.
Might implement some other stuff here at a later stage.
"""

from collections import namedtuple, deque
import random


class LongTermMemory(object):
    """Class implementing a long term memory object.

    We use this in the training of RL models. This object
    is also sometimes referred to as 'replay memory'.
    """

    def __init__(
            self,
            capacity: int,
            transition_contents: tuple[str, ...] =
            ('state', 'action', 'next_state', 'reward')):
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

        # If OK, append transition to memory
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of samples from the memory"""

        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return length"""

        return len(self.memory)

"""Module used to define a set of models."""

from typing import Union
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable


class FullyConnectedModel(nn.Module):
    """Class to represent a fully connected model"""

    def __init__(
            self,
            architecture: list[int],
            activation_functions: Union[str, list] = 'relu',
            device: Union[None, torch.device] = None):

        """Builds attributes for this fully connected model

        Parameters
        ----------
        architecture : list[int]
            List of number of neurons per layer
        activation_functions : str | list
            Either a single activation function (str) or a list
            of activation functions (list)
        device : None | torch.device
            Optional torch device
        """

        # Build attributes
        self.architecture = architecture
        self.activation_functions = activation_functions
        self.device = device

        # Initialize torch.nn.Module
        super(FullyConnectedModel, self).__init__()

        # Build architecture ordered dictionary
        self.build_architecture_dict()

        # Build fully connected stack
        if self.device:
            self.stack = nn.Sequential(self.architecture_dict).to(device)
        else:
            self.stack = nn.Sequential(self.architecture_dict)

    def build_architecture_dict(self):
        """Builds ordered dict containing model architecture"""

        # Init architecture list
        architecture_list = []
        # Fill list with required layers
        for layer_i in range(len(self.architecture)):
            # Define layer name
            if layer_i == len(self.architecture) - 1:
                layer_name = "output"
            else:
                layer_name = f'fc{layer_i + 1}'
            # Add linear layer
            architecture_list.append(
                (
                    layer_name,
                    nn.Linear(
                        self.architecture[max(0, layer_i - 1)],
                        self.architecture[layer_i])
                )
            )
            # Extract activation function
            if type(self.activation_functions) == str:
                activation_function = self.activation_functions
            elif type(self.activation_functions) == list:
                activation_function = str(self.activation_functions[layer_i])
            else:
                raise TypeError(
                    "Type of 'activation_functions' should be either "
                    f"list or str, but got {type(self.activation_functions)}"
                )
            # Add activation function
            # (ReLU, leaky ReLU, Tanh, sigmoid, softmax supported)
            if str(activation_function).lower() == 'relu':
                architecture_list.append(
                    (f"relu{layer_i + 1}", nn.ReLU())
                )
            elif str(activation_function).lower() == 'leaky_relu':
                architecture_list.append(
                    (f"leaky_relu{layer_i + 1}", nn.LeakyReLU())
                )
            elif str(activation_function).lower() == 'tanh':
                architecture_list.append(
                    (f"tanh{layer_i + 1}", nn.Tanh())
                )
            elif str(activation_function).lower() == 'sigmoid':
                architecture_list.append(
                    (f"sigmoid{layer_i + 1}", nn.Sigmoid())
                )
            elif str(activation_function).lower() == 'softmax':
                architecture_list.append(
                    (f"softmax{layer_i + 1}", nn.Softmax())
                )
            elif str(activation_function).lower() == 'none':
                pass
            else:
                raise ValueError(
                    "Activation function not supported. "
                    "Expected either 'relu', 'leaky_relu', 'tanh', "
                    f"'sigmoid' or 'softmax', but got '{activation_function}'."
                )

        # Fill ordered dict
        self.architecture_dict = OrderedDict(architecture_list)

    def forward(self, x):
        """Forward pass"""

        return self.stack(x)

    def __len__(self):
        """Returns __len__ of model"""

        return len(self.stack)

    def __getitem__(self, item: int):
        """Returns subscriptable item"""

        return self.stack[item]


class RNN(nn.Module):
    def __init__(
            self,
            input_size: int, hidden_size: int, output_size: int,
            output_activation: str,
            device: Union[None, torch.device] = None
    ):
        super(RNN, self).__init__()

        # Build attributes
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.device = device

        # Build layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # Select output activation function
        if output_activation == "relu":
            self.output_activation_func = nn.ReLU()
        elif output_activation == "tanh":
            self.output_activation_func = nn.Tanh()
        elif output_activation == "none":
            self.output_activation_func = None
        else:
            raise ValueError(
                "Only relu, tanh and none are supported."
            )

        # Move to device
        if self.device:
            # i2h, i2o
            self.i2h.to(device)
            self.i2o.to(device)
            # Output activation
            if self.output_activation_func:
                self.output_activation_func.to(device)

    def forward(self, input, hidden):
        """Implement forward pass"""

        # Pass through input2hidden, input2output
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)

        # Pass through activation function (if applicable)
        if self.output_activation_func:
            output = self.output_activation_func(output)

        # Return output, hidden
        return output, hidden

    def forward_sequence(self, input: torch.Tensor):
        """Implements an entire sequential forward pass"""

        # Initialize hidden state
        hidden = self.init_hidden()
        output = torch.Tensor(device=self.device)

        for i in range(input.size()[0]):
            output, hidden = self.forward(input[i], hidden)

        return output

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

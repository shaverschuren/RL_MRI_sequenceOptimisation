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


class RecurrentModel_LSTM(nn.Module):
    def __init__(
            self,
            input_size: int, hidden_size: int, output_size: int,
            fully_connected_architecture: list[int],
            output_activation: str,
            device: Union[None, torch.device] = None
    ):
        super(RecurrentModel_LSTM, self).__init__()

        # Build attributes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fully_connected_architecture = fully_connected_architecture
        self.output_activation = output_activation
        self.device = device

        # Build architecture dict
        self.build_architecture_dict()

        # Build stack
        if self.device:
            self.stack = nn.Sequential(self.architecture_dict).to(device)
        else:
            self.stack = nn.Sequential(self.architecture_dict)

    def build_architecture_dict(self):
        """Build model layer stack

        By default, we take a certain number of linear layers,
        then an LSTM block and finally the output layer.
        """

        # Init architecture list
        architecture_list = []

        # TODO: Testing
        self.lstm_idx = 2

        architecture_list.append(
            (
                "fc1",
                nn.Linear(self.input_size, self.hidden_size)
            )
        )
        architecture_list.append(
            (
                "relu1",
                nn.ReLU()
            )
        )
        architecture_list.append(
            (
                "lstm",
                nn.LSTMCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size
                )
            )
        )
        architecture_list.append(
            (
                "fc2",
                nn.Linear(self.hidden_size, 64)
            )
        )
        architecture_list.append(
            (
                "relu2",
                nn.ReLU()
            )
        )
        # architecture_list.append(
        #     (
        #         "fc3",
        #         nn.Linear(128, 128)
        #     )
        # )
        # architecture_list.append(
        #     (
        #         "relu3",
        #         nn.ReLU()
        #     )
        # )
        architecture_list.append(
            (
                "output",
                nn.Linear(64, self.output_size)
            )
        )
        if self.output_activation.lower() == "tanh":
            architecture_list.append(
                (
                    "tanh",
                    nn.Tanh()
                )
            )
        elif self.output_activation.lower() == "none":
            pass
        else:
            raise RuntimeError()
        # # Append list with layers
        # layer_i = -1
        # for layer_i in range(len(self.fully_connected_architecture) - 1):
        #     # Define layer name
        #     layer_name = f'fc{layer_i + 1}'
        #     # Add linear layer
        #     architecture_list.append(
        #         (
        #             layer_name,
        #             nn.Linear(
        #                 self.fully_connected_architecture[layer_i],
        #                 self.fully_connected_architecture[layer_i + 1])
        #         )
        #     )
        #     # Add activation function
        #     architecture_list.append(
        #         (f"relu{layer_i + 1}", nn.ReLU())
        #     )

        # # Add LSTM module and concurrent relu layer
        # self.lstm_idx = 2 * (layer_i + 1)
        # architecture_list.append(
        #     (
        #         "lstm",
        #         nn.LSTMCell(
        #             input_size=self.fully_connected_architecture[-1],
        #             hidden_size=self.hidden_size
        #         )
        #     )
        # )
        # architecture_list.append(
        #     (
        #         f"relu_lstm", nn.ReLU()
        #     )
        # )

        # # # Define layer name
        # # layer_name = f'fc{layer_i + 2}'
        # # # Add linear layer
        # # architecture_list.append(
        # #     (
        # #         layer_name,
        # #         nn.Linear(
        # #             self.hidden_size,
        # #             self.hidden_size)
        # #     )
        # # )
        # # # Add activation function
        # # architecture_list.append(
        # #     (f"relu{layer_i + 2}", nn.ReLU())
        # # )

        # # Add final output layer and its activation function
        # architecture_list.append(
        #     (
        #         "output",
        #         nn.Linear(
        #             self.hidden_size,
        #             self.output_size)
        #     )
        # )
        # if str(self.output_activation).lower() == 'relu':
        #     architecture_list.append(
        #         (f"relu_output", nn.ReLU())
        #     )
        # elif str(self.output_activation).lower() == 'leaky_relu':
        #     architecture_list.append(
        #         (f"leaky_relu_output", nn.LeakyReLU())
        #     )
        # elif str(self.output_activation).lower() == 'tanh':
        #     architecture_list.append(
        #         (f"tanh_output", nn.Tanh())
        #     )
        # elif str(self.output_activation).lower() == 'sigmoid':
        #     architecture_list.append(
        #         (f"sigmoid_output", nn.Sigmoid())
        #     )
        # elif str(self.output_activation).lower() == 'softmax':
        #     architecture_list.append(
        #         (f"softmax_output", nn.Softmax())
        #     )
        # elif str(self.output_activation).lower() == 'none':
        #     pass
        # else:
        #     raise ValueError(
        #         "Activation function not supported. "
        #         "Expected either 'relu', 'leaky_relu', 'tanh', "
        #         "'sigmoid' or 'softmax', but got "
        #         f"'{self.output_activation}'."
        #     )

        # Fill ordered dict
        self.architecture_dict = OrderedDict(architecture_list)

    def forward(self, x, hidden=None):
        """Implement forward pass"""

        # Pass through the first fully connected layers
        x = self.stack[:self.lstm_idx](x)
        # Pass through the LSTM module and extract x, hidden states
        if hidden is None:
            hx, cx = self.stack[self.lstm_idx](x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            hx, cx = self.stack[self.lstm_idx](x, hidden)
        # Pass through rest of the stack and extract output
        x = hx
        x = self.stack[self.lstm_idx + 1:](x)

        return x, (hx, cx)

    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state of the lstm module"""

        self.hx = Variable(torch.zeros(batch_size, self.hidden_size))
        self.cx = Variable(torch.zeros(batch_size, self.hidden_size))

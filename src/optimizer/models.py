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

        # TODO: Will have to revert this to the format we used before,
        # but for now this works quite well. So... I'm not touching it.
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
                nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=2
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
            x, (hx, cx) = self.stack[self.lstm_idx](
                torch.unsqueeze(x, 0), (self.hx, self.cx)
            )
            self.hx = hx
            self.cx = cx
        else:
            x, (hx, cx) = self.stack[self.lstm_idx](
                torch.unsqueeze(x, 0), hidden
            )
        # Pass through rest of the stack and extract output
        x = torch.squeeze(x, 0)
        x = self.stack[self.lstm_idx + 1:](x)

        return x, (hx, cx)

    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state of the lstm module"""

        self.hx = Variable(torch.zeros(2, batch_size, self.hidden_size))
        self.cx = Variable(torch.zeros(2, batch_size, self.hidden_size))


class RecurrentModel_ConvConcatFC(nn.Module):
    """Model used for processing a 2D image along with an 1D vector.
    
    It is comprised of a convolution part (for the image) and a fully
    connected part, after which the two are combined via concatenation
    and processed in the recurrent part of the model.
    """
    def __init__(
            self,
            input_img_size: tuple[int], input_vector_size: int,
            output_activation: str,
            output_size: int,
            hidden_size: int,
            conv_architecture: list = [0],
            fc_architecture: list = [0],
            rnn_architecture: list = [0],
            device: Union[None, torch.device] = None
    ):
        super(RecurrentModel_ConvConcatFC, self).__init__()

        # Build attributes
        self.input_img_size = input_img_size
        self.input_vector_size = input_vector_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.conv_architecture = conv_architecture
        self.fc_architecture = fc_architecture
        self.rnn_architecture = rnn_architecture
        self.device = device

        # Build architecture dict
        self.build_architecture_dicts()

        # Build stacks
        if self.device:
            self.stack_cnn = nn.Sequential(self.dict_cnn).to(device)
            self.stack_fc = nn.Sequential(self.dict_fc).to(device)
            self.stack_rnn = nn.Sequential(self.dict_rnn).to(device)
        else:
            self.stack_cnn = nn.Sequential(self.dict_cnn)
            self.stack_fc = nn.Sequential(self.dict_fc)
            self.stack_rnn = nn.Sequential(self.dict_rnn)

    def build_architecture_dicts(self):
        """Build model layer stack

        By default, we take a certain number of linear layers,
        then an LSTM block and finally the output layer.
        """

        # Define some parameters here for now. Might move to
        # __init__
        cnn_output_size = 128
        fc_output_size = 128
        rnn_input_size = cnn_output_size + fc_output_size

        # Create CNN architecture list
        cnn_list = [
            ("conv1", nn.Conv2d(1, 4, 5)),
            (("relu1", nn.ReLU())),
            ("pool1", nn.MaxPool2d(2, 2)),
            ("conv2", nn.Conv2d(4, 16, 5)),
            (("relu2", nn.ReLU())),
            ("pool2", nn.MaxPool2d(2, 2)),
            ("flatten", nn.Flatten(0, 1)),
            ("fc1", nn.Linear(16 * 5 * 5, cnn_output_size)),
            (("relu3", nn.ReLU()))
        ]

        # Create FC architecture list
        fc_list = [
            ("fc1", nn.Linear(self.input_vector_size, 128)),
            (("relu1", nn.ReLU())),
            ("fc2", nn.Linear(128, 128)),
            (("relu2", nn.ReLU())),
            ("fc3", nn.Linear(128, fc_output_size)),
            (("relu3", nn.ReLU())),
        ]

        # Create RNN architecture list
        rnn_list = []
        self.lstm_idx = 2
        rnn_list.append(
            (
                "fc1",
                nn.Linear(rnn_input_size, self.hidden_size)
            )
        )
        rnn_list.append(("relu1", nn.ReLU()))
        rnn_list.append(
            (
                "lstm",
                nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=2
                )
            )
        )
        rnn_list.append(
            (
                "fc2",
                nn.Linear(self.hidden_size, self.output_size)
            )
        )
        rnn_list.append(("relu2", nn.ReLU()))
        rnn_list.append(
            (
                "output",
                nn.Linear(self.output_size, self.output_size)
            )
        )
        if self.output_activation.lower() == "tanh":
            rnn_list.append(("tanh", nn.Tanh()))
        elif self.output_activation.lower() == "none":
            pass
        else:
            raise RuntimeError()

        # Fill ordered dicts
        self.dict_cnn = OrderedDict(cnn_list)
        self.dict_fc = OrderedDict(fc_list)
        self.dict_rnn = OrderedDict(rnn_list)

    def forward(self, img, vector, hidden=None):
        """Implement forward pass

        Here, we run the image (img) through the convolutional stack
        and the vector through the fully connected stack.
        Then, we concatenate the resulting 1D feature vectors and run
        this through the RNN (recurrent) stack to finally obtain the 1D
        vector output.

        Parameters
        ----------
            img : torch.Tensor
                2D tensor representing an image
            vector : torch.Tensor
                1D tensor representing a vector (of flip angles)
            hidden : tuple | None
                If given, this provides hx and cx for the LSTM modules.
                If not given, we simply rely on the internally stored
                hidden states.

        Returns
        ------
            output : torch.Tensor
                The 1D vector output of this model
            (hx, cx) : tuple[torch.Tensor]
                The hidden states after the forward pass
        """

        # Pass image through cnn stack
        cnn_out = self.stack_cnn(img)

        # Pass vector through fc stack
        fc_out = self.stack_fc(vector)

        # Concatenate cnn_out and fc_out
        rnn_in = self.stack_rnn[:self.lstm_idx](
            torch.concat([cnn_out, fc_out])
        )

        # Pass through the LSTM module and extract x, hidden states
        if hidden is None:
            rnn_out, (hx, cx) = self.stack_rnn[self.lstm_idx](
                torch.unsqueeze(rnn_in, 0), (self.hx, self.cx)
            )
            self.hx = hx
            self.cx = cx
        else:
            rnn_out, (hx, cx) = self.stack_rnn[self.lstm_idx](
                torch.unsqueeze(rnn_in, 0), hidden
            )
        # Pass through rest of the stack and extract output
        rnn_out = torch.squeeze(rnn_out, 0)
        out = self.stack_rnn[self.lstm_idx + 1:](rnn_out)

        return out, (hx, cx)

    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state of the lstm module"""

        self.hx = Variable(torch.zeros(2, batch_size, self.hidden_size))
        self.cx = Variable(torch.zeros(2, batch_size, self.hidden_size))

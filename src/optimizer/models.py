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


class CNR_Predictor_CNN(nn.Module):
    """Model used for processing a 2D brain image for predicting WM-GM CNR

    It is comprised of a convolution and fully connected part. It
    predicts CNR between white matter and grey matter on a GRE MRI
    sequence with some random pulse train.
    """

    def __init__(
            self,
            input_img_size: tuple[int, int],
            architecture: list = [0],
            device: Union[None, torch.device] = None
    ):
        super(CNR_Predictor_CNN, self).__init__()

        # Build attributes
        self.input_img_size = input_img_size
        self.architecture = architecture
        self.device = device

        # Build architecture dict
        self.build_architecture_dict()

        # Build stacks
        if self.device:
            self.stack = nn.Sequential(
                self.architecture_dict                          # type: ignore
            ).to(device)
        else:
            self.stack = nn.Sequential(self.architecture_dict)  # type: ignore

    def build_architecture_dict(self):
        """Build model layer stack

        By default, we take a certain number of linear layers,
        then an LSTM block and finally the output layer.
        """

        # Create CNN architecture dict
        self.architecture_dict = OrderedDict([
            ("conv1", nn.Conv2d(1, 4, 5, padding=2)),
            (("relu1", nn.ReLU())),
            ("pool1", nn.MaxPool2d(kernel_size=2)),
            ("conv2", nn.Conv2d(4, 16, 5, padding=2)),
            (("relu2", nn.ReLU())),
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("conv3", nn.Conv2d(16, 32, 5, padding=2)),
            (("relu3", nn.ReLU())),
            ("pool3", nn.MaxPool2d(kernel_size=2)),
            ("conv4", nn.Conv2d(32, 32, 5, padding=2)),
            (("relu4", nn.ReLU())),
            ("pool4", nn.MaxPool2d(kernel_size=2)),
            ("flatten", nn.Flatten(1)),
            ("fc1", nn.Linear(
                (self.input_img_size[0] * self.input_img_size[1]) // 8,
                64
            )),
            (("relu5", nn.ReLU())),
            ("fc2", nn.Linear(64, 32)),
            (("relu6", nn.ReLU())),
            ("fc3", nn.Linear(32, 8)),
            (("relu6", nn.ReLU())),
            ("output", nn.Linear(8, 1))
        ])

    def forward(self, img):
        """Implement forward pass

        Here, we run the image (img) through the convolutional stack
        and predict a CNR value

        Parameters
        ----------
            img : torch.Tensor
                2D tensor representing an image

        Returns
        ------
            output : torch.Tensor
                The 0D CNR output of this model
        """

        # Pass image through cnn stack
        return self.stack(img)


class CombinedModel_PulsetrainOptimizer(nn.Module):
    """Model used for processing a 2D image and two 1D vectors with recurrence

    It is comprised of a 2D convolution part (for the image), a 1D convolution
    part for an 1D vector and a fully connected part for the other 1D vector.
    The three parts are then combined via concatenation and processed in the
    recurrent part of the model.

    For a full schematic of the model and its workings, please refer to
    docs/pulseTrain_optimizer_schematic.png
    """

    def __init__(
            self,
            input_img_size: tuple[int, int],
            input_kspace_vector_size: int,
            input_theta_vector_size: int,
            output_activation: str,
            output_size: int,
            hidden_size: int,
            cnr_predictor: CNR_Predictor_CNN,
            conv_architecture: list = [0],
            fc_architecture: list = [0],
            rnn_architecture: list = [0],
            device: Union[None, torch.device] = None
    ):
        super(CombinedModel_PulsetrainOptimizer, self).__init__()

        # Build attributes
        self.input_img_size = input_img_size
        self.input_kspace_vector_size = input_kspace_vector_size
        self.input_theta_vector_size = input_theta_vector_size
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
            # CNR predictor model
            self.cnr_predictor = cnr_predictor.to(device)
            # Kspace encoder, theta encoder and RNN stacks
            self.stack_kspace = nn.Sequential(self.dict_kspace).to(device)
            self.stack_theta = nn.Sequential(self.dict_theta).to(device)
            self.stack_rnn = nn.Sequential(self.dict_rnn).to(device)
        else:
            # CNR predictor model
            self.cnr_predictor = cnr_predictor
            # Kspace encoder, theta encoder and RNN stacks
            self.stack_kspace = nn.Sequential(self.dict_kspace)
            self.stack_theta = nn.Sequential(self.dict_theta)
            self.stack_rnn = nn.Sequential(self.dict_rnn)

    def build_architecture_dicts(self):
        """Build model layer stack

        By default, we take a certain number of linear layers,
        then an LSTM block and finally the output layer.
        """

        # Define some parameters here for now. Might move to
        # __init__
        kspace_output_size = 16
        theta_output_size = 16
        rnn_input_size = kspace_output_size + theta_output_size + 1

        # Create k-space encoder architecture list
        kspace_list = [
            ("conv1", nn.Conv1d(1, 4, 5, padding=2)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool1d(kernel_size=2)),
            ("conv2", nn.Conv1d(4, 8, 5, padding=2)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool1d(kernel_size=2)),
            ("conv3", nn.Conv1d(8, 8, 5, padding=2)),
            ("relu3", nn.ReLU()),
            ("pool3", nn.MaxPool1d(kernel_size=2)),
            ("conv4", nn.Conv1d(8, 8, 5, padding=2)),
            ("relu4", nn.ReLU()),
            ("pool4", nn.MaxPool1d(kernel_size=2)),
            ("flatten", nn.Flatten(1)),
            ("fc1", nn.Linear(self.input_kspace_vector_size // 2, 128)),
            ("relu5", nn.ReLU()),
            ("fc2", nn.Linear(128, 64)),
            ("relu6", nn.ReLU()),
            ("fc3", nn.Linear(64, 32)),
            ("relu7", nn.ReLU()),
            ("fc4", nn.Linear(32, 16)),
            ("relu8", nn.ReLU()),
            ("fc5", nn.Linear(16, kspace_output_size)),
            ("relu9", nn.ReLU()),
        ]

        # Create theta knot encoder architecture list
        # Layer size is dependant on theta input size (i.e. actor or critic)
        # and is rounded up to the nearest power of 2
        theta_layer_size = \
            2**(self.input_theta_vector_size * 2 - 1).bit_length()
        theta_list = [
            ("fc1", nn.Linear(self.input_theta_vector_size, theta_layer_size)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(theta_layer_size, theta_layer_size * 2)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(theta_layer_size * 2, theta_layer_size * 2)),
            ("relu3", nn.ReLU()),
            ("fc4", nn.Linear(theta_layer_size * 2, theta_layer_size)),
            ("relu4", nn.ReLU()),
            ("fc5", nn.Linear(theta_layer_size, theta_output_size)),
            ("relu5", nn.ReLU())
        ]

        # Create RNN architecture list
        rnn_list = []
        self.lstm_idx = 6
        rnn_list = [
            ("fc1", nn.Linear(rnn_input_size, self.hidden_size)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(self.hidden_size, self.hidden_size * 2)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(self.hidden_size * 2, self.hidden_size)),
            ("relu3", nn.ReLU()),
            (
                "lstm", nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=3)
            ),
            ("fc4", nn.Linear(self.hidden_size, self.hidden_size)),
            ("relu4", nn.ReLU()),
            ("fc5", nn.Linear(self.hidden_size, self.hidden_size // 2)),
            ("relu5", nn.ReLU()),
            ("fc6", nn.Linear(self.hidden_size // 2, self.output_size * 2)),
            ("relu6", nn.ReLU()),
            ("output", nn.Linear(self.output_size * 2, self.output_size))
        ]
        if self.output_activation.lower() == "tanh":
            rnn_list.append(("tanh", nn.Tanh()))
        elif self.output_activation.lower() == "none":
            pass
        else:
            raise RuntimeError()

        # Fill ordered dicts
        self.dict_kspace = OrderedDict(kspace_list)
        self.dict_theta = OrderedDict(theta_list)
        self.dict_rnn = OrderedDict(rnn_list)

    def forward(self, img, kspace_vector, theta_vector, hidden=None):
        """Implement forward pass

        Here, we run the image (img) through the 2D convolutional neural net
        and the kspace + theta knot vectors through their own branches as well.
        Then, we concatenate the resulting CNR + 1D feature vectors and run
        this through the RNN (recurrent) stack to finally obtain the 1D
        vector output.

        Parameters
        ----------
            img : torch.Tensor
                2D tensor representing an image
            kspace_vector : torch.Tensor
                1D tensor representing a vector of mean signal per k-space line
            theta_vector : torch.Tensor
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

        # Pass image through cnn stack and store result for later
        # reference and logging
        cnr_out = self.cnr_predictor(img).detach()
        self.recent_cnr_pred = cnr_out

        # Pass kspace vector through appropriate stack
        # For now, we remove the phase and only optimize the amplitude
        kspace_out = self.stack_kspace(torch.abs(kspace_vector))
        # Pass theta vector through appropriate stack
        # For now, we remove the phase and only optimize the amplitude
        theta_out = self.stack_theta(torch.abs(theta_vector))

        # Concatenate cnn_out and fc_out
        rnn_in = self.stack_rnn[:self.lstm_idx](
            torch.concat([cnr_out, kspace_out, theta_out], 1)
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

        self.hx = Variable(
            torch.zeros(3, batch_size, self.hidden_size)
        ).to(self.device)
        self.cx = Variable(
            torch.zeros(3, batch_size, self.hidden_size)
        ).to(self.device)

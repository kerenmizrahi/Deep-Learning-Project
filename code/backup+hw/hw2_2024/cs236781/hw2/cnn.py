import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS
#from .mlp import ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS
#from .layers import MLP
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        # ====== YOUR CODE: ======
        for i, out_channels in enumerate(self.channels):
            layers.append(nn.Conv2d(in_channels, out_channels, **self.conv_params))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))             
            in_channels = out_channels
            if (i + 1) % self.pool_every == 0:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
                           
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            random_image = torch.randn(1, *self.in_size)
            with torch.no_grad():
                extracted_features = self.feature_extractor(random_image)
                n_features = extracted_features.view(1, -1).size(1)
            return n_features
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):

        mlp: MLP = None
        # ====== YOUR CODE: ======
        mlp = MLP(self._n_features(),
                  self.hidden_dims + [self.out_classes],
                  [ACTIVATIONS[self.activation_type](**self.activation_params) for dim in self.hidden_dims]+["none"]
                 )
                  
        # ========================
        return mlp

    def forward(self, x: Tensor):

        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.feature_extractor(x)
        out = self.mlp(out.flatten(start_dim=1))
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_layers = []
        act = ACTIVATIONS[activation_type](**activation_params)
        in_channel = in_channels

        for out_channel, kernel_size in zip(channels[:-1], kernel_sizes[:-1]):
            pad = (kernel_size - 1) // 2
            conv = nn.Conv2d(in_channel, out_channel, kernel_size, bias=True, padding=pad)
            drop = nn.Dropout2d(p=dropout)

            main_layers += [conv, drop]

            if batchnorm is True:
                main_layers.append(nn.BatchNorm2d(out_channel))

            main_layers += [act]

            in_channel = out_channel

        pad = (kernel_sizes[-1] - 1) // 2
        conv = nn.Conv2d(in_channel, channels[-1], kernel_sizes[-1], bias=True, padding=pad)
        main_layers.append(conv)

        self.main_path = nn.Sequential(*main_layers)

        shortcut_layers = []
        if in_channels != channels[-1]:
            shortcut_layers = [nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False)]

        self.shortcut_path = nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        main_out = self.main_path(x)
        shortcut_out = self.shortcut_path(x)
        out = main_out + shortcut_out
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions, 
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        inner_channels = [inner_channels[0]] + inner_channels + [in_out_channels]
        inner_kernel_sizes = [1] + inner_kernel_sizes + [1]

        super().__init__(in_out_channels, inner_channels, inner_kernel_sizes, **kwargs)
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions (make sure to use the right stride and padding).
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        #    Reminder: the number of convolutions performed in the bottleneck block is:
        #    2 + len(inner_channels). [1 for each 1X1 proection convolution] + [# inner convolutions].
        # - Use batchnorm and dropout as requested.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        num_blocks = N // P
        default_k_size = 3

        for i in range(num_blocks):
            inner_channels = self.channels[i * P: (i+1)*P]
            if i == 0:
                input_channel = in_channels
            else:
                input_channel = self.channels[i * P - 1]

            if self.bottleneck and input_channel == inner_channels[-1]:
                input_channel = inner_channels[0]
                block = ResidualBottleneckBlock(
                    in_out_channels = input_channel,
                    inner_channels = inner_channels[1:-1],
                    inner_kernel_sizes = [default_k_size] * len(inner_channels[1:-1]),
                    batchnorm = self.batchnorm,
                    dropout = self.dropout,
                    activation_type = self.activation_type,
                    activation_params = self.activation_params,
                )
            else:
                block = ResidualBlock(
                    in_channels = input_channel,
                    channels = inner_channels,
                    kernel_sizes = [default_k_size] * len(inner_channels),
                    batchnorm = self.batchnorm,
                    dropout = self.dropout,
                    activation_type = self.activation_type,
                    activation_params = self.activation_params,
                )

            layers.append(block)
            layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

        # Create the final residual block if necessary
        if num_blocks * P < N:
            inner_channels = self.channels[num_blocks*P : ]
            input_channel = self.channels[num_blocks*P - 1]
            if self.bottleneck and input_channel == inner_channels[-1]:
                input_channel = inner_channels[0]
                block = ResidualBottleneckBlock(
                    in_out_channels = input_channel,
                    inner_channels = inner_channels[1:-1],
                    inner_kernel_sizes = [default_k_size] * len(inner_channels[1:-1]),
                    batchnorm = self.batchnorm,
                    dropout = self.dropout,
                    activation_type = self.activation_type,
                    activation_params = self.activation_params,
                )
            else:
                block = ResidualBlock(
                    in_channels = input_channel,
                    channels = inner_channels,
                    kernel_sizes = [default_k_size] * len(inner_channels),
                    batchnorm = self.batchnorm,
                    dropout = self.dropout,
                    activation_type = self.activation_type,
                    activation_params = self.activation_params,
                )
            layers.append(block)
        # ========================
        seq = nn.Sequential(*layers)
        return seq


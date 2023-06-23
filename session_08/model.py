from enum import Enum
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


NUM_GROUPS: int = 8
DROPOUT: float = 0.05


class NormalizationMethod(Enum):
    BATCH = 1
    LAYER = 2
    GROUP = 3


def normalizer(
    method: NormalizationMethod,
    out_channels: int,
) -> nn.BatchNorm2d | nn.GroupNorm:
    if method is NormalizationMethod.BATCH:
        return nn.BatchNorm2d(out_channels)
    elif method is NormalizationMethod.LAYER:
        return nn.GroupNorm(1, out_channels)
    elif method is NormalizationMethod.GROUP:
        return nn.GroupNorm(NUM_GROUPS, out_channels)
    else:
        raise ValueError("Invalid NormalizationMethod")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        padding: int = 1,
        norm_method: NormalizationMethod = NormalizationMethod.BATCH,
        add_skip: bool = False,
    ):
        """Initialize Block [with num_layers ConvLayers]

        Args:
            in_channels (int): Input Channel Size
            out_channels (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm_method (enum, optional): Type of normalization to be used. Defaults to NormalizationMethod.BATCH
        """
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    padding=padding,
                    norm_method=norm_method,
                    add_skip=add_skip,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.block(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        norm_method: NormalizationMethod = NormalizationMethod.BATCH,
        add_skip: bool = False,
    ):
        """Initialize Layer [conv(3,3) + normalization + relu]

        Args:
            in_channels (int): Input Channel Size
            out_channels (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm_method (enum, optional): Type of normalization to be used. Defaults to NormalizationMethod.BATCH
        """
        super(ConvLayer, self).__init__()

        self.add_skip = add_skip
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=padding,
            bias=False,
        )
        self.norm = normalizer(method=norm_method, out_channels=out_channels)
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        y = x
        x = self.conv(x)
        x = self.norm(x)
        if self.add_skip:
            x = x + y
        x = self.activation(x)
        x = self.drop(x)
        return x


class TransBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Initialize Transition block [conv (1,1) + max pooling]

        Args:
            in_channels (int): Input Channel Size
            out_channels (int): Output Channel Size
        """
        super(TransBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.layer(x)
        return x


class Session08Model(nn.Module):
    def __init__(
        self,
        norm_method: NormalizationMethod = NormalizationMethod.BATCH,
        add_skip: bool = False,
    ) -> None:
        super(Session08Model, self).__init__()

        self.conv_block1 = ConvBlock(
            in_channels=3,
            out_channels=24,
            num_layers=2,
            padding=1,
            norm_method=norm_method,
            add_skip=False,
        )
        self.trans_block1 = TransBlock(in_channels=24, out_channels=16)

        self.conv_block2 = ConvBlock(
            in_channels=16,
            out_channels=24,
            num_layers=3,
            padding=1,
            norm_method=norm_method,
            add_skip=add_skip,
        )
        self.trans_block2 = TransBlock(in_channels=24, out_channels=16)

        self.conv_block3 = ConvBlock(
            in_channels=16,
            out_channels=24,
            num_layers=3,
            padding=1,
            norm_method=norm_method,
            add_skip=add_skip,
        )

        self.output_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), bias=False),
            nn.Flatten(),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: Tensor):
        x = self.conv_block1(x)
        x = self.trans_block1(x)
        x = self.conv_block2(x)
        x = self.trans_block2(x)
        x = self.conv_block3(x)
        x = self.output_block(x)

        return x


"""
Session 7 Models
"""


class Model_01(nn.Module):
    def __init__(self):
        super(Model_01, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)  # 1x1x10> 10
        return F.log_softmax(x, dim=-1)


class Model_02(nn.Module):
    def __init__(self):
        super(Model_02, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0
            nn.ReLU(),
        )  # output_size = 26

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0
            nn.ReLU(),  # output_size = 24
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1
            nn.ReLU(),  # output_size = 22
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0, output_size = 11
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),  # trans1_conv0, output_size = 11
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0
            nn.ReLU(),  # output_size = 9
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1
            nn.ReLU(),  # output_size = 7
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0
            nn.ReLU(),  # output_size = 7
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),  # conv3_1, output_size = 7x7x10 | 7x7x10x10 | 1x1x10
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_03(nn.Module):
    def __init__(self):
        super(Model_03, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0
            nn.ReLU(),
        )  # output_size = 26

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0
            nn.ReLU(),  # output_size = 24
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1
            nn.ReLU(),  # output_size = 22
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0  # output_size = 11
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0
            nn.ReLU(),  # output_size = 11
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0
            nn.ReLU(),  # output_size = 9
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1
            nn.ReLU(),  # output_size = 7
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0
            nn.ReLU(),  # output_size = 7
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),  # conv3_1  # output_size = 7x7x10 | 7x7x10x10 | 1x1x10
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_04(nn.Module):
    def __init__(self):
        super(Model_04, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0
            nn.ReLU(),
        )  # output_size = 26

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0
            nn.ReLU(),  # output_size = 24
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1
            nn.ReLU(),  # output_size = 22
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 11
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0
            nn.ReLU(),  # output_size = 11
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0
            nn.ReLU(),  # output_size = 9
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1
            nn.ReLU(),  # output_size = 7
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0
            nn.ReLU(),  # output_size = 7
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),  # conv3_1 # output_size = 7x7x10 | 7x7x10x10 | 1x1x10
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_05(nn.Module):
    def __init__(self):
        super(Model_05, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0 # output_size = 26
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0 # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1 # output_size = 22
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 11
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0 # output_size = 11
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0 # output_size = 9
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1 # output_size = 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0 # output_size = 7
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),  # conv3_1 # output_size = 7x7x10 | 7x7x10x10 | 1x1x10
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_06(nn.Module):
    def __init__(self):
        super(Model_06, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )  # output_size = 26

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0  # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1  # output_size = 22
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 11
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0  # output_size = 11
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0 # output_size = 9
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1 # output_size = 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0 # output_size = 7
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),  # conv3_1 # output_size = 7x7x10 | 7x7x10x10 | 1x1x10
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_07(nn.Module):
    def __init__(self):
        super(Model_07, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0 # output_size = 26
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0 # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_1 # output_size = 22
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 11
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0 # output_size = 11
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0 # output_size = 9
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_1 # output_size = 7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv3_0 # output_size = 7
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=7))  # gap0  # output_size = 1

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_08(nn.Module):
    def __init__(self):
        super(Model_08, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0 # output_size = 26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0 # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0 # output_size = 24
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 12
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0 # output_size = 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv3_0 # output_size = 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv4_0 # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # conv5_0 # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # gap0  # output_size = 1

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv6_0
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_09(nn.Module):
    def __init__(self):
        super(Model_09, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0 # output_size = 26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0 # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0 # output_size = 24
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 12
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0 # output_size = 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv3_0 # output_size = 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv4_0 # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # conv5_0 # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # gap0  # output_size = 1

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv6_0
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_10(nn.Module):
    def __init__(self):
        super(Model_10, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv0_0 # output_size = 26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv1_0 # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # trans1_conv0 # output_size = 24
            nn.MaxPool2d(2, 2),  # trans1_pool0 # output_size = 12
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv2_0 # output_size = 10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv3_0 # output_size = 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # conv4_0 # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # conv5_0 # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # gap0  # output_size = 1

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # conv6_0
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Model_11(nn.Module):
    def __init__(self):
        super(Model_11, self).__init__()
        # Input Block
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 26
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 24
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.05),
        )

        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # output_size = 24
            nn.MaxPool2d(2, 2),  # output_size = 12
        )

        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 10
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.05),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 8
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # output_size = 6
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # output_size = 1

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


"""
Session 6 Models
"""


class Session06Model(nn.Module):
    def __init__(self):
        super(Session06Model, self).__init__()
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)

        self.fc = nn.Linear(in_features=1 * 1 * 64, out_features=10)

    def forward(self, x):
        """
        ----------------------------------------------------------------------
        | Layer   | rf_in | n_in | j_in | s | p | k | rf_out | n_out | j_out |
        |---------|-------|------|------|---|---|---|--------|-------|-------|
        | conv1   | 1     | 28   | 1    | 1 | 0 | 5 | 5      | 24    | 1     |
        | relu    | -     | -    | -    | - | - | - | -      | -     | -     |
        | bn      | -     | -    | -    | - | - | - | -      | -     | -     |
        | conv2   | 5     | 24   | 1    | 1 | 0 | 3 | 7      | 22    | 1     |
        | relu    | -     | -    | -    | - | - | - | -      | -     | -     |
        | bn      | -     | -    | -    | - | - | - | -      | -     | -     |
        | maxpool | 7     | 22   | 1    | 2 | 0 | 2 | 8      | 11    | 2     |
        | drop    | -     | -    | -    | - | - | - | -      | -     | -     |
        | conv3   | 8     | 11   | 2    | 1 | 0 | 3 | 12     | 9     | 2     |
        | relu    | -     | -    | -    | - | - | - | -      | -     | -     |
        | bn      | -     | -    | -    | - | - | - | -      | -     | -     |
        | conv4   | 12    | 9    | 2    | 1 | 0 | 3 | 16     | 7     | 2     |
        | relu    | -     | -    | -    | - | - | - | -      | -     | -     |
        | bn      | -     | -    | -    | - | - | - | -      | -     | -     |
        | gap     | -     | -    | -    | - | - | - | -      | -     | -     |
        | drop    | -     | -    | -    | - | - | - | -      | -     | -     |
        | fc      | -     | -    | -    | - | - | - | -      | -     | -     |
        ----------------------------------------------------------------------

        Final RF = 16
        """
        x = self.conv1(x)  # 28x28x1 => 24x24x32
        x = F.relu(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)  # 24x24x32 => 22x22x16
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 22x22x16 => 11x11x16
        x = self.drop1(x)

        x = self.conv3(x)  # 11x11x16 => 9x9x16
        x = F.relu(x)
        x = self.batch_norm3(x)

        x = self.conv4(x)  # 9x9x16 => 7x7x64
        x = F.relu(x)
        x = self.batch_norm4(x)

        x = self.gap(x)
        x = self.drop2(x)

        x = x.reshape(-1, 64 * 1 * 1)

        x = self.fc(x)  # 64*1*1 => 10

        return F.log_softmax(x, dim=1)

import torch.nn as nn
import torch.nn.functional as F


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

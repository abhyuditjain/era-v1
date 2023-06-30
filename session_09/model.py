import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.05


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        """
        CONVOLUTION BLOCK 1
        NIn    RFIn   KernelSize  Padding    Stride  JumpIn  JumpOut   RFOut     NOut       Notes
        32      1       5           2           1       1       1        5        32     Normal Conv
        32      5       3           1           1       1       1        7        32     Normal Conv
        32      7       3           1           1       1       1        9        32     Normal Conv
        """
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 2
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut         Notes
        32       9      3+2          2         2        1        2        13        16     Dilated and Strided Conv
        16      13       3           1         1        2        2        17        16     Depthwise Separable Conv
        16      17       3           1         1        2        2        21        16     Depthwise Separable Conv
        """
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 3
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut         Notes
        16      21      3+2          2         2         2       4        29         8    Dilated and Strided Conv
         8      29      3+2          1         1         4       4        45         8    Dilated Conv
         8      45       3           1         1         4       4        53         8    Normal Conv
        """
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        """
        CONVOLUTION BLOCK 4
        NIn    RFIn   KernelSize  Padding   Stride    JumpIn  JumpOut    RFOut     NOut         Notes
         8      53      3+2          2        2         4        8        69        4      Dilated and Strided Conv
         4      69       3           1        1         8        8        85        4      Depthwise Separable Conv
         4      85       3           1        1         8        8        101       4      Normal Conv (Last conv before GAP)
        """
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            # OUTPUT to GAP so not followed by ReLU + BN + Dropout
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
        )

        """
        GAP
        NIn    NOut
         4      1
        """
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)

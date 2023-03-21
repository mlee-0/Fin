"""Network architectures."""


import torch
from torch.nn import *


residual = lambda input_channels, output_channels: Sequential(
    Conv2d(input_channels, output_channels, kernel_size=3, padding='same'),
    BatchNorm2d(output_channels),
    ReLU(inplace=False),
    Conv2d(output_channels, output_channels, kernel_size=3, padding='same'),
    BatchNorm2d(output_channels),
)


class ResNet(Module):
    def __init__(self) -> None:
        super().__init__()

        input_channels, output_channels = 2, 10
        c = 16

        self.convolution_1 = Sequential(
            Conv2d(input_channels, c*1, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_2 = Sequential(
            Conv2d(c*1, c*2, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_3 = Sequential(
            Conv2d(c*2, c*4, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*4),
            ReLU(inplace=True),
        )
        self.pooling = AdaptiveMaxPool2d(output_size=(1, 1))
        self.linear = Linear(c*4, 2*8)

        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*2, c*2)
        self.residual_3 = residual(c*4, c*4)

        self.deconvolution_1 = Sequential(
            ConvTranspose2d(1, c*2, 3, 2, 1, output_padding=(1, 1)),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.deconvolution_2 = Sequential(
            ConvTranspose2d(c*2, output_channels, 3, 2, 1, output_padding=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_2(x)
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_3(x)
        x = torch.relu(x + self.residual_3(x))
        x = self.pooling(x)
        x = self.linear(x.reshape(x.size(0), -1))
        x = x.reshape((x.size(0), 1, 2, 2*4))

        x = self.deconvolution_1(x)
        x = self.deconvolution_2(x)

        return x

class Autoencoder16(Module):
    def __init__(self) -> None:
        super().__init__()

        input_channels = 2
        c = 16

        self.convolution_1 = Sequential(
            Conv2d(input_channels, c*1, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_2 = Sequential(
            Conv2d(c*1, c*2, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_3 = Sequential(
            Conv2d(c*2, c*4, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*4),
            ReLU(inplace=True),
        )
        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*2, c*2)
        self.residual_3 = residual(c*4, c*4)
        self.pooling_1 = MaxPool2d(kernel_size=(5, 10), stride=1, padding=0, return_indices=True)
        self.linear_1 = Linear(c*4, 2*8)

        self.pooling_2 = MaxUnpool2d(kernel_size=(5, 10), stride=1, padding=0)
        self.linear_2 = Linear(2*8, c*4)
        self.convolution_4 = Sequential(
            ConvTranspose2d(c*4, c*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_5 = Sequential(
            ConvTranspose2d(c*2, c*1, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_6 = Sequential(
            ConvTranspose2d(c*1, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLU(inplace=True),
        )
        self.residual_4 = residual(c*4, c*4)
        self.residual_5 = residual(c*2, c*2)
        self.residual_6 = residual(c*1, c*1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_2(x)
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_3(x)
        x = torch.relu(x + self.residual_3(x))
        x, pooling_indices = self.pooling_1(x)
        x = self.linear_1(x.reshape(x.size(0), -1))

        x = self.linear_2(x).reshape(x.size(0), 16*4, 1, 1)
        x = self.pooling_2(x, pooling_indices)
        x = torch.relu(x + self.residual_4(x))
        x = self.convolution_4(x)
        x = torch.relu(x + self.residual_5(x))
        x = self.convolution_5(x)
        x = torch.relu(x + self.residual_6(x))
        x = self.convolution_6(x)

        return x

class Autoencoder32(Module):
    def __init__(self) -> None:
        super().__init__()

        input_channels = 2
        c = 32
        self.c = c

        self.convolution_1 = Sequential(
            Conv2d(input_channels, c*1, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_2 = Sequential(
            Conv2d(c*1, c*2, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_3 = Sequential(
            Conv2d(c*2, c*4, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*4),
            ReLU(inplace=True),
        )
        self.pooling_1 = MaxPool2d(kernel_size=(5, 10), stride=1, padding=0, return_indices=True)
        self.linear_1 = Linear(c*4, 4*16)

        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*2, c*2)
        self.residual_3 = residual(c*4, c*4)

        self.pooling_2 = MaxUnpool2d(kernel_size=(5, 10), stride=1, padding=0)
        self.linear_2 = Linear(4*16, c*4)
        self.convolution_4 = Sequential(
            ConvTranspose2d(c*4, c*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_5 = Sequential(
            ConvTranspose2d(c*2, c*1, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_6 = Sequential(
            ConvTranspose2d(c*1, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLU(inplace=True),
        )
        self.residual_4 = residual(c*4, c*4)
        self.residual_5 = residual(c*2, c*2)
        self.residual_6 = residual(c*1, c*1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_2(x)
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_3(x)
        x = torch.relu(x + self.residual_3(x))
        x, pooling_indices = self.pooling_1(x)
        x = self.linear_1(x.reshape(x.size(0), -1))

        x = self.linear_2(x).reshape(x.size(0), self.c*4, 1, 1)
        x = self.pooling_2(x, pooling_indices)
        x = torch.relu(x + self.residual_4(x))
        x = self.convolution_4(x)
        x = torch.relu(x + self.residual_5(x))
        x = self.convolution_5(x)
        x = torch.relu(x + self.residual_6(x))
        x = self.convolution_6(x)

        return x
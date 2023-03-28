"""Network architectures."""


import torch
from torch.nn import *


def print_model_summary(model: Module) -> None:
    """Print information about a model."""
    print(f"\n{type(model).__name__}")
    print(f"\tTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\tLearnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

residual = lambda input_channels, output_channels: Sequential(
    Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(output_channels),
    ReLU(inplace=False),
    Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(output_channels),
)

class ThermalNet(Module):
    def __init__(self, c: int, output_channels: int) -> None:
        super().__init__()

        input_channels = 3

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

        self.residual_11 = residual(c*1, c*1)
        self.residual_12 = residual(c*1, c*1)
        self.residual_13 = residual(c*1, c*1)
        self.residual_14 = residual(c*1, c*1)
        self.residual_21 = residual(c*2, c*2)
        self.residual_22 = residual(c*2, c*2)
        self.residual_23 = residual(c*2, c*2)
        self.residual_24 = residual(c*2, c*2)
        self.residual_31 = residual(c*4, c*4)
        self.residual_32 = residual(c*4, c*4)
        self.residual_33 = residual(c*4, c*4)
        self.residual_34 = residual(c*4, c*4)

        self.pooling = AdaptiveMaxPool2d(output_size=(1, 1))
        self.linear = Linear(c*4, 2*8)

        self.deconvolution_1 = Sequential(
            ConvTranspose2d(1, c*2, kernel_size=2, stride=2, padding=0, output_padding=0),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.deconvolution_2 = Sequential(
            ConvTranspose2d(c*2, output_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
            ReLU(inplace=True),
        )
        self.residual_41 = residual(c*2, c*2)
        self.residual_42 = residual(c*2, c*2)
        self.residual_43 = residual(c*2, c*2)
        self.residual_44 = residual(c*2, c*2)
        self.residual_51 = residual(output_channels, output_channels)
        self.residual_52 = residual(output_channels, output_channels)
        self.residual_53 = residual(output_channels, output_channels)
        self.residual_54 = residual(output_channels, output_channels)

        print_model_summary(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_11(x))
        x = torch.relu(x + self.residual_12(x))
        x = torch.relu(x + self.residual_13(x))
        x = torch.relu(x + self.residual_14(x))
        x = self.convolution_2(x)
        x = torch.relu(x + self.residual_21(x))
        x = torch.relu(x + self.residual_22(x))
        x = torch.relu(x + self.residual_23(x))
        x = torch.relu(x + self.residual_24(x))
        x = self.convolution_3(x)
        x = torch.relu(x + self.residual_31(x))
        x = torch.relu(x + self.residual_32(x))
        x = torch.relu(x + self.residual_33(x))
        x = torch.relu(x + self.residual_34(x))
        x = self.pooling(x)
        x = self.linear(x.reshape(x.size(0), -1))
        x = x.reshape((x.size(0), 1, 2, 8))

        x = self.deconvolution_1(x)
        x = torch.relu(x + self.residual_41(x))
        x = torch.relu(x + self.residual_42(x))
        x = torch.relu(x + self.residual_43(x))
        x = torch.relu(x + self.residual_44(x))
        x = self.deconvolution_2(x)
        x = torch.relu(x + self.residual_51(x))
        x = torch.relu(x + self.residual_52(x))
        x = torch.relu(x + self.residual_53(x))
        x = torch.relu(x + self.residual_54(x))

        return x

    def freeze_encoder(self) -> None:
        self.convolution_1.requires_grad_(False)
        self.convolution_2.requires_grad_(False)
        self.convolution_3.requires_grad_(False)
        self.residual_1.requires_grad_(False)
        self.residual_2.requires_grad_(False)
        self.residual_3.requires_grad_(False)

class Autoencoder3(Module):
    def __init__(self, c: int) -> None:
        super().__init__()

        input_channels = 2
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
        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*2, c*2)
        self.residual_3 = residual(c*4, c*4)
        self.pooling_1 = MaxPool2d(kernel_size=(5, 10), stride=1, padding=0, return_indices=True)

        self.pooling_2 = MaxUnpool2d(kernel_size=(5, 10), stride=1, padding=0)
        self.convolution_4 = Sequential(
            ConvTranspose2d(c*4, c*2, kernel_size=2, stride=2, padding=0, output_padding=0),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_5 = Sequential(
            ConvTranspose2d(c*2, c*1, kernel_size=2, stride=2, padding=0, output_padding=0),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_6 = Sequential(
            ConvTranspose2d(c*1, input_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
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

        x = self.pooling_2(x, pooling_indices)
        x = torch.relu(x + self.residual_4(x))
        x = self.convolution_4(x)
        x = torch.relu(x + self.residual_5(x))
        x = self.convolution_5(x)
        x = torch.relu(x + self.residual_6(x))
        x = self.convolution_6(x)

        return x

class Autoencoder4(Module):
    "4 convolutional layers."

    def __init__(self, c: int) -> None:
        super().__init__()

        input_channels = 2
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
        self.convolution_4 = Sequential(
            Conv2d(c*4, c*8, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(c*8),
            ReLU(inplace=True),
        )
        self.residual_1 = residual(c*1, c*1)
        self.residual_2 = residual(c*2, c*2)
        self.residual_3 = residual(c*4, c*4)
        self.residual_4 = residual(c*8, c*8)
        self.pooling_1 = MaxPool2d(kernel_size=(3, 5), stride=1, padding=0, return_indices=True)
        # self.linear_1 = Linear(c*4, 2*8)

        # self.linear_2 = Linear(2*8, c*4)
        self.pooling_2 = MaxUnpool2d(kernel_size=(3, 5), stride=1, padding=0)
        self.convolution_5 = Sequential(
            ConvTranspose2d(c*8, c*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*4),
            ReLU(inplace=True),
        )
        self.convolution_6 = Sequential(
            ConvTranspose2d(c*4, c*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*2),
            ReLU(inplace=True),
        )
        self.convolution_7 = Sequential(
            ConvTranspose2d(c*2, c*1, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(c*1),
            ReLU(inplace=True),
        )
        self.convolution_8 = Sequential(
            ConvTranspose2d(c*1, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLU(inplace=True),
        )
        self.residual_5 = residual(c*8, c*8)
        self.residual_6 = residual(c*4, c*4)
        self.residual_7 = residual(c*2, c*2)
        self.residual_8 = residual(c*1, c*1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution_1(x)
        x = torch.relu(x + self.residual_1(x))
        x = self.convolution_2(x)
        x = torch.relu(x + self.residual_2(x))
        x = self.convolution_3(x)
        x = torch.relu(x + self.residual_3(x))
        x = self.convolution_4(x)
        x = torch.relu(x + self.residual_4(x))
        x, pooling_indices = self.pooling_1(x)

        x = self.pooling_2(x, pooling_indices)
        x = torch.relu(x + self.residual_5(x))
        x = self.convolution_5(x)
        x = torch.relu(x + self.residual_6(x))
        x = self.convolution_6(x)
        x = torch.relu(x + self.residual_7(x))
        x = self.convolution_7(x)
        x = torch.relu(x + self.residual_8(x))
        x = self.convolution_8(x)

        return x
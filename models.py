"""Network architectures."""


from typing import *

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

    def load_encoder(self, weights: Dict[str, torch.Tensor]):
        """Load weights for the encoder only."""

        # Define the prefixes for all layers in the decoder.
        decoder_layer_prefixes = ['deconvolution', 'residual_4', 'residual_5']
        # Remove the decoder layers from the given dictionary of parameters.
        weights_encoder = {
            key: value for key, value in weights.items()
            if not any(key.startswith(_) for _ in decoder_layer_prefixes)
        }
        self.load_state_dict(weights_encoder, strict=False)

        for name, parameter in self.named_parameters():
            if not any(name.startswith(_) for _ in decoder_layer_prefixes):
                parameter.requires_grad = False
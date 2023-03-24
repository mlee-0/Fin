"""Load data."""


import numpy as np
from torch.utils.data import Dataset

from preprocessing import *


def generate_simulation_parameters() -> List[Tuple[float, float]]:
    """Return a list of tuples of simulation parameters for each simulation."""

    return [
        (thickness, taper_ratio, convection_coefficient, temperature)
        for thickness in np.arange(5, 10+1, 1).round(0)
        for taper_ratio in np.arange(0.1, 1+0.1, 0.1).round(1)
        for convection_coefficient in np.arange(10, 100+1, 10).round(0)
        for temperature in np.arange((30+273.15), (100+273.15)+1, 10).round(2)
    ]

def print_dataset_summary(inputs: torch.Tensor, outputs: torch.Tensor) -> None:
    """Print information about the given input and output data."""

    print(f"Input data:")
    print(f"\tShape: {inputs.size()}")
    print(f"\tMemory: {inputs.storage().nbytes()/1e6:,.2f} MB")
    print(f"\tMin, max: {inputs.min()}, {inputs.max()}")

    print(f"Label data:")
    print(f"\tShape: {outputs.size()}")
    print(f"\tMemory: {outputs.storage().nbytes()/1e6:,.2f} MB")
    print(f"\tMin, max: {outputs.min()}, {outputs.max()}")
    print(f"\tMean: {outputs.mean()}")


class FinDataset(Dataset):
    """Dataset of a specific response obtained in FEA."""

    def __init__(self, response: str) -> None:
        super().__init__()

        self.parameters = generate_simulation_parameters()
        self.inputs = make_inputs(self.parameters).float()

        if response == 'temperature':
            self.outputs = load_pickle('Thermal 2023-03-23/outputs.pickle')[..., 0].float()
        elif response == 'thermal gradient':
            self.outputs = load_pickle('Thermal 2023-03-23/outputs.pickle')[..., 1].float()
        elif response == 'stress':
            self.outputs = load_pickle('Structural 2023-03-23/outputs.pickle')[..., 0].float()
        else:
            raise Exception(f"Invalid response: '{response}'.")

        print_dataset_summary(self.inputs, self.outputs)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        return self.inputs[index], self.outputs[index], self.parameters[index]

class AutoencoderDataset(Dataset):
    """Dataset of input images for training an autoencoder."""

    def __init__(self) -> None:
        super().__init__()

        self.parameters = generate_simulation_parameters()
        self.inputs = make_inputs(self.parameters).float()

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.inputs[index]


if __name__ == '__main__':
    dataset = FinDataset('temperature')
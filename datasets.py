"""Load data."""


import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from preprocessing import *


def generate_simulation_parameters() -> List[Tuple[float, float, float, float]]:
    """Return a list of tuples of simulation parameters for each simulation."""

    return [
        (thickness, taper_ratio, convection_coefficient, temperature)
        for thickness in np.arange(5, 10+1, 1).round(0)
        for taper_ratio in np.arange(0.1, 1+0.1, 0.1).round(1)
        for convection_coefficient in np.arange(10, 100+1, 10).round(0)
        for temperature in np.arange((30+273.15), (100+273.15)+1, 10).round(2)
    ]

def histogram_simulation_parameters(parameters: List[Tuple[float, float, float, float]]) -> None:
    """Show a histogram of the given list of parameters."""

    parameters = np.array(parameters)
    plt.subplot(4, 1, 1)
    plt.hist(parameters[:, 0], bins=6)
    plt.subplot(4, 1, 2)
    plt.hist(parameters[:, 1], bins=10)
    plt.subplot(4, 1, 3)
    plt.hist(parameters[:, 2], bins=10)
    plt.subplot(4, 1, 4)
    plt.hist(parameters[:, 3], bins=8)
    plt.show()

def print_dataset_summary(inputs: torch.Tensor, outputs: torch.Tensor) -> None:
    """Print information about the given input and output data."""

    print(f"\nInput data:")
    print(f"\tShape: {inputs.size()}")
    print(f"\tMemory: {inputs.storage().nbytes()/1e6:,.2f} MB")
    print(f"\tMin, max: {inputs.min()}, {inputs.max()}")
    print(f"\tMean, standard deviation: {inputs.mean()}, {inputs.std()}")

    print(f"Label data:")
    print(f"\tShape: {outputs.size()}")
    print(f"\tMemory: {outputs.storage().nbytes()/1e6:,.2f} MB")
    print(f"\tMin, max: {outputs.min()}, {outputs.max()}")
    print(f"\tMean, standard deviation: {outputs.mean()}, {outputs.std()}")


class FinDataset(Dataset):
    """Dataset of a specific response obtained in FEA."""

    def __init__(self, response: Literal['temperature', 'thermal gradient', 'thermal stress'], normalize_inputs: bool=False, transforms: Tuple[Callable, Callable]=None) -> None:
        super().__init__()

        self.parameters = generate_simulation_parameters()
        self.inputs = make_inputs(self.parameters).float()

        if normalize_inputs:
            self.inputs -= self.inputs.min()
            self.inputs /= self.inputs.std()

        if response == 'temperature':
            self.outputs = load_pickle('Thermal 2023-03-23/outputs.pickle')[..., 0].float()
            self.outputs -= self.outputs.min()
        elif response == 'thermal gradient':
            self.outputs = load_pickle('Thermal 2023-03-23/outputs.pickle')[..., 1].float()
        elif response == 'thermal stress':
            self.outputs = load_pickle('Structural 2023-03-23/outputs.pickle')[..., 0].float()
            self.outputs /= self.outputs.max()
            self.outputs *= 78
        else:
            raise Exception(f"Invalid response: '{response}'.")

        if transforms:
            self.transform, self.inverse_transform = transforms

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
    data_original = dataset.outputs.numpy()

    # for x in (1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0):
    #     data = data_original.copy()
    #     data -= data.min()
    #     data /= data.max()
    #     data = np.log(data + x)
    #     data -= data.min()
    #     data /= data.max()
    #     data *= 78
    for x in ('1.50', '2.00', '2.50', '3.00', '3.50', '4.00'):
        x = 1/float(x)
        data = data_original.copy()
        data -= data.min()
        data /= data.max()
        data = data ** x
        data *= 78
        plt.hist(data_original.flatten(), bins=100, alpha=0.5, label='Original')
        plt.hist(data.flatten(), bins=100, alpha=0.5, label=f'{x}')
        plt.legend()
        plt.show()
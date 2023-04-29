"""Load data."""


import matplotlib.pyplot as plt
import numpy as np
import torch
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

def transform_exponentiation(data: torch.Tensor, power: float, inverse: bool):
    """Raise the data to a power. The data is assumed to already be in the range [0, 1]."""

    if not inverse:
        data = data ** power
    else:
        data = data ** (1/power)

    return data

def transform_logarithmic(data: torch.Tensor, input_range: Tuple[float, float], inverse: bool):
    """Scale the data to a range and then apply the natural logarithm. The data is assumed to already be in the range [0, 1]."""

    x_1, x_2 = input_range

    if not inverse:
        data = data * (x_2 - x_1) + x_1
        data = np.log(data)
    else:
        data = np.exp(data)
        data = (data - x_1) / (x_2 - x_1)

    return data

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
    """Load a thermal response dataset obtained in FEA.
    
    Inputs:
    `response`: A string representing the thermal response dataset to load.
    `transformation_exponentiation`: A power to which the labels are raised. Use None for no transformation.
    `transformation_logarithmic`: A tuple defining the range to which the labels are scaled, before the natural logarithm is applied. Use None for no transformation.
    `output_max`: The maximum value to which the labels are scaled after applying any transformations. Use None for no scaling.
    `normalize_inputs`: Normalize the input data to have zero mean and unit variance. Not recommended.
    """

    def __init__(self, response: Literal['temperature', 'thermal gradient', 'thermal stress'], transformation_exponentiation: float=None, transformation_logarithmic: Tuple[float, float]=None, output_max: float=None, normalize_inputs: bool=False) -> None:
        super().__init__()
        self.output_max = output_max

        self.parameters = generate_simulation_parameters()

        # Generate input data.
        self.inputs = make_inputs(self.parameters).float()
        if normalize_inputs:
            self.inputs -= self.inputs.min()
            self.inputs /= self.inputs.std()

        # Load label data from preprocessed .pickle files.
        if response == 'temperature':
            self.outputs = load_pickle(os.path.join('Thermal 2023-03-23', 'outputs_temperature.pickle')).float()
            self.outputs -= self.outputs.min()
        elif response == 'thermal gradient':
            self.outputs = load_pickle(os.path.join('Thermal 2023-03-23', 'outputs_thermal_gradient.pickle')).float()
        elif response == 'thermal stress':
            self.outputs = load_pickle(os.path.join('Structural 2023-03-23', 'outputs_thermal_stress.pickle')).float()
            self.outputs /= self.outputs.max()
            self.outputs *= 78
        else:
            raise Exception(f"Invalid response: '{response}'.")

        # The raw maximum value found in the entire dataset.
        self.output_max_raw = self.outputs.max()

        # Define the transformation and its inverse.
        if transformation_exponentiation is not None:
            self.transformation, self.transformation_parameter = transform_exponentiation, transformation_exponentiation
        elif transformation_logarithmic is not None:
            self.transformation, self.transformation_parameter = transform_logarithmic, transformation_logarithmic
        else:
            # Raise to a power of 1 for no transformation.
            self.transformation, self.transformation_parameter = transform_exponentiation, 1

        # Apply the label transformation.
        self.outputs = self.transform(self.outputs)

        print_dataset_summary(self.inputs, self.outputs)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        return self.inputs[index], self.outputs[index], self.parameters[index]

    def transform(self, y: torch.Tensor):
        # Scale to [0, 1].
        if self.output_max is not None:
            y = y / self.output_max_raw

        # Transform the data and store the resulting minimum and maximum values.
        y = self.transformation(y, self.transformation_parameter, inverse=False)
        self._min, self._max = y.min(), y.max()

        if self.output_max is not None:
            # Scale to [0, 1].
            y = y - self._min
            y = y / (self._max - self._min)

            # Scale the data to have the specified maximum.
            y = y * self.output_max

        return y

    def untransform(self, y: torch.Tensor):
        if self.output_max is not None:
            y = y / self.output_max

            y = y * (self._max - self._min)
            y = y + self._min

        y = self.transformation(y, self.transformation_parameter, inverse=True)

        if self.output_max is not None:
            y = y * self.output_max_raw

        return y


if __name__ == '__main__':
    dataset = FinDataset('temperature')
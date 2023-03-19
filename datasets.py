"""Load data."""


import numpy as np
from torch.utils.data import Dataset

from preprocessing import *


def get_simulation_parameters() -> List[Tuple[float, float]]:
    """Return a list of tuples of simulation parameters for each simulation."""

    heights = np.linspace(5, 10, 6)  # mm
    taper_ratios = np.linspace(0.2, 1, 5)  # —

    return [(height, taper_ratio) for height in heights for taper_ratio in taper_ratios]

def print_simulation_parameters() -> None:
    """Print lines that define the simulation parameters, to be copied to the Ansys script."""

    parameters = get_simulation_parameters()
    for i, (height, taper_ratio) in enumerate(parameters, 1):
        print(f"parameters(1,{i}) = {height},{taper_ratio:.1f}")

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


class TemperatureDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        parameters = get_simulation_parameters()

        self.inputs = make_inputs(parameters).float()
        self.outputs = load_pickle('Temperature/outputs.pickle').float()

        print_dataset_summary(self.inputs, self.outputs)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


if __name__ == '__main__':
    dataset = TemperatureDataset()
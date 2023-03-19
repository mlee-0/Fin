"""Convert data to tensors."""


import glob
import pickle
from typing import *

import numpy as np
import torch


def make_inputs(parameters: List[tuple]) -> torch.Tensor:
    """Return a 4D (data, channels, height, width) tensor for a list of parameters."""

    n = len(parameters)
    channels = 1
    height, width = 20, 40

    array = np.zeros([n, channels, height, width])
    x, y = np.meshgrid(np.arange(height), np.arange(width))

    for height, taper_ratio in parameters:
        pass

    array = torch.tensor(array, dtype=torch.float32)

    return array

def read_output(filename: str) -> np.ndarray:
    """Read the data in the given filename."""

    with open(filename, 'r') as f:
        lines = f.readlines()

    values = []
    for line in lines:
        values.append([float(_) for _ in line.split(',')])
    values.sort(key=lambda _: (_[1], _[2]))
    values = [value for value, x, y in values]

    array = np.reshape(values, (32, 8))
    array = array.transpose()
    array = array[::-1, :]

    return array

def save_pickle(data: Any, filename: str) -> None:
    """Save data as a .pickle file."""

    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved {type(data)} to {filename}.")

def load_pickle(filename: str) -> Any:
    """Read the data in a .pickle file."""

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {type(data)} from {filename}.")

    return data


# Run this file to read simulation data and save as a .pickle file.
if __name__ == '__main__':
    files = glob.glob('Temperature/*.txt')
    files.sort()
    # List of file names corresponding to each simulation, excluding the time step.
    basenames = ['_'.join(file.split('_')[:-1]) for file in files]
    
    outputs = []
    for basename in basenames:
        outputs.append([read_output(file) for file in files if file.startswith(basename)])
    
    # Convert to a tensor.
    outputs = np.array(outputs)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    save_pickle(outputs, 'Temperature/outputs.pickle')
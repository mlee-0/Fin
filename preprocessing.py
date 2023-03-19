"""Convert data to tensors."""


import glob
import pickle
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch


def make_inputs(parameters: List[tuple]) -> torch.Tensor:
    """Return a 4D (data, channels, height, width) tensor for a list of parameters."""

    # Scale factor to apply to the tensor to allow for enough resolution to properly visualize all taper ratios.
    scale = 4

    n = len(parameters)
    channels = 1
    height, width = 10*scale, 20*scale

    array = np.zeros([n, channels, height, width])
    for i, (height_, taper_ratio) in enumerate(parameters):
        height_ *= scale
        height_end = round(taper_ratio * height_)
        points = [
            (0, 0),
            (width - 1, (height_ - height_end)/2),
            (width - 1, (height_ - height_end)/2 + height_end - 1),
            (0, height_ - 1),
        ]

        image = Image.new('1', (width, height))
        draw = ImageDraw.Draw(image)
        draw.polygon(points, fill=1, outline=1)
        array[i, 0, ...] = np.asarray(image, dtype=float)

    array = torch.tensor(array, dtype=torch.float32)

    return array

def read_output(filename: str) -> np.ndarray:
    """Read the data in the given filename and convert to a 2D array."""

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
    basenames = {'_'.join(file.split('_')[:-1]) for file in files}
    basenames = sorted(basenames)
    
    outputs = []
    for basename in basenames:
        outputs.append([read_output(file) for file in files if file.startswith(basename)])
    
    # Convert to a tensor.
    outputs = np.array(outputs)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    save_pickle(outputs, 'Temperature/outputs.pickle')
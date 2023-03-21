"""Convert data to tensors."""


import glob
import os
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
    channels = 2
    height, width = 10*scale, 20*scale

    array = np.zeros([n, channels, height, width])
    for i, (thickness, taper_ratio, convection_coefficient) in enumerate(parameters):
        thickness *= scale
        thickness_end = round(taper_ratio * thickness)
        points = [
            (0, 0),
            (width - 1, (thickness - thickness_end)/2),
            (width - 1, (thickness - thickness_end)/2 + thickness_end - 1),
            (0, thickness - 1),
        ]

        # Create a filled quadrilateral in the first channel.
        image = Image.new('1', (width, height))
        draw = ImageDraw.Draw(image)
        draw.polygon(points, fill=1, outline=1)
        array[i, 0, ...] = np.asarray(image, dtype=float)

        # Create three lines in the second channel.
        draw.rectangle([(0, 0), (width, height)], fill=0)
        draw.line(points, fill=1)
        array[i, 1, ...] = np.asarray(image, dtype=float)
        array[i, 1, ...] *= (convection_coefficient / 100)

    array = torch.tensor(array, dtype=torch.float32)

    return array

def read_simulation(filename: str) -> np.ndarray:
    """Convert the data in the given file into a 3D array, with shape (height, width, response), where the last axis contains different types of responses, such as temperature or stress. The last two values on each line in the file must be the X and Y coordinates of the node, respectively."""

    with open(filename, 'r') as f:
        lines = f.readlines()

    values = []
    for line in lines:
        values.append([float(_) for _ in line.split(',')])
    values.sort(key=lambda _: (_[-2], _[-1]))
    values = [value[:-2] for value in values]

    array = np.reshape(values, (32, 8, -1))
    array = array.transpose([1, 0, 2])
    array = array[::-1, :]

    return array

def read_simulations_transient(folder: str) -> np.ndarray:
    """Convert all transient simulation data in the given folder into a 5D array, with shape (data, time steps, height, width, response).
    
    For transient simulations, there are multiple files corresponding to the same simulation, one for each time step. This function combines the files corresponding to each simulation.
    """

    files = glob.glob(os.path.join(folder, '*.txt'))
    files.sort()
    # List of file name prefixes that correspond to each simulation.
    basenames = {'_'.join(file.split('_')[:-1]) for file in files}
    basenames = sorted(basenames)

    outputs = []
    for basename in basenames:
        outputs.append([read_simulation(file) for file in files if file.startswith(basename)])
    
    return np.array(outputs)

def read_simulations_static(folder: str):
    """Convert all static simulation data in the given folder into a 5D array, with shape (data, 1, height, width, response)."""

    files = glob.glob(os.path.join(folder, '*.txt'))
    files.sort()

    outputs = [read_simulation(file)[None, ...] for file in files]
    
    return np.array(outputs)


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
    outputs = read_simulations_transient('Thermal')
    outputs = torch.tensor(outputs, dtype=torch.float32)
    save_pickle(outputs, 'Thermal/outputs.pickle')

    outputs = read_simulations_static('Structural')
    outputs = torch.tensor(outputs, dtype=torch.float32)
    save_pickle(outputs, 'Structural/outputs.pickle')

    # from metrics import *
    # outputs = read_simulations_transient('Thermal')
    # print(outputs.shape)
    # print(outputs[0].min(), outputs[1].min())
    # plot_comparison(outputs[0, ..., 0], outputs[1, ..., 0])

    # outputs = read_simulations_static('Structural')
    # plot_comparison(outputs[300], outputs[308])

    # inputs = make_inputs([[10, 0.2, 20]])
    # plt.subplot(1, 2, 1)
    # plt.imshow(inputs[0, 0, ...], cmap='gray', vmin=0, vmax=1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(inputs[0, 1, ...], cmap='gray', vmin=0, vmax=1)
    # plt.show()
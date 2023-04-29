"""Read and cache simulation data. Run this file to read simulation data, which are stored as text files, and convert them to tensors and save them as .pickle files."""


import glob
import os
import pickle
from typing import *

import numpy as np
from PIL import Image, ImageDraw
import torch


def make_inputs(parameters: List[tuple]) -> torch.Tensor:
    """Return a 4D tensor with shape (data, channels, height, width) for a list of parameters."""

    # Scale factor to apply to the tensor to allow for enough resolution to properly visualize all taper ratios.
    scale = 4

    n = len(parameters)
    channels = 3
    height, width = 10*scale, 20*scale

    array = np.zeros([n, channels, height, width])
    for i, (thickness, taper_ratio, convection_coefficient, temperature_left) in enumerate(parameters):
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

        # Create a line in the third channel.
        draw.rectangle([(0, 0), (width, height)], fill=0)
        draw.line([points[0], points[-1]], fill=1)
        array[i, 2, ...] = np.asarray(image, dtype=float)
        array[i, 2, ...] *= (temperature_left - 273.15) / 100

    array = torch.tensor(array, dtype=torch.float32)

    return array

def read_simulation(filename: str) -> List[np.ndarray]:
    """Convert the data in the given text file into a list of 2D arrays with shape (height, width), where each array corresponds to a different type of response, such as temperature or stress.
    
    Each line must contain numbers separated by commas, with an arbitrary number of response values followed by the X and Y coordinates of the node:
    response 1, response 2, response 3, ..., X coordinate, Y coordinate
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    values = []
    for line in lines:
        values.append([float(_) for _ in line.split(',')])
    # Sort the values by X coordinate first, then by Y coordinate. Because of the mapped meshes used, nodes are aligned along vertical columns (32 groups of nodes that all share the same X coordinate) but are not aligned along horizontal rows.
    values.sort(key=lambda _: (_[-2], _[-1]))
    # Remove coordinates.
    values = [value[:-2] for value in values]

    # Shape of (32, 8, response).
    array = np.reshape(values, (32, 8, -1))
    # Shape of (8, 32, response).
    array = np.transpose(array, (1, 0, 2))
    # Flip along the y-axis, which is inverted in FEA.
    array = array[::-1, ...]

    return [array[..., response] for response in range(array.shape[-1])]

def read_simulations_transient(folder: str) -> List[np.ndarray]:
    """Find all text files in the given transient response folder and combine all into a list of 4D arrays with shape (data, time steps, height, width), where each array corresponds to a different type of response.
    
    For transient simulations, there are multiple files corresponding to each simulation, one for each time step, which are combined into one array.
    """

    files = glob.glob(os.path.join(folder, '*.txt'))
    files.sort()
    # List of file name prefixes that correspond to each simulation.
    basenames = {'_'.join(file.split('_')[:-1]) for file in files}
    basenames = sorted(basenames)

    outputs = []
    for i, basename in enumerate(basenames, 1):
        print(f"Reading response {i} of {len(basenames)} in {folder}...", end='\r')
        outputs.append([read_simulation(file) for file in files if file.startswith(basename)])
    print()
    # Shape of (data, time steps, response, height, width).
    outputs = np.array(outputs)

    return [outputs[:, :, response, :, :] for response in range(outputs.shape[2])]

def read_simulations_static(folder: str) -> List[np.ndarray]:
    """Find all text files in the given static response folder and combine all into a list of 4D arrays with shape (data, 1, height, width), where each array corresponds to a different type of response.
    
    For static simulations, there is a single file corresponding to each simulation.
    """

    files = glob.glob(os.path.join(folder, '*.txt'))
    files.sort()

    outputs = []
    for i, file in enumerate(files, 1):
        print(f"Reading response {i} of {len(files)} in {folder}...", end='\r')
        outputs.append(read_simulation(file))
    print()
    # Shape of (data, response, height, width).
    outputs = np.array(outputs)
    # Add a channel dimension for a shape of (data, 1, response, height, width).
    outputs = outputs[:, None, ...]
    
    return [outputs[:, :, response, :, :] for response in range(outputs.shape[2])]


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


if __name__ == '__main__':
    outputs_temperature, outputs_thermal_gradient, *_ = read_simulations_transient('Thermal 2023-03-23')
    outputs_temperature = torch.tensor(outputs_temperature, dtype=torch.float32)
    outputs_thermal_gradient = torch.tensor(outputs_thermal_gradient, dtype=torch.float32)
    save_pickle(outputs_temperature, 'Thermal 2023-03-23/outputs_temperature.pickle')
    save_pickle(outputs_thermal_gradient, 'Thermal 2023-03-23/outputs_thermal_gradient.pickle')

    outputs_thermal_stress, *_ = read_simulations_static('Structural 2023-03-23')
    outputs_thermal_stress = torch.tensor(outputs_thermal_stress, dtype=torch.float32)
    save_pickle(outputs_thermal_stress, 'Structural 2023-03-23/outputs_thermal_stress.pickle')
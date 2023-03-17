"""Convert data to tensors."""


import pickle
from typing import *

import torch


def make_input_tensor() -> torch.Tensor:
    """Return a (1, channels, height, width) tensor for a single input data."""
    return

def make_output_tensor() -> torch.Tensor:
    """Return a (1, channels, height, width) tensor for a single output data."""
    return

def read_output(filename: str):
    """Read the data in the given filename."""
    return

def save_pickle(data: Any, filename: str) -> None:
    """Save data as a .pickle file."""
    return

def load_pickle(filename: str) -> Any:
    """Read the data in a .pickle file."""
    return
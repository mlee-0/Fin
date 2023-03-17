"""Load data."""


import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def get_simulation_parameters():
    height = np.linspace(2, 10, 2)  # mm
    taper_ratio = np.linspace(0.2, 1, 0.2)  # —


class TemperatureDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.inputs = ...
        self.outputs = ...

    def __len__(self) -> int:
        return
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
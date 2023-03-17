"""Evaluation metrics and result visualizations."""


import numpy as np
import matplotlib.pyplot as plt


def mae(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(prediction - true))

def mse(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean((prediction - true) ** 2)

def mre(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean((prediction - true) / true) * 100

def plot_parity(prediction: np.ndarray, true: np.ndarray) -> None:
    plt.figure()
    plt.plot(true.flatten(), prediction.flatten(), '.', linewidth=1)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'k--')
    plt.xlabel('True')
    plt.xlabel('Prediction')
    plt.show()

def plot_comparison(prediction: np.ndarray, true: np.ndarray) -> None:
    """Plot each 2D channel of predicted and true responses."""

    channels = true.shape[1]

    plt.figure()
    for i in range(0, channels, 2):
        plt.subplot(channels, 2, i+1)
        plt.imshow(prediction, cmap='Spectral_r')
        plt.ylabel(f'Channel {i+1}')
        if i == 0:
            plt.title('Prediction')

        plt.subplot(channels, 2, i+2)
        plt.imshow(true, cmap='Spectral_r')
        plt.ylabel(f'Channel {i+1}')
        if i == 0:
            plt.title('True')
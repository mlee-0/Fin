"""Evaluation metrics and result visualizations."""


import numpy as np
import matplotlib.pyplot as plt


def mae(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(prediction - true))

def mse(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean((prediction - true) ** 2)

def rmse(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.sqrt(np.mean((prediction - true) ** 2))

def mre(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(prediction - true) / true) * 100

def plot_parity(prediction: np.ndarray, true: np.ndarray) -> None:
    plt.figure()
    plt.plot(true.flatten(), prediction.flatten(), '.', linewidth=1)
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'k--')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.subplots_adjust(left=0.1, right=0.975)
    plt.show()

def plot_comparison(prediction: np.ndarray, true: np.ndarray, title: str=None) -> None:
    """Plot each 2D channel of predicted and true responses, both given as 3D arrays."""

    channels = true.shape[0]

    plt.figure(figsize=(5, 6))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.90, top=0.90)
    min_value, max_value = true.min(), true.max()

    for i in range(channels):
        plt.subplot(channels, 2, i*2+1)
        plt.imshow(prediction[i, ...], cmap='Spectral_r', vmin=min_value, vmax=max_value)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('Prediction')
        colorbar = plt.colorbar(ticks=[min_value, max_value], fraction=0.05, aspect=10)
        colorbar.ax.tick_params(labelsize=6)

        plt.subplot(channels, 2, i*2+2)
        plt.imshow(true[i, ...], cmap='Spectral_r', vmin=min_value, vmax=max_value)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('True')
        colorbar = plt.colorbar(ticks=[min_value, max_value], fraction=0.05, aspect=10)
        colorbar.ax.tick_params(labelsize=6)

    if title:
        plt.suptitle(title)

    plt.show()


if __name__ == '__main__':
    pass
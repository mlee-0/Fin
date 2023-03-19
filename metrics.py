"""Evaluation metrics and result visualizations."""


import numpy as np
import matplotlib.pyplot as plt


def mae(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(prediction - true))

def mse(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean((prediction - true) ** 2)

def mre(prediction: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(prediction - true) / true) * 100

def plot_parity(prediction: np.ndarray, true: np.ndarray) -> None:
    plt.figure()
    plt.plot(true.flatten(), prediction.flatten(), '.', linewidth=1)
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'k--')
    plt.xlabel('True')
    plt.xlabel('Prediction')
    plt.show()

def plot_comparison(prediction: np.ndarray, true: np.ndarray) -> None:
    """Plot each 2D channel of predicted and true responses, both given as 3D arrays."""

    channels = true.shape[0]

    plt.figure()
    min_value, max_value = true.min(), true.max()

    for i in range(channels):

        plt.subplot(channels, 2, i*2+1)
        axis = plt.imshow(prediction[i, ...], cmap='Spectral_r', vmin=min_value, vmax=max_value)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('Prediction')
            plt.colorbar(axis, ticks=[min_value, max_value])

        plt.subplot(channels, 2, i*2+2)
        axis = plt.imshow(true[i, ...], cmap='Spectral_r', vmin=min_value, vmax=max_value)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title('True')
            plt.colorbar(axis, ticks=[min_value, max_value])

    plt.show()


if __name__ == '__main__':
    from preprocessing import load_pickle
    outputs = load_pickle('Temperature/outputs.pickle').numpy()
    print(outputs.shape)
    i = 300-1
    plot_comparison(outputs[i], outputs[i])
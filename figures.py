import matplotlib.pyplot as plt

from datasets import *
from preprocessing import *


def plot_inputs():
    parameters = generate_simulation_parameters()
    parameters = [parameters[91], parameters[244], parameters[509]]
    inputs = make_inputs(parameters).numpy()

    for i in range(inputs.shape[0]):
        plt.subplot(inputs.shape[0], 2, i*2+1)
        plt.imshow(inputs[i, 0, ...], cmap='gray', vmin=0, vmax=1)
        plt.ylabel(str(parameters[i]))
        if i == 0:
            plt.title('Channel 1')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(inputs.shape[0], 2, i*2+2)
        plt.imshow(inputs[i, 1, ...], cmap='gray', vmin=0, vmax=1)
        if i == 0:
            plt.title('Channel 2')
        plt.xticks([])
        plt.yticks([])

    plt.show()


if __name__ == '__main__':
    plot_inputs()
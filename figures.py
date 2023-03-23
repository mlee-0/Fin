import matplotlib.pyplot as plt

from datasets import *
from preprocessing import *


def plot_inputs():
    """Show examples of input images."""

    indices = [91, 244, 509]
    parameters = generate_simulation_parameters()

    for i, index in enumerate(indices):
        array = make_inputs(parameters[index:index+1])

        plt.subplot(len(indices), 2, i*2+1)
        plt.imshow(array[0, 0, ...], cmap='gray', vmin=0, vmax=1)
        plt.ylabel(str(parameters[index]))
        if i == 0:
            plt.title('Channel 1')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(indices), 2, i*2+2)
        plt.imshow(array[0, 1, ...], cmap='gray', vmin=0, vmax=1)
        if i == 0:
            plt.title('Channel 2')
        plt.xticks([])
        plt.yticks([])

    plt.show()

def plot_labels():
    """Show examples of labels."""

    indices = [91]#, 244, 509]
    outputs_thermal = load_pickle('Thermal 2023-03-21/outputs.pickle').numpy()
    outputs_structural = load_pickle('Structural 2023-03-21/outputs.pickle').numpy()

    rows = outputs_thermal.shape[1]

    for index in indices:
        plt.figure()
        # plt.subplots_adjust(top=0.95, bottom=0.05)

        for t in range(rows):
            plt.subplot(rows, 3, t*3+1)
            plt.imshow(outputs_thermal[index, t, ..., 0], cmap='Spectral_r')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f"{(t+1)*10} s", rotation=0, horizontalalignment='right', verticalalignment='center')
            if t == 0:
                plt.title('Temperature')

            plt.subplot(rows, 3, t*3+2)
            plt.imshow(outputs_thermal[index, t, ..., 1], cmap='Spectral_r')
            plt.xticks([])
            plt.yticks([])
            if t == 0:
                plt.title('Thermal Gradient')

            # Show a blank image for thermal stress.
            plt.subplot(rows, 3, t*3+3)
            if t < rows-1:
                plt.imshow(np.ones(outputs_structural.shape[-2:]), cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
            else:
                plt.imshow(outputs_structural[index, 0, ...], cmap='Spectral_r')
                plt.xticks([])
                plt.yticks([])
            if t == 0:
                plt.title('Thermal Stress')

    plt.show()

if __name__ == '__main__':
    plot_labels()
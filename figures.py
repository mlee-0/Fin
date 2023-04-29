"""Miscellaneous plots of data and results."""


from matplotlib import cm
import matplotlib.pyplot as plt
import random

from datasets import *
from preprocessing import *


def plot_inputs(index: int=None):
    """Show examples of input images."""

    if index is None:
        index = random.choice(range(4800))
    parameters = generate_simulation_parameters()[index]
    array = make_inputs([parameters])

    plt.figure(figsize=(4, 5))

    plt.subplot(3, 1, 1)
    plt.imshow(array[0, 0, ...], cmap='gray', vmin=0, vmax=1)
    plt.colorbar(ticks=[0, 1])
    plt.title(f'Thickness = {parameters[0]} mm, Taper Ratio = {parameters[1]}')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 2)
    plt.imshow(array[0, 1, ...], cmap='gray', vmin=0, vmax=1)
    plt.colorbar(ticks=[0, 1])
    plt.title(fr'Convection Coefficient = {parameters[2]} W/m$^2$K')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3, 1, 3)
    plt.imshow(array[0, 2, ...], cmap='gray', vmin=0, vmax=1)
    plt.colorbar(ticks=[0, 1])
    plt.title(f'Temperature Boundary Condition = {parameters[3]} K')
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()

def plot_labels():
    """Show examples of labels."""

    outputs_thermal = load_pickle('Thermal 2023-03-23/outputs.pickle').numpy()
    outputs_structural = load_pickle('Structural 2023-03-23/outputs.pickle').numpy()
    indices = [79]  #random.sample(range(outputs_thermal.shape[0]), k=1)
    parameters = generate_simulation_parameters()

    rows = outputs_thermal.shape[1]

    for index in indices:
        temperature = outputs_thermal[index, ..., 0]
        thermal_gradient = outputs_thermal[index, ..., 1]
        thermal_stress = outputs_structural[index, ..., 0]

        plt.figure()
        # plt.subplots_adjust(top=0.95, bottom=0.05)

        for t in range(rows):
            plt.subplot(rows, 3, t*3+1)
            plt.imshow(temperature[t], cmap='Spectral_r', vmin=temperature.min(), vmax=temperature.max())
            colorbar = plt.colorbar(ticks=[temperature.min(), temperature.max()], fraction=0.05, aspect=10)
            colorbar.ax.tick_params(labelsize=6)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f"{(t+1)*10} s", rotation=0, horizontalalignment='right', verticalalignment='center')
            if t == 0:
                plt.title('Temperature')

            plt.subplot(rows, 3, t*3+2)
            plt.imshow(thermal_gradient[t], cmap='Spectral_r', vmin=thermal_gradient.min(), vmax=thermal_gradient.max())
            colorbar = plt.colorbar(ticks=[thermal_gradient.min(), thermal_gradient.max()], fraction=0.05, aspect=10)
            colorbar.ax.tick_params(labelsize=6)
            plt.xticks([])
            plt.yticks([])
            if t == 0:
                plt.title('Thermal Gradient')

            # Show a blank image for thermal stress.
            plt.subplot(rows, 3, t*3+3)
            if t < rows-1:
                plt.imshow(np.ones(thermal_stress.shape[-2:]), cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
            else:
                plt.imshow(thermal_stress[0], cmap='Spectral_r', vmin=thermal_stress.min(), vmax=thermal_stress.max())
                colorbar = plt.colorbar(ticks=[thermal_stress.min(), thermal_stress.max()], fraction=0.05, aspect=10)
                colorbar.ax.tick_params(labelsize=6)
                plt.xticks([])
                plt.yticks([])
            if t == 0:
                plt.title('Thermal Stress')

        plt.suptitle(str(parameters[index]))

    plt.show()

def plot_histograms_labels():
    """Histograms of the 3 responses."""

    for i, name in enumerate(['Temperature', 'Thermal Gradient', 'Thermal Stress']):
        dataset = FinDataset(name.lower())
        plt.subplot(1, 3, i+1)
        plt.hist(dataset.outputs.numpy().flatten(), bins=100)
        plt.title(name)

    plt.show()

def plot_hyperparameter_tuning_random_search(metric_name: Literal['mae', 'mse']):
    """Random search results."""

    results = {
        (-0.3412, 119, 78): (3.52896, 54.14465),
        (-3.2443, 46, 61): (0.83290, 3.42286),
        (-2.5316, 117, 35): (0.71798, 2.92119),
        (-0.4103, 126, 54): (3.09415, 48.63020),
        (-2.6539, 100, 34): (0.48940, 1.50047),
        (-2.7264, 126, 33): (0.48423, 1.40573),
        (-1.2824, 29, 2): (1.09687, 7.81357),
        (-1.8417, 70, 10): (0.52119, 1.95805),
        (-1.2817, 25, 45): (0.88517, 3.12811),
        (-2.9209, 98, 61): (0.31030, 1.01668),
        (-3.8509, 53, 4): (3.06154, 43.78319),
        (-3.911, 68, 28): (0.83196, 2.44644),
        (-0.1186, 101, 22): (1.04123, 5.87454),
        (-1.7963, 25, 19): (0.74936, 2.54645),
        (-0.859, 67, 22): (0.58083, 1.87270),
        (-0.851, 13, 19): (1.61699, 7.53452),
        (-0.5487, 15, 68): (3.15306, 49.87796),
        (-3.9303, 119, 17): (2.68603, 32.52049),
        (-2.7486, 49, 8): (0.35562, 0.53378),
        (-1.4123, 27, 36): (0.49547, 1.19986),
    }

    hyperparameter_names = ['Learning Rate Exponent', 'Batch Size', 'Model Size']

    hyperparameters = list(zip(*results.keys()))
    metrics = list(zip(*results.values()))
    if metric_name == 'mae':
        metrics = metrics[0]
    elif metric_name == 'mse':
        metrics = metrics[1]
    metrics = np.array(metrics)

    plt.figure()
    for j, hyperparameter in enumerate(hyperparameters):
        plt.subplot(1, len(hyperparameter_names), j+1)
        plt.scatter(hyperparameter, metrics)
        plt.xlabel(hyperparameter_names[j])
    plt.tight_layout()

    colormap = cm.get_cmap('Greens_r')
    plt.figure()
    for row, hyperparameter_1 in enumerate(hyperparameters):
        for column, hyperparameter_2 in enumerate(hyperparameters):
            if hyperparameter_1 is not hyperparameter_2 and column >= row:
                plt.subplot(len(hyperparameters), len(hyperparameters), row*len(hyperparameters)+(column+1))
                plt.grid()
                plt.scatter(hyperparameter_1, hyperparameter_2, color=colormap(metrics / metrics.max()))
                plt.xlabel(hyperparameter_names[row])
                plt.ylabel(hyperparameter_names[column])
    plt.tight_layout()

    plt.show()

def plot_hyperparameter_tuning_grid_search(metric_name: Literal['mae', 'mse', 'vminmse']):
    """Grid search results."""

    results = {
        (-3.5, 1, 2): (2.41314, 29.15243, 28.73374392191569),
        (-3.5, 1, 8): (2.19716, 24.92327, 24.57736790974935),
        (-3.5, 1, 16): (4.80206, 87.91097, 80.87745259602865),
        (-3.5, 1, 32): (2.58831, 47.80976, 23.39824822743734),
        (-3.5, 1, 64): (3.06266, 102.92319, 57.29917627970378),
        (-3.5, 8, 2): (0.43397, 0.66670, 0.6656761686007182),
        (-3.5, 8, 8): (0.44905, 1.08761, 1.0807913144429524),
        (-3.5, 8, 16): (0.40750, 0.82915, 0.5882941822210948),
        (-3.5, 8, 32): (0.19444, 0.16613, 0.17208831906318664),
        (-3.5, 8, 64): (0.28548, 0.58988, 0.5511559863885244),
        (-3.5, 16, 2): (0.66900, 2.56300, 2.291260560353597),
        (-3.5, 16, 8): (0.52275, 1.25559, 1.2480418761571248),
        (-3.5, 16, 16): (0.87891, 3.77172, 0.6135108013947804),
        (-3.5, 16, 32): (0.41915, 1.10876, 0.6884767691294352),
        (-3.5, 16, 64): (0.47810, 1.17794, 0.9844782590866089),
        (-3.5, 64, 2): (2.74504, 34.43520, 34.230089314778645),
        (-3.5, 64, 8): (0.79832, 4.22039, 4.189822483062744),
        (-3.5, 64, 16): (0.81548, 3.36345, 2.2073384364446005),
        (-3.5, 64, 32): (0.50848, 3.51168, 3.475134515762329),
        (-3.5, 64, 64): (0.31587, 0.43407, 0.16711463630199433),

        (-3.0, 1, 2): (3.66277, 95.52898, 60.61544189453125),
        (-3.0, 1, 8): (1.98977, 18.83694, 11.747453625996908),
        (-3.0, 1, 16): (1.75979, 15.69064, 6.7827672799428305),
        (-3.0, 1, 32): (2.08351, 24.47745, 16.165025774637858),
        (-3.0, 1, 64): (2.27322, 33.83149, 22.809620539347332),
        (-3.0, 8, 2): (0.72691, 2.36171, 0.758070973555247),
        (-3.0, 8, 8): (0.72508, 3.36323, 0.557118026415507),
        (-3.0, 8, 16): (0.29260, 0.38174, 0.38832791248957316),
        (-3.0, 8, 32): (0.28055, 0.32482, 0.3334269940853119),
        (-3.0, 8, 64): (0.33419, 0.51038, 0.403630264600118),
        (-3.0, 16, 2): (0.45849, 0.93196, 0.32577839493751526),
        (-3.0, 16, 8): (0.65520, 2.07029, 0.32516724864641827),
        (-3.0, 16, 16): (0.43163, 0.86900, 0.4739525298277537),
        (-3.0, 16, 32): (1.13079, 8.88445, 0.6025525212287903),
        (-3.0, 16, 64): (0.32176, 0.41463, 0.17805491387844086),
        (-3.0, 64, 2): (0.47130, 0.67348, 0.6813382863998413),
        (-3.0, 64, 8): (1.11210, 6.76848, 1.8183601379394532),
        (-3.0, 64, 16): (0.65361, 1.94653, 1.0181149005889893),
        (-3.0, 64, 32): (0.64821, 2.00014, 1.992452319463094),
        (-3.0, 64, 64): (0.28154, 0.29290, 0.29470769862333934),
        
        (-2.5, 1, 2): (2.57331, 27.96182, 21.202588081359863),
        (-2.5, 1, 8): (3.27092, 44.69901, 45.581555684407554),
        (-2.5, 1, 16): (2.78079, 76.63221, 28.86910451253255),
        (-2.5, 1, 32): (7.54907, 126.37780, 46.849290974934895),
        (-2.5, 1, 64): (1.87310, 21.88341, 10.990947119394939),
        (-2.5, 8, 2): (2.95795, 48.16554, 45.775838470458986),
        (-2.5, 8, 8): (0.44656, 0.91930, 0.8981006979942322),
        (-2.5, 8, 16): (0.52030, 2.02159, 0.31999359130859373),
        (-2.5, 8, 32): (0.85807, 4.28248, 0.47566845019658405),
        (-2.5, 8, 64): (0.43043, 0.74229, 0.3261733551820119),
        (-2.5, 16, 2): (0.29972, 0.38445, 0.3903211236000061),
        (-2.5, 16, 8): (0.85477, 3.44729, 0.2842241138219833),
        (-2.5, 16, 16): (0.35191, 0.45321, 0.4670252819856008),
        (-2.5, 16, 32): (0.45699, 0.77004, 0.3030557751655579),
        (-2.5, 16, 64): (2.62707, 54.11758, 0.5290398796399435),
        (-2.5, 64, 2): (0.64153, 1.54925, 1.2205420017242432),
        (-2.5, 64, 8): (0.39248, 0.55464, 0.5637620985507965),
        (-2.5, 64, 16): (0.21837, 0.19095, 0.1958310087521871),
        (-2.5, 64, 32): (0.37452, 0.78990, 0.31419180432955424),
        (-2.5, 64, 64): (0.48112, 1.52119, 0.42902090748151145),
    }

    hyperparameter_names = ['Learning Rate Exponent', 'Batch Size', 'Model Size']

    hyperparameters = [np.array(_) for _ in zip(*results.keys())]
    metrics = list(zip(*results.values()))
    if metric_name == 'mae':
        metrics = metrics[0]
    elif metric_name == 'mse':
        metrics = metrics[1]
    # Validation minimum MSE.
    elif metric_name == 'vminmse':
        metrics = metrics[2]
    metrics = np.reshape(metrics, (3, 4, 5))

    for axis_i in range(metrics.ndim):
        axis_j, axis_k = [_ for _ in range(metrics.ndim) if _ != axis_i]

        plt.figure()
        for subplot in range(metrics.shape[axis_i]):
            plt.subplot(1, metrics.shape[axis_i], subplot+1)
            plt.imshow(metrics.take(subplot, axis=axis_i), cmap='Greens_r', vmin=metrics.min(), vmax=metrics.max())
            plt.xticks(ticks=range(metrics.shape[axis_k]), labels=np.unique(hyperparameters[axis_k]))
            plt.yticks(ticks=range(metrics.shape[axis_j]), labels=np.unique(hyperparameters[axis_j]))
            plt.xlabel(hyperparameter_names[axis_k])
            plt.ylabel(hyperparameter_names[axis_j])
            plt.title(f"{hyperparameter_names[axis_i]} = {np.unique(hyperparameters[axis_i])[subplot]}")
            # colorbar = plt.colorbar(ticks=[metrics.min(), metrics.max()], fraction=0.01)
            # colorbar.set_ticklabels(['Best', 'Worst'])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)

    plt.show()

def plot_lt_exp_histograms():
    """Exponentiation label transformation metrics."""

    dataset = FinDataset('temperature')
    data = dataset.outputs.numpy().flatten()
    data -= data.min()
    data /= data.max()

    plt.figure()

    for i, power in enumerate((1.50, 2.00, 2.50, 3.00)):
        plt.subplot(1, 4, i+1)
        plt.hist(data, bins=50, color=[0.5]*3, alpha=0.5, label=f'Original')
        plt.hist(data ** (1 / power), bins=50, alpha=0.5, label=f'1/{power}')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
    
    plt.show()

def plot_lt_log_histograms():
    """Logarithmic label transformation metrics."""

    dataset = FinDataset('temperature')
    data = dataset.outputs.numpy().flatten()
    data -= data.min()
    data /= data.max()

    for i, x1 in enumerate((1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0)):
        transformed = data.copy()
        transformed -= transformed.min()
        transformed /= transformed.max()
        transformed += x1
        transformed = np.log(transformed)
        transformed -= transformed.min()
        transformed /= transformed.max()
        plt.subplot(2, 4, i+1)
        plt.hist(data, bins=50, color=[0.5]*3, alpha=0.5, label=f'Original')
        plt.hist(transformed, bins=50, alpha=0.5, label=f'({x1}, 1+{x1})')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
    
    plt.show()

def plot_lt_exp_results():
    """Exponentiation label transformation metrics."""

    x = ('1/3.00', '1/2.50', '1/2.00', '1/1.75', '1/1.50', '1/1.25')
    results = [
        [0.40231, 1.29919, 1.13982, 1.43221, 4.04778, 2.01191],
        [0.24069, 0.47855, 0.69178, 2.37617, 11.91026, 3.45112],
        [0.23893, 0.33682, 0.58036, 2.13630, 5.85784, 2.42030],
        [0.30688, 0.68685, 0.82876, 2.96126, 17.44611, 4.17685],
        [0.22451, 0.25370, 0.50369, 1.51836, 3.30359, 1.81758],
        [0.28119, 0.39734, 0.63035, 1.28269, 2.48790, 1.57731],
    ]
    result_baseline = [0.34830, 0.38985, 0.62438, 1.74938, 3.52862, 1.87846]
    mae, mse, rmse, maxima_mae, maxima_mse, maxima_rmse = list(zip(*results))

    plt.figure(figsize=(8, 3))

    plt.subplot(1, 3, 1)
    plt.plot(mae, '.-')
    plt.axhline(result_baseline[0], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mae)), labels=x, rotation=90)
    plt.legend()
    plt.title('MAE')

    plt.subplot(1, 3, 2)
    plt.plot(mse, '.-')
    plt.axhline(result_baseline[1], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mse)), labels=x, rotation=90)
    plt.legend()
    plt.title('MSE')

    plt.subplot(1, 3, 3)
    plt.plot(rmse, '.-')
    plt.axhline(result_baseline[2], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(rmse)), labels=x, rotation=90)
    plt.legend()
    plt.title('RMSE')

    plt.tight_layout()
    plt.show()

def plot_lt_log_results():
    """Logarithm label transformation metrics."""

    x = ('1e-10', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1e-0')
    results = [
        [1.34812, 11.19305, 3.34560, 10.41481, 135.02179, 11.61989],
        [0.51847, 2.18231, 1.47726, 1.49165, 3.48687, 1.86732],
        [0.45072, 1.38373, 1.17632, 5.44326, 36.89414, 6.07405],
        [0.29617, 0.72960, 0.85417, 3.03669, 13.53780, 3.67938],
        [0.28363, 0.67760, 0.82317, 1.29165, 3.05590, 1.74811],
        [0.27384, 0.42481, 0.65177, 3.66738, 17.27131, 4.15588],
        [0.40119, 0.92861, 0.96365, 1.66493, 5.44539, 2.33354],
    ]
    result_baseline = [0.34830, 0.38985, 0.62438, 1.74938, 3.52862, 1.87846]
    mae, mse, rmse, maxima_mae, maxima_mse, maxima_rmse = list(zip(*results))

    plt.figure(figsize=(7, 3))

    plt.subplot(1, 3, 1)
    plt.plot(mae, '.-')
    plt.axhline(result_baseline[0], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mae)), labels=x, rotation=90)
    plt.legend()
    plt.title('MAE')

    plt.subplot(1, 3, 2)
    plt.plot(mse, '.-')
    plt.axhline(result_baseline[1], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mse)), labels=x, rotation=90)
    plt.legend()
    plt.title('MSE')

    plt.subplot(1, 3, 3)
    plt.plot(rmse, '.-')
    plt.axhline(result_baseline[2], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(rmse)), labels=x, rotation=90)
    plt.legend()
    plt.title('RMSE')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
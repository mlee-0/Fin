"""Train and test the model."""


import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader

from metrics import *
from models import *
from datasets import *


FOLDER_CHECKPOINTS = 'Checkpoints'


def save_model(filename: str, **kwargs) -> None:
    torch.save(kwargs, filename)
    print(f"Saved model to {filename}.")

def load_model(filename: str) -> dict:
    checkpoint = torch.load(filename, map_location=None)
    print(f"Loaded model from {filename} trained for {checkpoint['epoch']} epochs.")
    return checkpoint

def split_dataset(dataset_size: int, splits: List[float]) -> List[int]:
    """Return the subset sizes according to the fractions defined in `splits`."""

    assert sum(splits) == 1.0, f"The fractions {splits} must sum to 1."

    # Define the last subset size as the remaining number of data to ensure that they all sum to dataset_size.
    subset_sizes = []
    for fraction in splits[:-1]:
        subset_sizes.append(int(fraction * dataset_size))
    subset_sizes.append(dataset_size - sum(subset_sizes))

    return subset_sizes

def plot_loss(losses_training: List[float], losses_validation: List[float]) -> None:
    plt.figure()
    plt.semilogy(range(1, len(losses_training)+1), losses_training, '-', label='Training')
    plt.semilogy(range(1, len(losses_validation)+1), losses_validation, '-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, train_dataloader: DataLoader):
    """Train on the training dataset."""

    model.train(True)
    loss = 0
    for batch, (input_data, label_data, *_) in enumerate(train_dataloader, 1):
        # Predict an output from the model with the given input.
        output_data = model(input_data)
        # Calculate the loss.
        loss_current = loss_function(output_data, label_data)
        # Update the cumulative loss.
        loss += loss_current.item() * input_data.size(0)
        
        # Reset gradients of model parameters.
        optimizer.zero_grad()
        # Calculate gradients.
        loss_current.backward()
        # Adjust model parameters.
        optimizer.step()

        if batch % 5 == 0:
            print(f"Batch {batch}/{len(train_dataloader)}: {loss_current.item():,.2e}...", end='\r')

    return loss / len(train_dataloader.dataset)

def validate_model(model: torch.nn.Module, loss_function: torch.nn.Module, validate_dataloader: DataLoader):
    """Test on the validation dataset."""

    # Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
    model.train(False)
    loss = 0
    outputs = []
    labels = []
    with torch.no_grad():
        for batch, (input_data, label_data, *_) in enumerate(validate_dataloader, 1):
            output_data = model(input_data)
            loss += loss_function(output_data, label_data.float()).item() * input_data.size(0)

            output_data = output_data.cpu()
            label_data = label_data.cpu()

            outputs.append(output_data)
            labels.append(label_data)

            if batch % 10 == 0:
                print(f"Batch {batch}/{len(validate_dataloader)}...", end='\r')

    # # Calculate evaluation metrics on validation results.
    # outputs = torch.cat(outputs, dim=0)
    # labels = torch.cat(labels, dim=0)
    # evaluate_results(outputs.numpy(), labels.numpy())

    return loss / len(validate_dataloader.dataset)

def test_model(
    model: torch.nn.Module, loss_function: torch.nn.Module, test_dataloader: DataLoader,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    model.train(False)

    loss = 0
    inputs = []
    outputs = []
    labels = []

    with torch.no_grad():
        for batch, (input_data, label_data, *_) in enumerate(test_dataloader, 1):
            output_data = model(input_data)
            loss += loss_function(output_data, label_data.float()).item()

            input_data = input_data.cpu().detach()
            output_data = output_data.cpu().detach()
            label_data = label_data.cpu()

            inputs.append(input_data)
            labels.append(label_data)
            outputs.append(output_data)
            
            if batch % 10 == 0:
                print(f"Batch {batch}/{len(test_dataloader)}...", end='\r')

    loss /= batch
    print(f"Testing loss: {loss:,.2e}")

    # Concatenate testing results from all batches into a single array.
    inputs = torch.cat(inputs, dim=0)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, labels, inputs

def evaluate_results(outputs: np.ndarray, labels: np.ndarray):
    """Print and return evaluation metrics."""

    results = {
        'MAE': mae(outputs, labels),
        'MSE': mse(outputs, labels),
        'RMSE': rmse(outputs, labels),
        'MRE': mre(outputs, labels),
    }
    for metric, value in results.items():
        print(f"{metric}: {value:,.5f}")

    return results

def main(
    epoch_count: int, learning_rate: float, batch_sizes: Tuple[int, int, int], dataset_split: Tuple[float, float, float],
    train: bool, test: bool, train_existing: bool, save_model_every: int, save_best_separately: bool,
    dataset: Dataset, model: torch.nn.Module, filename_model: str, loss_function: torch.nn.Module, Optimizer: torch.optim.Optimizer, scheduler=None,
    show_loss: bool=True, show_parity: bool=True, show_results: bool=True,
) -> None:
    """
    Inputs:
    `epoch_count`: Number of epochs to train.
    `learning_rate`: Learning rate for the optimizer.
    `batch_sizes`: Tuple of batch sizes for the training, validation, and testing datasets.
    `dataset_split`: A tuple of three floats in [0, 1] of the training, validation, and testing ratios.

    `train`: Train the model.
    `test`: Test the model.
    `evaluate`: Calculate evaluation metrics on the testing results.
    `train_existing`: Load a previously saved model and continue training.
    `save_model_every`: Number of epochs after which to save the model.
    `save_best_separately`: Save the best model as a separate file when the lowest validation loss so far is observed.

    `model`: The network, as an instance of a Module subclass.
    `filename_model`: Name of the .pth file to load and save to during training.
    `dataset`: The dataset, as an instance of a Dataset subclass.
    `loss_function`: The loss function, as an instance of a Module subclass.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `scheduler`: A learning rate scheduler.

    `show_loss`: Show a loss history plot.
    `show_parity`: Show a parity plot after testing.
    `show_results`: Show the results after testing.
    """

    filepath_model = os.path.join(FOLDER_CHECKPOINTS, filename_model)

    # Load the previously saved checkpoint.
    if (test and not train) or (train and train_existing):
        checkpoint = load_model(filepath_model)
        # Load the last learning rate used.
        learning_rate = checkpoint.get('learning_rate', learning_rate)
    else:
        checkpoint = {}

    dataset.outputs = dataset.transform(dataset.outputs)

    # Split the dataset into training, validation, and testing.
    train_dataset, validate_dataset, test_dataset = random_split(
        dataset,
        split_dataset(len(dataset), dataset_split),
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_sizes[1], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False)
    print(f"Split {len(dataset):,} data into {len(train_dataset):,} training / {len(validate_dataset):,} validation / {len(test_dataset):,} testing.")

    # Initialize the optimizer.
    optimizer = Optimizer(model.parameters(), lr=learning_rate)

    # Load previously saved model and optimizer parameters.
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if train:
        epoch = checkpoint.get('epoch', 0) + 1
        epochs = range(epoch, epoch+epoch_count)

        losses_training = checkpoint.get('losses_training', [])
        losses_validation = checkpoint.get('losses_validation', [])

        for epoch in epochs:
            time_start = time.time()
            
            loss = train_model(model, optimizer, loss_function, train_dataloader)
            losses_training.append(loss)

            # Adjust the learning rate if a scheduler is used.
            if scheduler:
                scheduler.step()
                learning_rate = optimizer.param_groups[0]["lr"]
                print(f"Learning rate: {learning_rate}")

            loss = validate_model(model, loss_function, validate_dataloader)
            losses_validation.append(loss)

            # Show a summary of the epoch.
            time_end = time.time()
            duration = time_end - time_start
            if duration >= 60:
                duration_text = f"{duration/60:.1f} minutes"
            else:
                duration_text = f"{duration:.1f} seconds"
            print(f"Epoch {epoch}/{epochs[-1]} ({time.strftime('%I:%M %p')}, {duration_text}): {losses_training[-1]:,.2e} (training), {losses_validation[-1]:,.2e} (validation)")

            # Save the model periodically and in the last epoch.
            if epoch % save_model_every == 0 or epoch == epochs[-1]:
                save_model(
                    filepath_model,
                    epoch = epoch,
                    model_state_dict = model.state_dict(),
                    optimizer_state_dict = optimizer.state_dict(),
                    learning_rate = optimizer.param_groups[0]['lr'],
                    losses_training = losses_training,
                    losses_validation = losses_validation,
                )
            # Save the model if the model achieved the lowest validation loss so far.
            if save_best_separately and losses_validation[-1] <= min(losses_validation):
                save_model(
                    f"{filepath_model[:-4]}[best]{filepath_model[-4:]}",
                    epoch = epoch,
                    model_state_dict = model.state_dict(),
                    optimizer_state_dict = optimizer.state_dict(),
                    learning_rate = optimizer.param_groups[0]['lr'],
                    losses_training = losses_training,
                    losses_validation = losses_validation,
                )

    # Show the loss history.
    if show_loss:
        checkpoint = load_model(filepath_model)
        losses_training = checkpoint.get('losses_training', [])
        losses_validation = checkpoint.get('losses_validation', [])
        plot_loss(losses_training, losses_validation)

    # Load the best model.
    checkpoint = load_model(f"{filepath_model[:-4]}[best]{filepath_model[-4:]}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if test:
        outputs, labels, inputs = test_model(
            model = model,
            loss_function = loss_function,
            test_dataloader = test_dataloader,
        )

        results = evaluate_results(dataset.inverse_transform(outputs.numpy()), dataset.inverse_transform(labels.numpy()))
        # results = evaluate_results(outputs.numpy(), labels.numpy())
        output_range = dataset.outputs.max() - dataset.outputs.min()
        print(f"MAE (normalized): {results['MAE'] / output_range}")
        print(f"MSE (normalized): {results['MSE'] / (output_range)}")
        print(f"RMSE (normalized): {results['RMSE'] / output_range}")

        # Show a parity plot.
        if show_parity:
            plot_parity(outputs, labels)

        # Show a comparison plot of the results with labels.
        if show_results:
            for index in random.sample(range(len(test_dataset)), k=5):
                plot_comparison(
                    outputs[index],
                    labels[index],
                    str(test_dataset[index][2]),
                )


if __name__ == '__main__':
    # Load pretrained model and copy encoder to new model.
    # checkpoint = load_model('Checkpoints/TemperatureNet.pth')
    # weights = checkpoint['model_state_dict']
    # model = ThermalNet(32, 10)
    # model.load_encoder(weights)

    dataset = FinDataset('temperature')

    for x in ('1.50', '2.00', '2.50', '3.00', '3.50', '4.00'):
        filename_model = f"exp_{x}.pth"
        x = float(x)

        def transform(data):
            """Exponentiation."""
            data = data - dataset.outputs.min().item()
            data = data / (dataset.outputs.max().item() - dataset.outputs.min().item())
            data = data ** x
            data = data * 78
            return data

        # def transform(data):
        #     """Logarithm."""
        #     data = data - dataset.outputs.min().item()
        #     data = data / (dataset.outputs.max().item() - dataset.outputs.min().item())
        #     data = data + x
        #     data = np.log(data)
        #     data = data - np.log(x)
        #     data = data / (np.log(1 + x) - np.log(x))
        #     data = data * 78
        #     return data

        def inverse_transform(data):
            """Exponentiation."""
            data = data / 78
            data = data ** (1/x)
            data = data * (dataset.outputs.max().item() - dataset.outputs.min().item())
            data = data + dataset.outputs.min().item()
            return data

        # def inverse_transform(data):
        #     """Logarithm."""
        #     data = data / 78
        #     data = data * (np.log(1 + x) - np.log(x))
        #     data = data + np.log(x)
        #     data = np.exp(data)
        #     data = data - x
        #     data = data * (dataset.outputs.max().item() - dataset.outputs.min().item())
        #     data = data + dataset.outputs.min().item()
        #     return data

        main(
            epoch_count = 10,
            learning_rate = 10**(-3.5),
            batch_sizes = (8, 32, 32),
            dataset_split = (0.8, 0.1, 0.1),

            train = True,
            test = True,
            train_existing = not True,
            save_model_every = 10,
            save_best_separately = True,

            dataset = FinDataset('temperature', transforms=(transform, inverse_transform)),
            model = ThermalNet(32, 10),
            filename_model = filename_model,
            loss_function = MSELoss(),
            Optimizer = torch.optim.Adam,
            scheduler = None,

            show_loss = not True,
            show_parity = not True,
            show_results = not True,
        )
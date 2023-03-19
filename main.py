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


def train_model(
    epoch_count: int, checkpoint: dict, filepath_model: str, save_model_every: int,
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    scheduler = None,
) -> torch.nn.Module:

    # Load data from the checkpoint.
    epoch = checkpoint.get('epoch', 0) + 1
    losses_training = checkpoint.get('losses_training', [])
    losses_validation = checkpoint.get('losses_validation', [])

    epochs = range(epoch, epoch+epoch_count)

    for epoch in epochs:
        print(f"\nEpoch {epoch}/{epochs[-1]} ({time.strftime('%I:%M %p')})")
        time_start = time.time()
        
        # Train on the training dataset.
        model.train(True)
        loss = 0
        for batch, (input_data, label_data) in enumerate(train_dataloader, 1):
            # Predict an output from the model with the given input.
            output_data = model(input_data)
            # Calculate the loss.
            loss_current = loss_function(output_data, label_data)
            # Update the cumulative loss.
            loss += loss_current.item()

            if loss_current is torch.nan:
                print(f"Stopping due to nan loss.")
                break
            
            # Reset gradients of model parameters.
            optimizer.zero_grad()
            # Calculate gradients.
            loss_current.backward()
            # Adjust model parameters.
            optimizer.step()

            if batch % 10 == 0:
                print(f"Batch {batch}/{len(train_dataloader)}: {loss/batch:,.2e}...", end="\r")

        print()
        loss /= batch
        losses_training.append(loss)
        print(f"Training loss: {loss:,.2e}")

        # Adjust the learning rate if a scheduler is used.
        if scheduler:
            scheduler.step()
            learning_rate = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {learning_rate}")

        # Test on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
        model.train(False)
        loss = 0
        outputs = []
        labels = []
        with torch.no_grad():
            for batch, (input_data, label_data) in enumerate(validate_dataloader, 1):
                output_data = model(input_data)
                loss += loss_function(output_data, label_data.float()).item()

                output_data = output_data.cpu()
                label_data = label_data.cpu()

                outputs.append(output_data)
                labels.append(label_data)

                if batch % 10 == 0:
                    print(f"Batch {batch}/{len(validate_dataloader)}...", end="\r")
 
        print()
        loss /= batch
        losses_validation.append(loss)
        print(f"Validation loss: {loss:,.2e}")

        # # Calculate evaluation metrics on validation results.
        # outputs = torch.cat(outputs, dim=0)
        # labels = torch.cat(labels, dim=0)
        # evaluate_results(outputs.numpy(), labels.numpy())

        # Save the model parameters periodically and in the last iteration of the loop.
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

        # Show the elapsed time during the epoch.
        time_end = time.time()
        duration = time_end - time_start
        if duration >= 60:
            duration_text = f"{duration/60:.1f} minutes"
        else:
            duration_text = f"{duration:.1f} seconds"
        print(f"Finished epoch {epoch} in {duration_text}.")

    # Plot the loss history.
    plt.figure()
    plt.semilogy(range(1, len(losses_training)+1), losses_training, '-', label='Training')
    plt.semilogy(range(1, len(losses_validation)+1), losses_validation, '-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss {loss_function}')
    plt.legend()
    plt.show()

    return model

def test_model(
    model: torch.nn.Module, loss_function: torch.nn.Module, test_dataloader: DataLoader,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    model.train(False)

    loss = 0
    inputs = []
    outputs = []
    labels = []

    with torch.no_grad():
        for batch, (input_data, label_data) in enumerate(test_dataloader, 1):
            output_data = model(input_data)
            loss += loss_function(output_data, label_data.float()).item()

            input_data = input_data.cpu().detach()
            output_data = output_data.cpu().detach()
            label_data = label_data.cpu()

            inputs.append(input_data)
            labels.append(label_data)
            outputs.append(output_data)
            
            if batch % 1 == 0:
                print(f"Batch {batch}/{len(test_dataloader)}...", end="\r")

    print()
    loss /= batch
    print(f"Testing loss: {loss:,.2e}")

    # Concatenate testing results from all batches into a single array.
    inputs = torch.cat(inputs, dim=0)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, labels, inputs

def evaluate_results(outputs: np.ndarray, labels: np.ndarray, queue=None, info_gui: dict=None):
    """Calculate and return evaluation metrics."""

    results = {
        "MAE": mae(outputs, labels),
        "MSE": mse(outputs, labels),
        "MRE": mre(outputs, labels),
    }
    for metric, value in results.items():
        print(f"{metric}: {value:,.3f}")

    # Show a parity plot.
    plot_parity(outputs, labels)

    return results

def main(
    epoch_count: int, learning_rate: float, batch_sizes: Tuple[int, int, int], dataset_split: Tuple[float, float, float],
    train: bool, test: bool, evaluate: bool, train_existing: bool, save_model_every: int,
    model: torch.nn.Module, filename_model: str, dataset: Dataset, loss_function: torch.nn.Module, Optimizer: torch.optim.Optimizer, scheduler=None
):
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

    `model`: The network, as an instance of a Module subclass.
    `filename_model`: Name of the .pth file to load and save to during training.
    `dataset`: The dataset, as an instance of a Dataset subclass.
    `loss_function`: The loss function, as an instance of a Module subclass.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `scheduler`: A learning rate scheduler.
    """

    filepath_model = os.path.join(FOLDER_CHECKPOINTS, filename_model)

    # Load the previously saved checkpoint.
    if (test and not train) or (train and train_existing):
        checkpoint = load_model(filepath_model)
        # Load the last learning rate used.
        learning_rate = checkpoint.get('learning_rate', learning_rate)
    else:
        checkpoint = {}

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
        model = train_model(
            epoch_count = epoch_count,
            checkpoint = checkpoint,
            filepath_model = filepath_model,
            save_model_every = save_model_every,
            model = model,
            optimizer = optimizer,
            loss_function = loss_function,
            train_dataloader = train_dataloader,
            validate_dataloader = validate_dataloader,
            scheduler = scheduler,
            )

    if test:
        outputs, labels, inputs = test_model(
            model = model,
            loss_function = loss_function,
            test_dataloader = test_dataloader,
        )

        if evaluate:
            results = evaluate_results(outputs.numpy(), labels.numpy())

            for i in range(3):
                index = random.randint(0, len(test_dataset)-1)
                plot_comparison(outputs[index], labels[index])

    return results


if __name__ == '__main__':
    main(
        epoch_count = 500,
        learning_rate = 1e-4,
        batch_sizes = (8, 8, 8),
        dataset_split = (0.8, 0.1, 0.1),

        train = True,
        test = True,
        evaluate = True,
        train_existing = True,
        save_model_every = 10,

        model = ResNet(),
        filename_model = 'temperature.pth',
        dataset = TemperatureDataset(),
        loss_function = MSELoss(),
        Optimizer = torch.optim.Adam,
        scheduler = None,
    )
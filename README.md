# Fin
2D transient thermal response predictions in fins using convolutional neural networks.

Variable parameters:
* Fin thickness
* Fin taper ratio
* Temperature boundary condition
* Convection coefficient

Input images (3×40×80):
* Channel 1: Fin shape
* Channel 2: Convection coefficient
* Channel 3: Temperature boundary condition

Output images (10×8×32 or 1×8×32):
* 2D temperature distribution over 10 instances in time (10 channels), or
* 2D thermal gradient distribution over 10 instances in time (10 channels), or
* 2D thermal stress distribution at final time only (1 channel)


## Creating the Dataset
Ansys (Mechanical APDL) is used to run simulations and generate response data, which are then processed in Python.

Define the simulation parameters in the script `ansys_script.lgw`. Constants are defined near the top of the file, and simulation parameters are defined in the for loops. The script loops over each combination of simulation parameters, and it simulates responses and saves the response data to text files. Refer to [this reference](https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_cmd/Hlp_C_CmdTOC.html) on Ansys APDL commands.

Open Ansys and run the script `ansys_script.lgw`.
* File > Change Directory... > [Select the folder where the script exists]
* File > Read Input from... > [Select the script]
* To maximize performance, close the File Explorer window of the folder in which the program is running while simulations are running.

Each line in a text file contains the response value at a different node (not an element) along with the X and Y coordinates of that node. The coordinates of the node are used later to insert each response value into the proper location in a matrix. For transient simulations, a separate text file is written for each time step rather than having a single text file be written for all time steps. For static simulations, one text file is written for each simulation. Because of this difference in format between transient and static data, move all transient response files into one folder, and move all static response files into another folder.

Run `preprocessing.py` to process and cache the text files for easier loading when training models. Specifically, this code reads the response data contained in the text files and converts them into tensors and saves them as [`.pickle` files](https://docs.python.org/3/library/pickle.html). Caching the data in this way avoids having to read and convert the text files directly every time a model is trained, which takes longer.

Define which `.pickle` files should be used for which response data in the `FinDataset` class in `datasets.py`.


## Training and Testing the Model
Call the `main()` function in `main.py` to train and test the model. The model architecture is defined in `models.py`. Results and model predictions are plotted after testing.

Set `train` to `True` to train the model, and set `test` to `True` to test the model. When training a model from scratch, `train_existing` must be set to `False`. Manually specify training hyperparameters, such as number of epochs, learning rate, batch size, etc.

During training, the best model weights (determined by lowest validation loss) are tracked and saved separately.
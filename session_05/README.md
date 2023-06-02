# Session 5

This repository contains files for training and testing a CNN model. The folder structure is as follows:

- `model.py`: Contains the implementation of the CNN model covered in **Session 5**.
- `utils.py`: Contains utility functions for printing model summary, visualizing datasets, training, and testing the model.
- `S5.ipynb`: A Python notebook demonstrating the usage of the model and utilities for training and testing.

## Usage
```python
# Import the model
from model import Session5Model

# Import the functions to get the training and testing dataloaders
# and fit the model
from utils import fit_model, get_dataloaders

# Create the dataloaders
train_loader, test_loader = get_dataloaders(batch_size=512)

# Train the model for any number of epochs
fit_model(model, train_loader, test_loader, num_epochs=20)

```

## Model

`model.py` contains the the CNN model covered in **Session 5**

```
============================================================================================================================================
Layer (type (var_name))                  Kernel Shape              Input Shape               Output Shape              Param #
============================================================================================================================================
Session5Model (Session5Model)            --                        [1, 1, 28, 28]            [1, 10]                   --
├─Conv2d (conv1)                         [3, 3]                    [1, 1, 28, 28]            [1, 32, 26, 26]           288
├─Conv2d (conv2)                         [3, 3]                    [1, 32, 26, 26]           [1, 64, 24, 24]           18,432
├─Conv2d (conv3)                         [3, 3]                    [1, 64, 12, 12]           [1, 128, 10, 10]          73,728
├─Conv2d (conv4)                         [3, 3]                    [1, 128, 10, 10]          [1, 256, 8, 8]            294,912
├─Linear (fc1)                           --                        [1, 4096]                 [1, 50]                   204,800
├─Linear (fc2)                           --                        [1, 50]                   [1, 10]                   500
============================================================================================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
Total mult-adds (M): 37.26
============================================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 2.37
Estimated Total Size (MB): 3.08
============================================================================================================================================
```

## Utilities

The `utils.py` file contains utility functions that facilitate various tasks related to the model training, testing and visualizing datasets and results, including:

- **get_rows_cols(num: int) -> Tuple[int, int]**: A function to calculate the number of rows and columns for organizing subplots based on the given number.
- **model_summary(model, input_size)**: A function to print the summary of the model, displaying the architecture and number of parameters using the `torchinfo` package.
- **plot_data(data_list: List[List[int | float]], titles: List[str])**: A function to plot multiple data series with corresponding titles using subplots.
- **visualize_data(loader, num_figures: int = 12, label: str = "")**: A function to visualize a batch of images from a data loader, displaying a grid of images with their labels.
- **get_train_transforms()**: A function to return the data augmentation and transformation pipeline for training data.
- **get_test_transforms()**: A function to return the transformation pipeline for test data.
- **get_dataloaders(batch_size: int = 512)**: A function to create data loaders for training and test data. As of now, it only supports the MNIST dataset from torchvision.
- **get_correct_pred_count(pred, targets)**: A function to calculate the number of correct predictions given the model's output predictions and the target labels.
- **Trainer**: A class that encapsulates the training process, including the training loop, loss calculation, and accuracy calculation.
- **Tester**: A class that encapsulates the testing process, including the evaluation loop, loss calculation, and accuracy calculation.
- **fit_model(model, train_loader, test_loader, num_epochs: int = 20)**: A function to train the model using the specified data loaders for a given number of epochs, utilizing the Trainer and Tester classes.
- **collect_results(trainer: Trainer, tester: Tester)**: A function to collect and return the training and testing results (losses and accuracies) from the Trainer and Tester instances.

Please refer to the individual files for more details on their implementation and usage.

## Requirements

The code in this repository requires the following dependencies:

- Python 3.10.x
- torch (version 2.0 or higher)
- torchvision (version 0.15 or higher)
- torchinfo (version 1.8 or higher)
- tqdm (version 4.65 or higher)
- matplotlib (version 3.7 or higher)

To install the required dependencies, you can use pip:

```bash
pip install torch torchvision torchinfo tqdm matplotlib
```
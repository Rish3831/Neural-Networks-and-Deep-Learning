# Project Overview
This project implements a simple neural network in Python to perform regression on a dataset (`ce889_dataCollection.csv`). The model includes:
- **Training** and **Validation** processes to evaluate the model's Root Mean Square Error (RMSE) over epochs.
- **Weight adjustments** for layers between inputs, hidden neurons, and outputs.
- **RMSE plots** for visualizing model error trends over training epochs.

# Prerequisites
You will need the following libraries:
```python
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
```

## 1. Data Loading and Preprocessing
The data file ce889_dataCollection.csv is loaded, and the dataset is prepared by:

Removing duplicates and NaN values.
Normalizing each feature to bring all values between 0 and 1.
Splitting the data into three sets which are 70% Training set,15% Validation set and 15% Test set.

## 2. Neural Network Parameters
The following parameters can be set by the user:

Number of neurons in the hidden layer.

Lambda (λ) for activation function scaling.

Number of epochs for training.

Eta (η), the learning rate for weight updates.

These inputs initialize the model with a user-defined structure and training configurations.

## 3. Model Initialization
   
Random weights are assigned:

Between input and hidden layers (w1).

Between hidden and output layers (w2).





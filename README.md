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

**Number of neurons** in the hidden layer.

**Lambda (λ)** for activation function scaling.

**Number of epochs** for training.

**Eta (η)**, the learning rate for weight updates.

These inputs initialize the model with a user-defined structure and training configurations.

## 3. Model Initialization
   
Random weights are assigned:

Between input and hidden layers (w1).

Between hidden and output layers (w2).

## 4. Forward Pass Calculations
For each sample in the data:

Weighted sums are computed at each layer. 

**Activation functions:** A sigmoid function scales the outputs at both hidden and output layers.

## 5. Backpropagation and Weight Updates
The error for each output is calculated and used for weight adjustments:

**Output Layer Gradients:** Calculated to update weights from hidden to output layer.

**Hidden Layer Gradients:** Calculated to update weights from input to hidden layer.

The weights are updated using the gradient descent method with the specified learning rate (η).

## 6. RMSE Calculation
After each epoch, the model calculates the RMSE for both the training and validation sets. These values are stored for later visualization.

## 7. Testing and Final RMSE Calculation
Using the final weights, the model evaluates performance on the test set, printing the final RMSE value.

## 8. Plotting RMSE
The RMSE values for the training and validation sets are plotted over the epochs to observe model convergence.

## Results and Observations
The output weights (w2) and input weights (w1) are printed, representing the final learned weights.
The Test RMSE reflects the model's final error on unseen data, indicating its predictive performance.
**RMSE Trend:** The plot of RMSE over epochs helps visualize if the model is learning effectively and if overfitting occurs on the validation set.

## Conclusion
This project demonstrates a basic feedforward neural network with gradient descent, RMSE evaluation, and weight updates. The resulting RMSE values give insight into model performance, with RMSE convergence indicating effective training.




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

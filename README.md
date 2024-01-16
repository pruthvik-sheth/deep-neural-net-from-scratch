# Neural Networks from Scratch

Welcome to the Neural Networks from Scratch project, Deep-net. In this project, we aim to build and implement neural networks from the ground up, without relying on external libraries. By doing so, we gain valuable insights into the inner workings of neural networks and the mathematics involved in training them.

## Project Structure

The project is organized as follows:

Deep-net/
|
|__Layers/
| |__ActivationLayer.py
| |__DeepLayer.py
| |__activations.py
| |__losses.py
|
|__Network/
| |__Network.py
|
|__data_loader.py
|__main.py
|__README.md



### 1. `ActivationLayer.py`

This module contains the implementation of the `ActivationLayer` class, responsible for applying activation functions to the output of a layer.

```python
# Sample code for ActivationLayer.py

import numpy as np

class ActivationLayer:
    def __init__(self, activation, activation_prime, dropout=False, keep_prob=0.6):
        # ...
    
    def forward(self, input_data, predict_dropout_switch=False):
        # ...
    
    def backward(self, dA, learning_rate, regularization, lambd):
        # ...
```
### 2. `DeepLayer.py`

The DeepLayer class represents a single layer of a neural network and includes the weights, biases, and forward propagation logic.

```python
# Sample code for DeepLayer.py

import numpy as np

class DeepLayer:
    def __init__(self, n_inputs, n_neurons) -> None:
        # ...
    
    def forward(self, input_data):
        # ...
    
    def backward(self, dZ, learning_rate, regularization, lambd=0.7):
        # ...
```
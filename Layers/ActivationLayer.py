import numpy as np


class ActivationLayer:

    def __init__(self, activation, activation_prime, dropout = False, keep_prob = 0.6):
        self.name = 'activation_layer'
        self.activation = activation
        self.activation_prime = activation_prime
        self.dropout = dropout
        self.keep_prob = keep_prob

    def forward(self, input_data, predict_dropout_switch = False):
        self.inputs = input_data
        self.outputs = self.activation(self.inputs)

        if self.dropout and not predict_dropout_switch:
            self.drops = np.random.rand(self.inputs.shape[0], self.inputs.shape[1])
            self.drops = (self.drops < self.keep_prob) * 1.0
            self.outputs = self.drops * self.outputs
            self.outputs = self.outputs / self.keep_prob

        return self.outputs

    def backward(self, dA):
        if self.dropout:
            dA = dA * self.drops
            dA = dA * self.keep_prob
        return self.activation_prime(self.inputs) * dA

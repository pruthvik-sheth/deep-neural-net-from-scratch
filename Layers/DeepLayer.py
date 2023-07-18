import cupy as np
np.random.seed(3)


class DeepLayer:

    def __init__(self, n_inputs, n_neurons) -> None:
        # Using He initialization technique
        self.name = 'fc_layer'
        self.weights = np.random.randn(n_neurons, n_inputs) * (np.sqrt(2. / n_inputs))
        self.biases = np.zeros((n_neurons, 1))

    def forward(self, input_data):
        self.inputs = input_data
        self.outputs = np.dot(self.weights, self.inputs) + self.biases
        return self.outputs

    def backward(self, dZ, learning_rate, regularization, lambd = 0.7):
        m = self.inputs.shape[1]
        # Calculating the gradients required
        if regularization:
            dW = (np.dot(dZ, self.inputs.T) + (lambd * self.weights)) / m
        else:
            dW = np.dot(dZ, self.inputs.T) / m
            
        dB = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)

        # Updating weights and biases
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB

        return dA_prev

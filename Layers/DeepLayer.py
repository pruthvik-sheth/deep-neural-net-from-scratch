import cupy as np
np.random.seed(3)


class DeepLayer:

    def __init__(self, n_inputs, n_neurons) -> None:
        # Using He initialization technique
        self.name = 'fc_layer'
        self.weights = np.random.randn(n_neurons, n_inputs) * (np.sqrt(2. / n_inputs))
        self.biases = np.zeros((n_neurons, 1))
        # Initializing velcocity and rms prop for ADAM
        self.Vdw = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        self.Vdb = np.zeros((self.biases.shape[0], self.biases.shape[1]))

        self.Sdw = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        self.Sdb = np.zeros((self.biases.shape[0], self.biases.shape[1]))

    def forward(self, input_data):
        self.inputs = input_data
        self.outputs = np.dot(self.weights, self.inputs) + self.biases
        return self.outputs

    def backward(
        self, 
        dZ,
        learning_rate, 
        regularization, 
        lambd, 
        beta1, 
        beta2, 
        t, 
        epsilon
        ):

        m = self.inputs.shape[1]
        # Calculating the gradients required
        if regularization:
            dW = (np.dot(dZ, self.inputs.T) + (lambd * self.weights)) / m
        else:
            dW = np.dot(dZ, self.inputs.T) / m
        dB = np.sum(dZ, axis = 1, keepdims = True) / m
        dA_prev = np.dot(self.weights.T, dZ)

        # Updating velocity and rms prop for ADAM
        self.Vdw = beta1 * self.Vdw + (1 - beta1) * dW
        self.Vdb = beta1 * self.Vdb + (1 - beta1) * dB

        # Bias correction for velocity and rms prop
        Vdw_corrected = self.Vdw / (1 - (beta1 ** t))
        Vdb_corrected = self.Vdb / (1 - (beta1 ** t))

        self.Sdw = beta2 * self.Sdw + ((1 - beta2) * np.square(dW))
        self.Sdb = beta2 * self.Sdb + ((1 - beta2) * np.square(dB))

        Sdw_corrected = self.Sdw / (1 - (beta2 ** t))
        Sdb_corrected = self.Sdb / (1 - (beta2 ** t))

        # Updating weights and biases (ADAM)
        self.weights -= (learning_rate * Vdw_corrected) / np.sqrt(Sdw_corrected + epsilon)
        self.biases -= (learning_rate * Vdb_corrected) / np.sqrt(Sdb_corrected + epsilon)

        return dA_prev

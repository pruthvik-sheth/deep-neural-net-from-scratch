import cupy as np
import datetime
import tensorflow as tf


class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        m = input_data.shape[1]
        y_prediction = np.zeros((1, m))

        output = input_data
        for layer in self.layers:
            if layer.name == 'activation_layer':
                output = layer.forward(output, predict_dropout_switch = True)
            else:
                output = layer.forward(output)
                

        y_prediction = (output >= 0.5) * 1.0

        return y_prediction

    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate, regularization = False, lambd = 0.7):
        m = x_train.shape[1]
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + current_time + '/train'
        test_log_dir = './logs/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for i in range(epochs):
            output = x_train

            # Forward Propogation
            for layer in self.layers:
                output = layer.forward(output)

            # Calculating Costs
            if regularization:
                frobenius_norm = 0.0
                for layer in self.layers:
                    if layer.name == 'fc_layer':
                        frobenius_norm += np.sum(np.square(layer.weights))
                
                L2_cost_part = frobenius_norm * (lambd / (2 * m))
                cost = self.loss(y_train, output) + L2_cost_part
            else:
                cost = self.loss(y_train, output)
            cupy_cost = cost.get()

            # Backward Propogation
            error = self.loss_prime(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate, regularization, lambd)

            if (i + 1) % 100 == 0:
                print('-------------------------------')
                print('Epoch %d/%d   Error=%f' % (i+1, epochs, cost))

                y_prediction_train = self.predict(x_train)
                y_prediction_test = self.predict(x_test)

                train_accuracy = 100 - \
                    np.mean(np.abs(y_prediction_train - y_train)) * 100
                test_accuracy = 100 - \
                    np.mean(np.abs(y_prediction_test - y_test)) * 100

                cupy_train_accuracy = train_accuracy.get()
                cupy_test_accuracy = test_accuracy.get()

                print(f"Train accuracy: {train_accuracy}")
                print(f"Test accuracy: {test_accuracy}")

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', cupy_cost.item(), step=i + 1)
                    tf.summary.scalar('accuracy', cupy_train_accuracy, step=i + 1)

                with test_summary_writer.as_default():
                    tf.summary.scalar('accuracy', cupy_test_accuracy, step=i + 1)

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

    def generate_batches(self, train_x, train_y, batch_size):
        mini_batches = []
        num_of_images = train_x.shape[1]
        num_of_batches = num_of_images // batch_size
        leftover = num_of_images % batch_size

        permutation = list(np.random.permutation(num_of_images))
        shuffled_x = train_x[:, permutation]
        shuffled_y = train_y[:, permutation].reshape((1, num_of_images))
        
        for i in range(num_of_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            mini_batch_x = shuffled_x[:, start_index:end_index]
            mini_batch_y = shuffled_y[:, start_index:end_index]
            mini_batches.append((mini_batch_x, mini_batch_y))
        
        if leftover > 0:
            last_mini_batch_x = shuffled_x[:, -leftover:]
            last_mini_batch_y = shuffled_y[:, -leftover:]
            mini_batches.append((last_mini_batch_x, last_mini_batch_y))

        return mini_batches

    def fit(
            self, 
            x_train, 
            y_train, 
            x_test, 
            y_test, 
            epochs, 
            learning_rate, 
            regularization = False, 
            lambd = 0.7,
            batch_size = 64
        ):

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + current_time + '_lr_' + str(learning_rate) + '_reg_' + str(regularization) + '_lmd_' + str(lambd) + '/train'
        test_log_dir = './logs/' + current_time + '_lr_' + str(learning_rate) + '_reg_' + str(regularization) + '_lmd_' + str(lambd) + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for i in range(epochs):
            mini_batches = self.generate_batches(x_train, y_train, batch_size)
            
            for mini_batch in mini_batches:
                mini_x_train, mini_y_train = mini_batch

                m = mini_x_train.shape[1]
                # Forward Propogation
                output = mini_x_train
                for layer in self.layers:
                    output = layer.forward(output)

                # Calculating Costs
                if regularization:
                    frobenius_norm = 0.0
                    for layer in self.layers:
                        if layer.name == 'fc_layer':
                            frobenius_norm += np.sum(np.square(layer.weights))
                    
                    L2_cost_part = frobenius_norm * (lambd / (2 * m))
                    cost = self.loss(mini_y_train, output) + L2_cost_part
                else:
                    cost = self.loss(mini_y_train, output)
                cupy_cost = cost.get()

                # Backward Propogation
                error = self.loss_prime(mini_y_train, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate, regularization, lambd)

            if (i + 1) % 100 == 0:
                print('-------------------------------')
                print('Epoch: %d/%d | Error: %f' % (i+1, epochs, cost))

                y_prediction_train = self.predict(x_train)
                y_prediction_test = self.predict(x_test)

                train_accuracy = 100 - np.mean(np.abs(y_prediction_train - y_train)) * 100
                test_accuracy = 100 - np.mean(np.abs(y_prediction_test - y_test)) * 100

                cupy_train_accuracy = train_accuracy.get()
                cupy_test_accuracy = test_accuracy.get()

                print(f"Train accuracy: {train_accuracy}")
                print(f"Test accuracy: {test_accuracy}")

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', cupy_cost.item(), step=i + 1)
                    tf.summary.scalar('accuracy', cupy_train_accuracy, step=i + 1)

                with test_summary_writer.as_default():
                    tf.summary.scalar('accuracy', cupy_test_accuracy, step=i + 1)

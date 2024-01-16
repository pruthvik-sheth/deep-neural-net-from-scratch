import numpy as np


def log_loss(y, y_hat):
    m = y.shape[1]
    return - (np.dot(y, np.log(y_hat).T) + np.dot((1 - y), np.log(1 - y_hat).T)) / m


def log_loss_prime(y, y_hat):
    return ((np.divide((1 - y), (1 - y_hat))) - (np.divide(y, y_hat)))

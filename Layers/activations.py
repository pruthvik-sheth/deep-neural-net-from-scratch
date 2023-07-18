import cupy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def RelU(x):
    return np.maximum(0, x)


def RelU_prime(x):
    return np.where(x <= 0, 0, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

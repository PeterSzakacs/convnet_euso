import numpy as np


def convert_weights_external(weights):
    return np.moveaxis(weights, 0, 1)


def convert_weights_internal(weights):
    return np.moveaxis(weights, 1, 0)

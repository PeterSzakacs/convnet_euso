# -*- coding: utf-8 -*-

# Original MNIST network changed (no batch_normalization and no dropout)

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Building convolutional network
def create(inputShape, learning_rate=None, optimizer=None, loss_fn=None):
    learning_rate = 0.01 if learning_rate is None else learning_rate
    optimizer = 'adam' if optimizer is None else optimizer
    loss_fn = 'categorical_crossentropy' if loss_fn is None else loss_fn

    network = input_data(shape=inputShape['yx'], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    conv1 = max_pool_2d(network, 2)
    network = conv_2d(conv1, 64, 3, activation='relu', regularizer="L2")
    conv2 = max_pool_2d(network, 2)
    fc1 = fully_connected(conv2, 128, activation='tanh')
    fc2 = fully_connected(fc1, 256, activation='tanh')
    fc3 = fully_connected(fc2, 2, activation='softmax')
    network = regression(fc3, optimizer=optimizer, learning_rate=learning_rate, loss=loss_fn, name='target')
    return network, (conv1, conv2), (fc1, fc2, fc3)

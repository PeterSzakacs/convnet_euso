# -*- coding: utf-8 -*-

# github_net3 with a different number of filters in the first conv layer
# (to sum up: layers structure before the first fc layer is modeled exactly
# after the mnist network)

from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Building convolutional network
def create(inputShape, learning_rate=None, optimizer=None, loss_fn=None):
    learning_rate = 0.001 if learning_rate is None else learning_rate
    optimizer = 'adam' if optimizer is None else optimizer
    loss_fn = 'categorical_crossentropy' if loss_fn is None else loss_fn


    network = input_data(shape=inputShape['yx'], name='input')
    network = conv_2d(network, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv1 = max_pool_2d(network, 2)
    network = local_response_normalization(conv1)
    network = conv_2d(network, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv2 = max_pool_2d(network, 2)
    network = local_response_normalization(conv2)
    fc1 = fully_connected(network, 128, activation='relu')
    network = dropout(fc1, 0.5)
    fc2 = fully_connected(network, 50, activation='relu')
    network = dropout(fc2, 0.5)
    fc3 = fully_connected(network, 2, activation='softmax')
    network = regression(fc3, optimizer=optimizer, learning_rate=learning_rate, loss=loss_fn, name='target')
    return network, (conv1, conv2), (fc1, fc2, fc3)

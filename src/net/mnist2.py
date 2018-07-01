# -*- coding: utf-8 -*-

# Original MNIST network changed (fully connected layers, different number of filters in convolutional layers)

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Building convolutional network
def create(inputShape):
    network = input_data(shape=inputShape, name='input')
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    conv1 = max_pool_2d(network, 2)
    network = local_response_normalization(conv1)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    conv2 = max_pool_2d(network, 2)
    network = local_response_normalization(conv2)
    fc1 = fully_connected(network, 256, activation='relu')
    network = dropout(fc1, 0.8)
    fc2 = fully_connected(network, 256, activation='relu')
    network = dropout(fc2, 0.8)
    fc3 = fully_connected(network, 2, activation='softmax')
    network = regression(fc3, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
    return network, (conv1, conv2), (fc1, fc2, fc3)

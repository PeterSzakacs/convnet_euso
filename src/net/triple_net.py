# -*- coding: utf-8 -*-

# first 3-convolutional layer network, named github_net4 because the fc layers are the same as in github_net and variants

from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge

# Building convolutional network
def create(inputShape, learning_rate=None, optimizer=None, loss_fn=None):
    learning_rate = 0.001 if learning_rate is None else learning_rate
    optimizer = 'adam' if optimizer is None else optimizer
    loss_fn = 'categorical_crossentropy' if loss_fn is None else loss_fn

    conv_yx = input_data(shape=inputShape['yx'], name='yx_input')
    conv_yx = conv_2d(conv_yx, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv_yx = max_pool_2d(conv_yx, 2)
    conv_yx = local_response_normalization(conv_yx)
    conv_yx = conv_2d(conv_yx, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv_yx = max_pool_2d(conv_yx, 2)
    conv_yx = local_response_normalization(conv_yx)
    conv_yx = flatten(conv_yx)

    conv_gtux = input_data(shape=inputShape['gtux'], name='gtux_input')
    conv_gtux = conv_2d(conv_gtux, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv_gtux = max_pool_2d(conv_gtux, 2)
    conv_gtux = local_response_normalization(conv_gtux)
    conv_gtux = conv_2d(conv_gtux, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv_gtux = max_pool_2d(conv_gtux, 2)
    conv_gtux = local_response_normalization(conv_gtux)
    conv_gtux = flatten(conv_gtux)

    conv_gtuy = input_data(shape=inputShape['gtuy'], name='gtuy_input')
    conv_gtuy = conv_2d(conv_gtuy, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv_gtuy = max_pool_2d(conv_gtuy, 2)
    conv_gtuy = local_response_normalization(conv_gtuy)
    conv_gtuy = conv_2d(conv_gtuy, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv_gtuy = max_pool_2d(conv_gtuy, 2)
    conv_gtuy = local_response_normalization(conv_gtuy)
    conv_gtuy = flatten(conv_gtuy)

    network = merge((conv_yx, conv_gtux, conv_gtuy), 'concat')
    fc1 = fully_connected(network, 128, activation='relu')
    network = dropout(fc1, 0.5)
    fc2 = fully_connected(network, 50, activation='relu')
    network = dropout(fc2, 0.5)
    fc3 = fully_connected(network, 2, activation='softmax')
    network = regression(fc3, optimizer=optimizer, learning_rate=learning_rate, loss=loss_fn, name='target')
    return network, (), (fc1, fc2, fc3)

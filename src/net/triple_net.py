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

    conv_xy = input_data(shape=inputShape[0], name='xy_input')
    conv_xy = conv_2d(conv_xy, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv_xy = max_pool_2d(conv_xy, 2)
    conv_xy = local_response_normalization(conv_xy)
    conv_xy = conv_2d(conv_xy, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv_xy = max_pool_2d(conv_xy, 2)
    conv_xy = local_response_normalization(conv_xy)

    conv_xgtu = input_data(shape=inputShape[1], name='xgtu_input')
    conv_xgtu = conv_2d(conv_xgtu, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv_xgtu = max_pool_2d(conv_xgtu, 2)
    conv_xgtu = local_response_normalization(conv_xgtu)
    conv_xgtu = conv_2d(conv_xgtu, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv_xgtu = max_pool_2d(conv_xgtu, 2)
    conv_xgtu = local_response_normalization(conv_xgtu)

    conv_ygtu = input_data(shape=inputShape[2], name='ygtu_input')
    conv_ygtu = conv_2d(conv_ygtu, 32, 3, strides=1, activation='relu', regularizer='L2')
    conv_ygtu = max_pool_2d(conv_ygtu, 2)
    conv_ygtu = local_response_normalization(conv_ygtu)
    conv_ygtu = conv_2d(conv_ygtu, 64, 3, strides=1, activation='relu', regularizer='L2')
    conv_ygtu = max_pool_2d(conv_ygtu, 2)
    conv_ygtu = local_response_normalization(conv_ygtu)

    network = merge((conv_xy, conv_xgtu, conv_ygtu), 'concat')
    fc1 = fully_connected(network, 128, activation='relu')
    network = dropout(fc1, 0.5)
    fc2 = fully_connected(network, 50, activation='relu')
    network = dropout(fc2, 0.5)
    fc3 = fully_connected(network, 2, activation='softmax')
    network = regression(fc3, optimizer=optimizer, learning_rate=learning_rate, loss=loss_fn, name='target')
    return network, (), (fc1, fc2, fc3)

# -*- coding: utf-8 -*-

# first triple-input convolutional network, derived from github_net3b

from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge

import net.convnet_classes as conv_classes


def create(inputShape, **optsettings):
    network = TripleNet(inputShape, **optsettings)
    return network.output_layer, network.conv_layers, network.fc_layers


def create_model(inputShape, **optsettings):
    network = TripleNet(inputShape, **optsettings)
    return conv_classes.Conv2DNetworkModel(network, **optsettings)


class TripleNet(conv_classes.Conv2DNetwork):

    def __init__(self, inputShape, **optsettings):
        lr = optsettings.get('learning_rate') or 0.001
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        conv, fc = [], []
        conv_yx = input_data(shape=inputShape['yx'], name='yx_input')
        conv_yx = conv_2d(conv_yx, 32, 3, strides=1, activation='relu',
                          regularizer='L2')
        conv.append(conv_yx)
        conv_yx = max_pool_2d(conv_yx, 2)
        conv_yx = local_response_normalization(conv_yx)
        conv_yx = conv_2d(conv_yx, 64, 3, strides=1, activation='relu',
                          regularizer='L2')
        conv.append(conv_yx)
        conv_yx = max_pool_2d(conv_yx, 2)
        conv_yx = local_response_normalization(conv_yx)
        conv_yx = flatten(conv_yx)

        conv_gtux = input_data(shape=inputShape['gtux'], name='gtux_input')
        conv_gtux = conv_2d(conv_gtux, 32, 3, strides=1, activation='relu',
                            regularizer='L2')
        conv.append(conv_gtux)
        conv_gtux = max_pool_2d(conv_gtux, 2)
        conv_gtux = local_response_normalization(conv_gtux)
        conv_gtux = conv_2d(conv_gtux, 64, 3, strides=1, activation='relu',
                            regularizer='L2')
        conv.append(conv_gtux)
        conv_gtux = max_pool_2d(conv_gtux, 2)
        conv_gtux = local_response_normalization(conv_gtux)
        conv_gtux = flatten(conv_gtux)

        conv_gtuy = input_data(shape=inputShape['gtuy'], name='gtuy_input')
        conv_gtuy = conv_2d(conv_gtuy, 32, 3, strides=1, activation='relu',
                            regularizer='L2')
        conv.append(conv_gtuy)
        conv_gtuy = max_pool_2d(conv_gtuy, 2)
        conv_gtuy = local_response_normalization(conv_gtuy)
        conv_gtuy = conv_2d(conv_gtuy, 64, 3, strides=1, activation='relu',
                            regularizer='L2')
        conv.append(conv_gtuy)
        conv_gtuy = max_pool_2d(conv_gtuy, 2)
        conv_gtuy = local_response_normalization(conv_gtuy)
        conv_gtuy = flatten(conv_gtuy)

        network = merge((conv_yx, conv_gtux, conv_gtuy), 'concat')
        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 50, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, name='target', learning_rate=lr,
                             optimizer=optimizer, loss=loss_fn)
        super(TripleNet, self).__init__(conv, fc, network)

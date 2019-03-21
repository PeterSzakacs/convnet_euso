# -*- coding: utf-8 -*-

# github_net with added local_response_normalization layers after max pooling
# and removed dropout and flatten layers before first fc layer

from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import net.convnet_classes as conv_classes


def create(input_shapes, **optsettings):
    network = GithubNet3(input_shapes, **optsettings)
    return network.output_layer, network.conv_layers, network.fc_layers


def create_model(input_shapes, **optsettings):
    network = GithubNet3(input_shapes, **optsettings)
    return conv_classes.Conv2DNetworkModel(network, **optsettings)


class GithubNet3(conv_classes.Conv2DNetwork):

    def __init__(self, input_shapes, input_type='yx', **optsettings):
        lr = optsettings.get('learning_rate') or 0.001
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        hidden, trainable, conv, fc = [], [], [], []
        input_shape = input_shapes[input_type]
        network = input_data(shape=input_shape, name='input')
        inputs = {input_type: network}
        network = reshape(network, [-1, *input_shape, 1])
        network = conv_2d(network, 64, 3, strides=1, activation='relu',
                          regularizer='L2')
        conv.append(network); hidden.append(network); trainable.append(network)
        network = max_pool_2d(network, 2)
        hidden.append(network)
        network = local_response_normalization(network)
        hidden.append(network)
        network = conv_2d(network, 64, 3, strides=1, activation='relu',
                          regularizer='L2')
        conv.append(network); hidden.append(network); trainable.append(network)
        network = max_pool_2d(network, 2)
        hidden.append(network)
        network = local_response_normalization(network)
        hidden.append(network)
        network = fully_connected(network, 128, activation='relu')
        fc.append(network); hidden.append(network); trainable.append(network)
        network = dropout(network, 0.5)
        network = fully_connected(network, 50, activation='relu')
        fc.append(network); hidden.append(network); trainable.append(network)
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        fc.append(network); trainable.append(network)
        network = regression(network, name='target', learning_rate=lr,
                             optimizer=optimizer, loss=loss_fn)
        layers = {'hidden': hidden, 'trainable': trainable,
                  'conv2d': conv, 'fc': fc}
        super(self.__class__, self).__init__(inputs, network, layers)

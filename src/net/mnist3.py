# -*- coding: utf-8 -*-

# Original MNIST network changed (no normalization and dropout layers)

from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import net.convnet_classes as conv_classes


def create(inputShape, **optsettings):
    network = MNISTNet3(inputShape, **optsettings)
    return network.output_layer, network.conv_layers, network.fc_layers


def create_model(inputShape, **optsettings):
    network = MNISTNet3(inputShape, **optsettings)
    return conv_classes.Conv2DNetworkModel(network, **optsettings)


class MNISTNet3(conv_classes.Conv2DNetwork):

    def __init__(self, inputShape, **optsettings):
        lr = optsettings.get('learning_rate') or 0.01
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        hidden, trainable, conv, fc = [], [], [], []
        network = input_data(shape=inputShape['yx'], name='input')
        inputs = {'yx': network}
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        conv.append(network)
        network = max_pool_2d(network, 2)
        hidden.append(network)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        conv.append(network)
        network = max_pool_2d(network, 2)
        hidden.append(network)
        network = fully_connected(network, 128, activation='tanh')
        fc.append(network); hidden.append(network); trainable.append(network)
        network = fully_connected(network, 256, activation='tanh')
        fc.append(network); hidden.append(network); trainable.append(network)
        network = fully_connected(network, 2, activation='softmax')
        fc.append(network); trainable.append(network)
        network = regression(network, name='target', learning_rate=lr,
                             optimizer=optimizer, loss=loss_fn)
        layers = {'hidden': hidden, 'trainable': trainable,
                  'conv2d': conv, 'fc': fc}
        super(self.__class__, self).__init__(inputs, network, layers)

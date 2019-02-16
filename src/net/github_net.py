# -*- coding: utf-8 -*-

# based on: https://github.com/nlinc1905/Particle-Identification-Neural-Net
# (conv_classifier_2.py)

from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import net.convnet_classes as conv_classes


def create(inputShape, **optsettings):
    network = GithubNet(inputShape, **optsettings)
    return network.output_layer, network.conv_layers, network.fc_layers


def create_model(inputShape, **optsettings):
    network = GithubNet(inputShape, **optsettings)
    return conv_classes.Conv2DNetworkModel(network, **optsettings)


class GithubNet(conv_classes.Conv2DNetwork):

    def __init__(self, inputShape, **optsettings):
        lr = optsettings.get('learning_rate') or 0.001
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        conv, fc = [], []
        network = input_data(shape=inputShape['yx'], name='input')
        network = conv_2d(network, 64, 3, strides=3, activation='relu',
                          padding="valid")
        conv.append(network)
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, strides=3, activation='relu',
                          padding="valid")
        conv.append(network)
        network = max_pool_2d(network, 2)
        network = dropout(network, 0.3)
        network = flatten(network)
        network = fully_connected(network, 128, activation='relu')
        fc.append(network)
        network = dropout(network, 0.5)
        network = fully_connected(network, 50, activation='relu')
        fc.append(network)
        network = fully_connected(network, 2, activation='softmax')
        fc.append(network)
        network = regression(network, name='target', learning_rate=lr,
                             optimizer=optimizer, loss=loss_fn)
        super(GithubNet, self).__init__(conv, fc, network)

# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
#import net.mnist as mnistNet
#import net.mnist_variant1 as mnistNet
import net.mnist_variant2 as mnistNet

# Data loading and preprocessing

baseDir = "../res/"
#X = np.load(baseDir + "/visible_events.npy")\
X = np.load(baseDir + "simu_testing_x.npy")\
    .reshape([-1, 48, 48, 1])\
    .astype(np.uint32)
#Y = np.tile([1,0], [int(X.shape[0]/2)])\
Y = np.load(baseDir + "simu_testing_y.npy")\
    .reshape([-1, 1])

# Building convolutional network
network=mnistNet.create(inputShape=[None, 48, 48, 1])

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=11,
           validation_set=0.1,
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

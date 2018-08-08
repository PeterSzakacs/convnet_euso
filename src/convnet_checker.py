# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
import os
import importlib

import tflearn
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

import utils.cmdint.cmd_interface_checker as cmd


# command line argument parsing
cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])
print(args)

module_name = args.network
modelfile = args.model
logdir = args.logdir
if not os.path.exists(logdir):
    os.mkdir(logdir)

# data loading and preprocessing
X_all = np.load(args.infile)
Y_all = np.load(args.targetfile).reshape([-1, 2]).astype(np.uint8)

n, w, h = X_all.shape[0], X_all.shape[1], X_all.shape[2]

#select the first 100 frames as a "quasi-validation set"
#test_n = round(0.1*n)

X_test = X_all[:100]
Y_test = Y_all[:100]

net_mod = importlib.import_module("net." + module_name)
network, conv_layers, fc_layers = net_mod.create(inputShape=[None, w, h, 1])
model = tflearn.DNN(network, tensorboard_verbose = 0)
model.load(modelfile)
for idx in range(len(X_test)):
    prediction = model.predict(X_test[idx].reshape(1, w, h, 1))[0]
    prediction = np.round(prediction).astype(np.uint8)
    out = 'noise' if np.array_equal(prediction, [0, 1]) else 'shower'
    targ = 'noise' if np.array_equal(Y_test[idx], [0, 1]) else 'shower'
    print("Prediction {}: output: {}, target: {}".format(idx, prediction, Y_test[idx]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(X_test[idx])
    plt.colorbar(im)
    plt.savefig(os.path.join(args.logdir, 'prediction-{}-o-{}-t-{}'.format(str(idx).zfill(3), out, targ)))
    plt.close()



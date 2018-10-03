# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
import os
import importlib
from datetime import datetime as dt

import tflearn
import tensorflow as tf
import numpy as np

import cmdint.cmd_interface_trainer as cmd
import visualization.network.filters_visualizer as filtersViz
import visualization.network.conv_layer_visualizer as convViz
import visualization.network.fc_layer_visualizer as fcViz


# command line argument parsing
cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])
print(args)

lr, epochs = args.learning_rate, args.epochs
optimizer, loss = args.optimizer, args.loss
network_module_names = args.networks
save = args.save

logdir = args.logdir
if not os.path.exists(logdir):
    os.mkdir(logdir)


# New subdirectory for logging the results of the current script invocation
current_run_dir = os.path.join(logdir, dt.now().strftime('%Y-%m-%d_%H:%M:%S'))
if not os.path.exists(current_run_dir):
    os.mkdir(current_run_dir)


# data loading and preprocessing
Xs, input_shapes = [], []
for filename in args.infiles:
    if filename != 'None':
        X = np.load(filename)
        n, h, w = X.shape[0:3]
        X = X.reshape(n, h, w, 1).astype(np.uint32)
        Xs.append(X)        
        input_shapes.append([None, h, w, 1])
Y_all = np.load(args.targetfile).reshape([-1, 2])

#select the first 10% of frames into the validation (actually test) set
test_n = round(0.1*n)

X_trains, X_tests = [], []
for X in Xs:
    X_trains.append(X[test_n:])
    X_tests.append(X[:test_n])
Y_train = Y_all[test_n:]
Y_test = Y_all[:test_n]

if len(Xs) == 1:
    X_trains = X_trains[0]
    X_tests = X_tests[0]

# uncomment callbacks-related code if you want to see visually in generated images at each output what the outputs 
# of the convolutional (well, technically, max-pooling, and only first 10 filters) and FC layers are after each epoch

for module_name in network_module_names:
    module_dir = os.path.join(current_run_dir, module_name)
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    net_mod = importlib.import_module("net." + module_name)
    graph = tf.Graph()
    with graph.as_default():
        network, conv_layers, fc_layers = net_mod.create(inputShape=input_shapes, learning_rate = lr, 
                                                         optimizer = optimizer, loss_fn = loss)
        model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = module_dir)
#        visual_output_dir = os.path.join(module_dir, 'viz')
#        if not os.path.exists(visual_output_dir):
#            os.mkdir(visual_output_dir)
#        weights_output_dir = os.path.join(visual_output_dir, 'weights')
#        if not os.path.exists(weights_output_dir):
#            os.mkdir(os.path.join(visual_output_dir, 'weights'))
#        weights_callback = filtersViz.VisualizerCallback(model, conv_layers, weights_output_dir)
#        convCallback = convViz.VisualizerCallback(model, X[0:100], visual_ouptut_dir, conv_layers)
#        fcCallback = fcViz.VisualizerCallback(model, X[0:100], visual_output_dir, fc_layers)
        model.fit(X_trains, Y_train, n_epoch=epochs, validation_set=(X_tests, Y_test),
                  snapshot_step=100, show_metric=True, run_id=module_name)#, callbacks=[weights_callback])
        if save:
            model.save(os.path.join(module_dir, "{}.tflearn".format(module_name)))

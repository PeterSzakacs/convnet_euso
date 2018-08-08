# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
import os
import importlib
from datetime import datetime as dt

import tflearn
import tensorflow as tf
import numpy as np

import utils.cmd_interface_convnet as cmd
import utils.filters_visualizer as filtersViz
import utils.conv_layer_visualizer as convViz
import utils.fc_layer_visualizer as fcViz


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
Xfile = args.infile
Yfile = args.targetfile
X = np.load(Xfile)
Y = np.load(Yfile).reshape([-1, 2])

w, h = X.shape[1], X.shape[2]
X = X.reshape([-1, w, h, 1]).astype(np.uint32)


# uncomment callbacks-related code if you want to see visually in generated images at each output what the outputs 
# of the convolutional (well, technically, max-pooling, and only first 10 filters) and FC layers are after each epoch
#
# NOTE: however, the network to be trained next, if there is one and it also has the relevant callback lines of code 
# uncommented, will overwrite the images generated by the previous network (TODO: might want to implement specifying 
# an output directory when creating these callbacks to avoid this side-effect)

for module_name in network_module_names:
    module_dir = os.path.join(current_run_dir, module_name)
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    net_mod = importlib.import_module("net." + module_name)
    graph = tf.Graph()
    with graph.as_default():
        network, conv_layers, fc_layers = net_mod.create(inputShape=[None, w, h, 1], learning_rate = lr, 
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
        model.fit({'input': X}, {'target': Y}, n_epoch=epochs, validation_set=0.1,
                  snapshot_step=100, show_metric=True, run_id=module_name)#, callbacks=[weights_callback])
        if save:
            model.save(os.path.join(module_dir, "{}.tflearn".format(module_name)))

# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import importlib
from datetime import datetime as dt

import tflearn
import tensorflow as tf
import numpy as np

import utils.dataset_utils as ds
#import visualization.network.filters_visualizer as filtersViz
#import visualization.network.conv_layer_visualizer as convViz
#import visualization.network.fc_layer_visualizer as fcViz

class convnet_trainer:

    def __init__(self, logdir, input_shapes, n_epochs=11, save=False,
                 optimizer=None, loss_fn=None, learning_rate=None):
        self.logdir = logdir
        self.input_shapes = input_shapes
        self.n_epochs = n_epochs
        self.save = save
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.save = save

    @property
    def logdir(self):
        return self._logdir

    @logdir.setter
    def logdir(self, value):
        if not os.path.exists(value):
            raise ValueError('Logging directory {} does not exist'.format(
                             value))
        self._logdir = value

    def train_networks(self, network_module_names, data, targets, eval_set=0.1,
                       log_verbosity=0):
        # New subdirectory for logging the results of the current training
        current_run_dir = os.path.join(self.logdir, dt.now().strftime(
                                       '%Y-%m-%d_%H:%M:%S'))
        os.mkdir(current_run_dir)
        for module_name in network_module_names:
            module_dir = os.path.join(current_run_dir, module_name)
            os.mkdir(module_dir)
            graph = tf.Graph()
            with graph.as_default():
                network_module = importlib.import_module(module_name)
                network, conv_layers, fc_layers = network_module.create(
                    inputShape=self.input_shapes, learning_rate=self.learning_rate,
                    optimizer=self.optimizer, loss_fn=self.loss_fn
                )
                model = tflearn.DNN(network, tensorboard_verbose=log_verbosity,
                                    tensorboard_dir=module_dir)
                model.fit(data, targets, n_epoch=self.n_epochs,
                        validation_set=eval_set, snapshot_step=100,
                        show_metric=True, run_id=module_name)
                if save:
                    model_file = os.path.join(module_dir, "{}.tflearn".format(
                        module_name
                    ))
                    model.save(model_file)


if __name__ == '__main__':
    import cmdint.cmd_interface_trainer as cmd
    import sys

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

    data, targets = ds.load_dataset(args.infiles, args.targetfile)
    data = tuple(dh.reshape([*dh.shape, 1]) for dh in data)
    eval_data, eval_targets = ds.get_evalutation_set(data, targets)

    # tflearn does not seem to like single element sequences,
    # (the single element within is the actual input data).
    if (len(data) == 1):
        data = data[0]
        eval_data = eval_data[0]

    input_shapes = args.input_shapes
    args.networks = tuple('net.' + net_name for net_name in args.networks)

    trainer = convnet_trainer(logdir, input_shapes, n_epochs=epochs,
                              optimizer=optimizer, loss_fn=loss,
                              learning_rate=lr)
    trainer.train_networks(args.networks, data, targets, eval_set=(eval_data, eval_targets))

# uncomment callbacks-related code if you want to see visually in generated images at each output what the outputs
# of the convolutional (well, technically, max-pooling, and only first 10 filters) and FC layers are after each epoch

# for module_name in network_module_names:
#     module_dir = os.path.join(current_run_dir, module_name)
#     if not os.path.exists(module_dir):
#         os.mkdir(module_dir)
#     net_mod = importlib.import_module("net." + module_name)
#     graph = tf.Graph()
#     with graph.as_default():
#         network, conv_layers, fc_layers = net_mod.create(inputShape=input_shapes, learning_rate = lr,
#                                                          optimizer = optimizer, loss_fn = loss)
#         model = tflearn.DNN(network, tensorboard_verbose = 0, tensorboard_dir = module_dir)
#        visual_output_dir = os.path.join(module_dir, 'viz')
#        if not os.path.exists(visual_output_dir):
#            os.mkdir(visual_output_dir)
#        weights_output_dir = os.path.join(visual_output_dir, 'weights')
#        if not os.path.exists(weights_output_dir):
#            os.mkdir(os.path.join(visual_output_dir, 'weights'))
#        weights_callback = filtersViz.VisualizerCallback(model, conv_layers, weights_output_dir)
#        convCallback = convViz.VisualizerCallback(model, X[0:100], visual_ouptut_dir, conv_layers)
#        fcCallback = fcViz.VisualizerCallback(model, X[0:100], visual_output_dir, fc_layers)
        # model.fit(X_trains, Y_train, n_epoch=epochs, validation_set=(X_tests, Y_test),
        #           snapshot_step=100, show_metric=True, run_id=module_name)#, callbacks=[weights_callback])
        # if save:
        #     model.save(os.path.join(module_dir, "{}.tflearn".format(module_name)))
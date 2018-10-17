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

    def __init__(self, logdir, n_epochs=11, save=False,
                 optimizer=None, loss_fn=None, learning_rate=None):
        current_time = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.current_run_dir = os.path.join(logdir, current_time)
        os.mkdir(self.current_run_dir)
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

    def _reshape_data(self, dataset, eval_set_fraction=0.1, eval_set_num=None):
        train_data, train_targets, test_data, test_targets = ds.get_train_and_test_sets(
            dataset, test_fraction=eval_set_fraction, test_num_items=eval_set_num
        )
        item_shapes = dataset.item_shapes
        data_train, data_test = [], []
        for key in ds.ALL_ITEM_TYPES:
            if train_data.get(key, None) is not None:
                data_train.append(train_data[key].reshape(-1, *item_shapes[key], 1))
                data_test.append(test_data[key].reshape(-1, *item_shapes[key], 1))
        # tflearn does not seem to like single element sequences,
        # (the single element within is the actual input data).
        if len(data_train) == 1:
            data_train = data_train[0]
            data_test = data_test[0]
        return data_train, train_targets, data_test, test_targets 


    def train_network(self, network_module_name, dataset, log_verbosity=0, 
                      eval_set_fraction=0.1, eval_set_num=None):
        # New subdirectory for logging the results of the current training
        module_dir = os.path.join(self.current_run_dir, network_module_name)
        os.mkdir(module_dir)
        input_shapes = dataset.item_shapes
        data, targ, test_data, test_targ = self._reshape_data(
            dataset, eval_set_fraction=eval_set_fraction, eval_set_num=eval_set_num
        )
        eval_set = (test_data, test_targ)
        item_shapes = {k:[None, *v, 1] for k, v in dataset.item_shapes.items() if v != None}

        graph = tf.Graph()
        with graph.as_default():
            network_module = importlib.import_module(network_module_name)
            network, conv_layers, fc_layers = network_module.create(
                inputShape=item_shapes, learning_rate=self.learning_rate,
                optimizer=self.optimizer, loss_fn=self.loss_fn
            )
            model = tflearn.DNN(network, tensorboard_verbose=log_verbosity,
                                tensorboard_dir=module_dir)
            model.fit(data, targ, n_epoch=self.n_epochs,
                      validation_set=eval_set, snapshot_step=100,
                      show_metric=True, run_id=network_module_name)
            if save:
                model_file = os.path.join(module_dir, "{}.tflearn".format(
                    network_module_name
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
    save = args.save

    logdir = args.logdir
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    dataset = ds.numpy_dataset.load_dataset(args.srcdir, args.name, 
                                            item_types=args.item_types)

    trainer = convnet_trainer(logdir, n_epochs=epochs, save=save,
                              optimizer=optimizer, loss_fn=loss,
                              learning_rate=lr)
    for network_name in args.networks:
        network_module_name = 'net.' + network_name
        trainer.train_network(network_module_name, dataset, eval_set_fraction=0.1)

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

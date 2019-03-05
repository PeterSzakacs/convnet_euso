# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os

import tensorflow as tf

import dataset.dataset_utils as ds
import net.constants as net_cons
import net.network_utils as netutils
import utils.io_utils as io_utils
#import visualization.network.filters_visualizer as filtersViz
#import visualization.network.conv_layer_visualizer as convViz
#import visualization.network.fc_layer_visualizer as fcViz


if __name__ == '__main__':
    import cmdint.cmd_interface_trainer as cmd
    import sys

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    name, srcdir = args.name, args.srcdir
    item_types = args.item_types
    mode = args.split_mode
    fraction, num_items = args.test_items_fraction, args.test_items_count
    optsettings = {'learning_rate': args.learning_rate, 'loss_fn': args.loss,
                   'optimizer': args.optimizer}

    input_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)
    splitter = netutils.DatasetSplitter(mode, items_fraction=fraction,
                                        num_items=num_items)
    data_dict = splitter.get_data_and_targets(dataset)
    data_dict['train_data'] = netutils.reshape_data_for_convnet(
        data_dict['train_data'])
    data_dict['test_data'] = netutils.reshape_data_for_convnet(
        data_dict['test_data'])
    num_epochs = args.epochs
    trainer = netutils.TfModelTrainer(data_dict, num_epochs=num_epochs,
                                      **optsettings)

    for network_name in args.networks:
        network_module_name = 'net.' + network_name
        run_id = netutils.get_default_run_id(network_module_name)
        tb_dir = os.path.join(net_cons.DEFAULT_TRAIN_LOGDIR, run_id)
        optsettings['tb_dir'] = tb_dir
        graph = tf.Graph()
        with graph.as_default():
            input_shapes=dataset.item_shapes
            model = netutils.import_model(network_module_name, input_shapes,
                                          **optsettings)
            epochs = args.epochs
            trainer.train_model(model)
            if args.save:
                save_file = os.path.join(tb_dir, '{}.tflearn'.format(
                    network_module_name
                ))
                netutils.save_model(model, save_file)

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

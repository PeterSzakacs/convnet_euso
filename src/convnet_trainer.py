# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os

import tensorflow as tf

import dataset.io.fs_io as io_utils
import net.network_utils as netutils
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

    # load dataset
    name, srcdir, item_types = args['name'], args['srcdir'], args['item_types']
    input_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    # create splitter and split dataset into train and test data
    mode = args['split_mode']
    fraction, num_items = args['test_items_fraction'], args['test_items_count']
    splitter = netutils.DatasetSplitter(mode, items_fraction=fraction,
                                        num_items=num_items)

    # import network model
    network, model_file = 'net.' + args['network'], args['model_file']
    tb_dir, run_id = args['tb_dir'], netutils.get_default_run_id(network)
    args['tb_dir'] = os.path.join(tb_dir, run_id)
    shapes = netutils.convert_item_shapes_to_convnet_input_shapes(dataset)
    model = netutils.import_model(network, shapes, **args)

    # prepare network trainer
    data_dict = splitter.get_data_and_targets(dataset)
    data_dict['train_data'] = netutils.reshape_data_for_convnet(
        model.network_graph, data_dict['train_data'])
    data_dict['test_data'] = netutils.reshape_data_for_convnet(
        model.network_graph, data_dict['test_data'])
    trainer = netutils.TfModelTrainer(data_dict, **args)

    # train model and optionally save if requested
    trainer.train_model(model)
    if args['save']:
        save_file = os.path.join(tb_dir, '{}.tflearn'.format(network))
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

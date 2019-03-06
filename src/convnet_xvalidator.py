import os

import tensorflow as tf

import dataset.io.fs_io as io_utils
import net.network_utils as netutils


def main(**settings):
    # load dataset
    name, srcdir = settings['name'], settings['srcdir']
    item_types = settings['item_types']
    input_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    # create dataset splitter
    fraction = settings['test_items_fraction']
    num_items = settings['test_items_count']
    splitter = netutils.DatasetSplitter('RANDOM', items_fraction=fraction,
                                        num_items=num_items)

    # import network
    net_module_name = settings['network']
    shapes = netutils.convert_item_shapes_to_convnet_input_shapes(dataset)
    model = netutils.import_model(net_module_name, shapes, **settings)

    # prepare network trainer
    num_epochs = settings['epochs']
    trainer = netutils.TfModelTrainer(splitter.get_data_and_targets(dataset),
                                      num_epochs=num_epochs, **settings)

    # main loop
    weights = model.trainable_layer_weights
    run_id = 'cval_{}'.format(netutils.get_default_run_id(net_module_name))
    num_crossvals = settings['num_crossvals']
    for run_idx in range(num_crossvals):
        print('Starting run {}'.format(run_idx + 1))
        data_dict = splitter.get_data_and_targets(dataset)
        data_dict['train_data'] = netutils.reshape_data_for_convnet(
            data_dict['train_data'])
        data_dict['test_data'] = netutils.reshape_data_for_convnet(
            data_dict['test_data'])
        trainer.train_model(model, data_dict=data_dict, run_id=run_id)
        model.trainable_layer_weights = weights


if __name__ == '__main__':
    import cmdint.cmd_interface_xvalidator as cmd
    import sys

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)
    args['network'] = 'net.' + args['network']
    main(**args)

import os

import tensorflow as tf

import dataset.dataset_utils as ds
import utils.io_utils as io_utils
import utils.network_utils as netutils


def main(**settings):
    name, srcdir = settings['name'], settings['srcdir']
    item_types = settings['item_types']
    input_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    fraction = settings['test_items_fraction']
    num_items = settings['test_items_count']
    splitter = netutils.DatasetSplitter('RANDOM', items_fraction=fraction,
                                        num_items=num_items)

    net_module_name = settings['network']
    model = netutils.import_model(net_module_name, dataset.item_shapes,
                                  **settings)
    weights = model.trainable_layer_weights
    num_epochs = settings['epochs']

    run_id = 'cval_{}'.format(netutils.get_default_run_id(net_module_name))
    num_crossvals = settings['num_crossvals']
    for run_idx in range(num_crossvals):
        print('Starting run {}'.format(run_idx + 1))
        data_dict = splitter.get_data_and_targets(dataset)
        data_dict['train_data'] = netutils.reshape_data_for_convnet(
            data_dict['train_data'])
        data_dict['test_data'] = netutils.reshape_data_for_convnet(
            data_dict['test_data'])
        netutils.train_model(model, data_dict, num_epochs=num_epochs,
                             run_id=run_id)
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

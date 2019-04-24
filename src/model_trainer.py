import os

import dataset.io.fs_io as io_utils
import net.network_utils as netutils


if __name__ == '__main__':
    import cmdint.cmd_interface_trainer as cmd
    import sys

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    # load dataset
    name, srcdir, item_types = args['name'], args['srcdir'], args['item_types']
    input_handler = io_utils.DatasetFsPersistencyHandler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    # create splitter and split dataset into train and test data
    mode = args['split_mode']
    fraction, num_items = args['test_items_fraction'], args['test_items_count']
    splitter = netutils.DatasetSplitter(mode, items_fraction=fraction,
                                        num_items=num_items)

    # import network model
    network, model_file = 'net.' + args['network'], args['model_file']
    tb_dir = args['tb_dir']
    model = netutils.import_model(network, dataset.item_shapes, **args)

    # prepare network trainer
    data_dict = splitter.get_data_and_targets(dataset)
    data_dict['train_data'] = netutils.convert_dataset_items_to_model_inputs(
        model, data_dict['train_data'])
    data_dict['test_data'] = netutils.convert_dataset_items_to_model_inputs(
        model, data_dict['test_data'])
    trainer = netutils.TfModelTrainer(data_dict, **args)

    # train model and optionally save if requested
    trainer.train_model(model)
    if args['save']:
        save_file = os.path.join(tb_dir, '{}.tflearn'.format(network))
        netutils.save_model(model, save_file)

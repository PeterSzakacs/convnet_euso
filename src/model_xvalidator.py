import dataset.io.fs_io as io_utils
import net.network_utils as netutils
import net.training.utils as train_utils


def main(**settings):
    # load dataset
    name, srcdir = settings['name'], settings['srcdir']
    item_types = settings['item_types']
    input_handler = io_utils.DatasetFsPersistencyHandler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    # create dataset splitter
    fraction = settings['test_items_fraction']
    num_items = settings['test_items_count']
    splitter = netutils.DatasetSplitter('RANDOM', items_fraction=fraction,
                                        num_items=num_items)

    # import network
    net_module_name = settings['network']
    model = netutils.import_model(net_module_name, dataset.item_shapes,
                                  **settings)
    graph = model.network_graph

    # prepare network trainer
    num_epochs = settings['num_epochs']
    trainer = train_utils.TfModelTrainer(
        splitter.get_data_and_targets(dataset), **settings)

    # main loop
    weights = {layer: model.get_layer_weights(layer)
               for layer in graph.trainable_layers}
    biases = {layer: model.get_layer_biases(layer)
              for layer in graph.trainable_layers}
    run_id = 'cval_{}'.format(netutils.get_default_run_id(net_module_name))
    num_crossvals = settings['num_crossvals']
    for run_idx in range(num_crossvals):
        print('Starting run {}'.format(run_idx + 1))
        data_dict = splitter.get_data_and_targets(dataset,
                                                  dict_format='PER_SET')
        tr, te = data_dict['train'], data_dict['test']
        inputs_dict = {
            'train_data': netutils.convert_to_model_inputs_dict(model, tr),
            'train_targets': netutils.convert_to_model_outputs_dict(model, tr),
            'test_data': netutils.convert_to_model_inputs_dict(model, te),
            'test_targets': netutils.convert_to_model_outputs_dict(model, te),
        }
        trainer.train_model(model, data_dict=inputs_dict, run_id=run_id)
        # restore initial weights and biases
        for layer in graph.trainable_layers:
            model.set_layer_weights(layer, weights[layer])
            model.set_layer_biases(layer, biases[layer])


if __name__ == '__main__':
    import cmdint.cmd_interface_xvalidator as cmd
    import sys

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)
    args['network'] = 'net.samples.' + args['network']
    main(**args)

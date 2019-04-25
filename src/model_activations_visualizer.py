import os

import dataset.io.fs_io as dset_io
import net.network_utils as netutils
import visualization.activations_visualization as acviz
acviz.use('svg')


def main(**settings):
    logdir = settings['logdir']

    # disable CUDA device access
    if settings['usecpu']:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load dataset
    name, srcdir = settings['name'], settings['srcdir']
    item_types = settings['item_types']
    input_handler = dset_io.DatasetFsPersistencyHandler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)
    items_slice = settings.get('items_slice', slice(0, None))
    data = dataset.get_data_as_dict(items_slice)

    # import network model
    net_module_name = settings['network']
    model = netutils.import_model(net_module_name, dataset.item_shapes,
                                  create_hidden_models=True, **settings)
    graph = model.network_graph

    # feed data to network model
    inputs = netutils.convert_dataset_items_to_model_inputs(model, data)
    activations = model.get_hidden_layer_activations(inputs)
    input_layers, hidden_layers = graph.input_layers, graph.hidden_layers
    #data_paths = model.network_graph.data_paths
    for layer_name, layer_activations in activations.items():
        out_dir = os.path.join(logdir, layer_name)
        os.makedirs(out_dir, exist_ok=True)
        layer = hidden_layers[layer_name]
        if len(layer.shape[1:]) == 3:
            fig_creator = acviz.visualize_3d_activations
        elif len(layer.shape[1:]) == 1:
            fig_creator = acviz.visualize_1d_activations
        print('creating activation figures for layer {}'.format(layer_name))
        for idx in range(len(layer_activations)):
            fig = fig_creator(layer_activations[idx], layer_name)
            savefile = os.path.join(out_dir, 'layer_{}_item_{}.svg'.format(
                                    layer_name, items_slice.start + idx))
            acviz.save_figure(fig, savefile)


if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_acviz as cmd

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)
    args['network'] = 'net.' + args['network']
    main(**args)

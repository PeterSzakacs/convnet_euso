import os

import dataset.constants as cons
import dataset.data_utils as dat
import net.network_utils as netutils

import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt


def visualize_conv_weights(conv_layer_weights):
    """
    Visualize conv2d weights of a single layer (optionally pre-sliced)
    and return as matplotlib.figure.Figure object.

    Parameters
    ----------
    :param conv_layer_weights:  the conv weights of a single layer (optionally
                                sliced by number of filters or depth)
    :type conv_layer_weights:   numpy.ndarray with shape (num_filters,
                                filter_depth, filter_height, filter_width)
    """
    f_count, f_depth = conv_layer_weights.shape[0:2]
    #f_height, f_width = conv_layer_weights.shape[2:4]
    fig, axes = plt.subplots(nrows=f_count, ncols=f_depth)
    axes = axes.flatten()
    for filter_idx in range(f_count):
        for depth_idx in range(f_depth):
            img = conv_layer_weights[filter_idx, depth_idx]
            ax = axes[filter_idx*f_depth + depth_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)
    return fig


def main(**args):
    # disable access to the installed CUDA device
    if args['usecpu']:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # a lil hacky way to get the model to restore properly
    network = args['network']
    packet_shape = args['packet_shape']
    item_types = dict.fromkeys(cons.ALL_ITEM_TYPES, True)
    item_shapes = dat.get_data_item_shapes(packet_shape, item_types)
    model = netutils.import_model(network, item_shapes, **args)

    filter_slice = args.get('filter_slice', slice(0, None))
    depth_slice = args.get('depth_slice', slice(0, None))
    conv_weights = model.conv_weights
    for layer_name, weights in conv_weights.items():
        weights_sliced = weights[filter_slice, depth_slice]
        fig = visualize_conv_weights(weights_sliced)
        fig.savefig('weights_of_layer_{}'.format(layer_name))
        plt.close(fig)


if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_filterviz as cmd

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    args['network'] = 'net.' + args['network']
    main(**args)
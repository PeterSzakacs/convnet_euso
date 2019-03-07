import dataset.constants as cons
import dataset.dataset_utils as ds
import net.constants as net_cons
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
        print('\tplotting filter {}'.format(filter_idx))
        for depth_idx in range(f_depth):
            print('\t\tplotting depth slice {}'.format(depth_idx))
            img = conv_layer_weights[filter_idx, depth_idx]
            ax = axes[filter_idx*f_depth + depth_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)
    return fig


def main(**args):
    network = 'net.' + args['network']

    # a lil hacky way to get the model to restore properly
    packet_shape = args['packet_shape']
    item_types = dict.fromkeys(cons.ALL_ITEM_TYPES, True)
    dataset = ds.numpy_dataset('name', packet_shape, item_types=item_types)
    shapes = netutils.convert_item_shapes_to_convnet_input_shapes(dataset)
    model = netutils.import_model(network, shapes, **args)

    layer_slice = args.get('layer_slice', slice(0, None))
    filter_slice = args.get('filter_slice', slice(0, None))
    depth_slice = args.get('depth_slice', slice(0, None))
    conv_weights = model.conv_weights[layer_slice]
    for layer_idx in range(len(conv_weights)):
        print('plotting layer {}'.format(layer_idx))
        weights = conv_weights[layer_idx][filter_slice, depth_slice]
        fig = visualize_conv_weights(weights)
        layer = layer_slice.start + layer_idx
        fig.savefig('weights_of_layer_{}'.format(layer + 1))
        plt.close(fig)


if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_filterviz as cmd

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    main(**args)
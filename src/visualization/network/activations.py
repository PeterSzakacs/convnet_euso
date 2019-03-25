import math

import matplotlib
import numpy as np


def use(mpl_backend='svg'):
    """
    Specify matplotlib render backend to use (default: svg).

    This function must be called upon import of this module, otherwise pyplot
    will not be available for creating figures.

    Parameters
    ----------
    :param mpl_backend: matplotlib backend to use
    :type mpl_backend:  str
    """
    global plt
    matplotlib.use(mpl_backend)
    import matplotlib.pyplot as plt


def visualize_1d_activations(activations, layer_name):
    """
    Visualize the activations (optionally pre-sliced) of a single layer whose
    output is a 1d tensor, i.e. a vector of values (such as a fully connected
    layer and return as a matplotlib.figure.Figure object.

    Parameters
    ----------
    :param activations: the output activations of the single layer (optionally
                        sliced)
    :type activations:  numpy.ndarray with shape (width, height, num_channels)
    :param layer_name:  the name of the layer which produced the activations,
                        will be used to set the figure suptitle
    :type layer_name:   str
    """
    n_outputs = len(activations)
    img_side = math.ceil(math.sqrt(n_outputs))
    activations_img = np.zeros((img_side*img_side, ), dtype=activations.dtype)
    activations_img[slice(n_outputs)] = activations
    activations_img = activations_img.reshape(img_side, img_side)

    fig, ax = plt.subplots(1, 1, frameon=False)
    fig.set_figwidth(img_side)
    fig.set_figheight(img_side)
    im = ax.imshow(activations_img, cmap='gray')
    x = [idx % img_side for idx in range(n_outputs, img_side*img_side)]
    y = [math.floor(idx / img_side)
         for idx in range(n_outputs, img_side*img_side)]
    bbox = im.get_window_extent()
    s = (bbox.max[0] - bbox.min[0])/img_side
    ax.scatter(x, y, s=s, color='red')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(layer_name, fontsize=32)
    return fig


def visualize_3d_activations(activations, layer_name):
    """
    Visualize the activations (optionally pre-sliced) of a single layer whose
    output is a 3d tensor, i.e. a 3d matrix of values (such as max_pool2d,
    conv2d, various normalization layers etc.) and return the activations as a
    matplotlib.figure.Figure object.

    Parameters
    ----------
    :param activations: the output activations of the single layer (optionally
                        sliced by output depth/number of channels)
    :type activations:  numpy.ndarray with shape (width, height, num_channels)
    :param layer_name:  the name of the layer which produced the activations,
                        will be used to set the figure suptitle
    :type layer_name:   str
    """
    n_channels = activations.shape[2]
    img_side = math.ceil(math.sqrt(n_channels))
    minval, maxval = activations.min(), activations.max()
    fig, axes = plt.subplots(img_side, img_side, sharey=True, frameon=False)
    fig.set_figwidth(img_side)
    fig.set_figheight(img_side)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    axes = axes.flatten()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for channel_idx in range(n_channels):
        channel = activations[slice(None), slice(None), channel_idx]
        channel_ax = axes[channel_idx]
        channel_ax.imshow(channel, vmin=minval, vmax=maxval, cmap='gray')
    for idx in range(n_channels, img_side*img_side):
        axes[idx].axis('off')
    fig.suptitle(layer_name, fontsize=32)
    return fig


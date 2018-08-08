import os
import tflearn
import numpy as np
import numpy.ma as ma

import matplotlib
# python3-tk is not availble here, have to use something else
matplotlib.use('svg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class VisualizerCallback(tflearn.callbacks.Callback):
    def __init__(self, model, conv_layers, outdir):
        self.model = model
        self.conv_layers = conv_layers
        self.outdir = outdir

    def __nice_imshow(self, ax, data, vmin=None, vmax=None, cmap=None):
        """Wrapper around plt.imshow"""
        if cmap is None:
            cmap = cm.jet
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
        plt.colorbar(im, cax=cax)

    def __make_mosaic(self, imgs, nrows, ncols, border=1):
        #TODO: fix 
        """
        Given a set of images with all the same shape, makes a
        mosaic with nrows and ncols
        """
        nimgs = imgs.shape[0]
        imshape = imgs.shape[1:]

        mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                                ncols * imshape[1] + (ncols - 1) * border),
                                dtype=np.float32)

        paddedh = imshape[0] + border
        paddedw = imshape[1] + border
        for i in range(nimgs):
            row = int(np.floor(i / ncols))
            col = i % ncols

            mosaic[row * paddedh:row * paddedh + imshape[0],
                col * paddedw:col * paddedw + imshape[1]] = imgs[i]
        return mosaic

    def on_train_end(self, training_state):


        for idx in range(len(self.conv_layers)):
            layer = self.conv_layers[idx]
            W_visu = self.model.get_weights(layer.W)
            W_visu = np.squeeze(W_visu)
            print(W_visu)
            #plt.figure(figsize=(15, 15))
            #plt.title('conv{} weights'.format(idx))
            #self.__nice_imshow(plt.gca(), self.__make_mosaic(W_visu, 6, 6), cmap=cm.binary)
            #plt.savefig(os.path.join(self.outdir, 'conv{}_weights'.format(idx)))
            #plt.close()


import tflearn
import matplotlib
# python3-tk is not availble where this is mostly used, have to use something else
matplotlib.use('svg')

import matplotlib.pyplot as plt


class VisualizerCallback(tflearn.callbacks.Callback):
    def __init__(self, model, x,
                 layers_to_observe=(),
                 kernels=10,
                 inputs=3):
        self.model = model
        self.x = x
        self.kernels = kernels
        self.inputs = inputs
        self.observers = [tflearn.DNN(l) for l in layers_to_observe]

    def on_epoch_end(self, training_state):
        print ("conv_observers:", self.observers)
        # get the outputs produced by all observed hidden layers
        outputs = [o.predict(self.x) for o in self.observers]

        #print ("outputs:", outputs.shape)
        # for the first 'self.inputs'-sized subset of all input images, create a set of figures at the end of a training epoch
        for idx in range(self.inputs):
            plt.figure(frameon=False)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            ix = 1
            print("output length:", len(outputs))
            # for the output of every observed hidden layer
            for output in outputs:
                print("output shape:", output.shape)
                # for the output of the first 'self.kernels'-sized subset of all kernels of this hidden layer
                for kernel in range(self.kernels):
                    plt.subplot(len(outputs), self.kernels, ix)
                    plt.imshow(output[idx, :, :, kernel])
                    plt.axis('off')
                    ix += 1
            plt.savefig('conv_outputs-for-image:%i-at-epoch:%i.png'
                % (idx, training_state.epoch))
        plt.close("all")

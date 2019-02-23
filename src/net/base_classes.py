import abc

import tflearn

class NetworkModel():

    def __init__(self, neural_network, **model_settings):
        self._net = neural_network
        self.initialize_model(**model_settings)

    @property
    def network_graph(self):
        return self._net

    @property
    def network_model(self):
        return self._model

    @property
    def trainable_layer_weights(self):
        model, layers = self._model, self._net.trainable_layers
        return tuple(model.get_weights(layer.W) for layer in layers)

    @trainable_layer_weights.setter
    def trainable_layer_weights(self, values):
        model, layers = self._model, self._net.trainable_layers
        for idx in range(len(layers)):
            model.set_weights(layers[idx].W, values[idx])

    @property
    def trainable_layer_biases(self):
        model, layers = self._model, self._net.trainable_layers
        return tuple(model.get_weights(layer.b) for layer in layers)

    @trainable_layer_biases.setter
    def trainable_layer_biases(self, values):
        model, layers = self._model, self._net.trainable_layers
        for idx in range(len(layers)):
            model.set_weights(layers[idx].b, values[idx])

    @property
    def trainable_layer_weights_snapshot(self):
        return self._train_layer_w

    @property
    def trainable_layer_biases_snapshot(self):
        return self._train_layer_b

    def initialize_model(self, **model_settings):
        tb_dir = model_settings.get('tb_dir', '/tmp/tflearn_logs/')
        tb_verbosity = model_settings.get('tb_verbosity', 0)
        model = tflearn.DNN(self._net.output_layer, tensorboard_dir=tb_dir,
                            tensorboard_verbose=tb_verbosity)
        self._model = model
        self.update_snapshots()

    def load_from_file(model_file, **optargs):
        return self._model.load(model_file, **optargs)

    def restore_from_snapshot(self):
        model, layers = self._model, self._net.trainable_layers
        weights, biases = self._train_layer_w, self._train_layer_b
        for idx in range(len(layers)):
            model.set_weights(layers[idx].W, weights[idx])
            model.set_weights(layers[idx].b, biases[idx])

    def update_snapshots(self):
        model, layers = self._model, self._net.trainable_layers
        self._train_layer_w = self.trainable_layer_weights
        self._train_layer_b = self.trainable_layer_biases


class NeuralNetwork(abc.ABC):

    def __init__(self, trainable_layers, output_layer):
        self._train_layers = trainable_layers
        self._out = output_layer

    # properties

    @abc.abstractproperty
    def network_type(self):
        pass

    @property
    def trainable_layers(self):
        return self._train_layers

    @property
    def output_layer(self):
        return self._out

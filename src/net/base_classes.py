import abc

import tflearn

class NetworkModel():

    def __init__(self, neural_network, **model_settings):
        self._net = neural_network
        self.initialize_model(**model_settings)

    def _convert_weights_to_internal_form(self, layer, weights):
        return weights

    def _convert_biases_to_internal_form(self, layer, biases):
        return biases

    def _convert_weights_to_external_form(self, layer, weights):
        return weights

    def _convert_biases_to_external_form(self, layer, biases):
        return biases

    @property
    def network_graph(self):
        return self._net

    @property
    def network_model(self):
        return self._model

    @property
    def trainable_layer_weights(self):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_weights_to_external_form
        return tuple(convert(layer, model.get_weights(layer.W))
                     for layer in layers)

    @trainable_layer_weights.setter
    def trainable_layer_weights(self, values):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_weights_to_internal_form
        for idx in range(len(layers)):
            layer = layers[idx]
            model.set_weights(layer.W, convert(layer, values[idx]))

    @property
    def trainable_layer_biases(self):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_biases_to_external_form
        return tuple(convert(layer, model.get_weights(layer.b))
                     for layer in layers)

    @trainable_layer_biases.setter
    def trainable_layer_biases(self, values):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_biases_to_internal_form
        for idx in range(len(layers)):
            layer = layers[idx]
            model.set_weights(layer.b, convert(layer, values[idx]))

    def initialize_model(self, **model_settings):
        tb_dir = model_settings.get('tb_dir', '/tmp/tflearn_logs/')
        tb_verbosity = model_settings.get('tb_verbosity', 0)
        model = tflearn.DNN(self._net.output_layer, tensorboard_dir=tb_dir,
                            tensorboard_verbose=tb_verbosity)
        self._model = model

    def load_from_file(self, model_file, **optargs):
        w_only = optargs.get('weights_only', False)
        return self._model.load(model_file, weights_only=w_only)


class NeuralNetwork(abc.ABC):

    def __init__(self, inputs, outputs, layers):
        trainable_layers = layers['trainable']
        hidden_layers = layers['hidden']
        self._inputs = inputs
        self._out = outputs
        self._trainable = trainable_layers
        self._hidden = hidden_layers

    # properties

    @abc.abstractproperty
    def network_type(self):
        pass

    @property
    def trainable_layers(self):
        return self._trainable

    @property
    def hidden_layers(self):
        return self._hidden

    @property
    def input_layers(self):
        return self._inputs

    @property
    def output_layer(self):
        return self._out

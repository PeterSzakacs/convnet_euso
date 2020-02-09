import abc
import re


class NeuralNetwork(abc.ABC):

    # legal names contain alphanumeric characters, underscore, dash and dots
    # (essentially, filesystem friendly)
    __allowable_layer_name_regex__ = re.compile('[a-zA-Z0-9_\\-.]+')
    __default_layer_name_regex__ = re.compile('([^/]+)/.+')

    def __init__(self, builder):
        layers, categories = builder.layers_dict, builder.layer_categories
        trainable_layers = categories['trainable']
        hidden_layers = categories['hidden']
        input_layers = categories['input']
        self._inputs = {name: layers[name] for name in input_layers}
        self._output = {builder.output_layer: layers[builder.output_layer]}
        self._trainable = {
            name: layers[name] for name in trainable_layers
        }
        self._hidden = {name: layers[name] for name in hidden_layers}
        self._paths = builder.data_paths.copy()

    # properties

    @property
    @abc.abstractmethod
    def network_type(self):
        pass

    @property
    @abc.abstractmethod
    def input_spec(self):
        pass

    @property
    @abc.abstractmethod
    def output_spec(self):
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
        return self._output

    @property
    def data_paths(self):
        return self._paths


class AutoEncoder(NeuralNetwork):

    def __init__(self, builder, encoder_layer_name):
        super(AutoEncoder, self).__init__(builder)
        self._enc_layer_name = encoder_layer_name
        self._enc_layer = builder.layers_dict[encoder_layer_name]

    @property
    def network_type(self):
        return 'autoencoder'

    @property
    def encoder_layer(self):
        return self._enc_layer

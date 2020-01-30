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
        self._out = layers[builder.output_layer]
        self._out_name = builder.output_layer
        self._inputs = {name: layers[name] for name in input_layers}
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
        return self._out

    @property
    def output_layer_name(self):
        return self._out_name

    @property
    def data_paths(self):
        return self._paths

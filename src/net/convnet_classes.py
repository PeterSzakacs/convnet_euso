import numpy as np

import net.base_classes as bclasses


class Conv2DNetworkModel(bclasses.NetworkModel):

    def __init__(self, neural_network, **model_settings):
        super(Conv2DNetworkModel, self).__init__(
            neural_network, **model_settings)

    def _convert_weights_to_external_form(self, name, weights):
        net = self.network_graph
        if name in net.conv_layers:
            return np.moveaxis(weights, [2, 3], [1, 0])
        elif name in net.fc_layers:
            return np.moveaxis(weights, 0, 1)
        super()._convert_weights_to_external_form(name, weights)

    def _convert_weights_to_internal_form(self, name, weights):
        net = self.network_graph
        if name in net.conv_layers:
            return np.moveaxis(weights, [0, 1], [3, 2])
        elif name in net.fc_layers:
            return np.moveaxis(weights, 1, 0)
        super()._convert_weights_to_internal_form(name, weights)

    # properties

    @property
    def conv_weights(self):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        convert = self._convert_weights_to_external_form
        return {name: convert(name, model.get_weights(layer.W))
                for name, layer in conv_layers.items()}

    @conv_weights.setter
    def conv_weights(self, values):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        convert = self._convert_weights_to_internal_form
        for name, layer in conv_layers.items():
            model.set_weights(layer.W, convert(name, values[name]))

    @property
    def conv_biases(self):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        return {name: model.get_weights(layer.b)
                for name, layer in conv_layers.items()}

    @conv_biases.setter
    def conv_biases(self, values):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        for name, layer in conv_layers.items():
            model.set_weights(layer.b, values[name])

    @property
    def fc_weights(self):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        convert = self._convert_weights_to_external_form
        return {name: convert(name, model.get_weights(layer.W))
                for name, layer in fc_layers.items()}

    @fc_weights.setter
    def fc_weights(self, values):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        convert = self._convert_weights_to_internal_form
        for name, layer in fc_layers.items():
            model.set_weights(layer.W, convert(name, values[name]))

    @property
    def fc_biases(self):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        return {name: model.get_weights(layer.b)
                for name, layer in fc_layers.items()}

    @fc_biases.setter
    def fc_biases(self, values):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        for name, layer in fc_layers.items():
            model.set_weights(layer.b, values[name])


class Conv2DNetwork(bclasses.NeuralNetwork):

    def __init__(self, builder):
        super(Conv2DNetwork, self).__init__(builder)
        layers, layer_types = builder.layers_dict, builder.layer_types
        conv_layers = layer_types['Conv2D']
        fc_layers = layer_types['FC']
        self._conv = {name: layers[name]['layer'] for name in conv_layers}
        self._fc = {name: layers[name]['layer'] for name in fc_layers}

    @property
    def network_type(self):
        return 'CONV2D'

    @property
    def conv_layers(self):
        return self._conv

    @property
    def fc_layers(self):
        return self._fc

    def get_filter_size(self, conv_layer_name):
        conv_layer = self._conv[conv_layer_name]
        tensor_shape = conv_layer.W.shape
        return tuple(int(tensor_shape[idx]) for idx in range(0,3))

    def get_num_filters(self, conv_layer_name):
        conv_layer = self._conv[conv_layer_name]
        return int(conv_layer.W.shape[3])

    def get_num_neurons(self, fc_layer_name):
        fc_layer = self._conv[fc_layer_name]
        return int(fc_layer.W.shape[1])

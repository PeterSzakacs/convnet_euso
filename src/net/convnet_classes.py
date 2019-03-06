import numpy as np

import net.base_classes as bclasses


class Conv2DNetworkModel(bclasses.NetworkModel):

    def __init__(self, neural_network, **model_settings):
        super(Conv2DNetworkModel, self).__init__(
            neural_network, **model_settings)

    def _convert_weights_to_external_form(self, layer, weights):
        net = self.network_graph
        if layer in net.conv_layers:
            return np.moveaxis(weights, [2, 3], [1, 0])
        elif layer in net.fc_layers:
            return np.moveaxis(weights, 0, 1)
        super()._convert_weights_to_external_form(layer, weights)

    def _convert_weights_to_internal_form(self, layer, weights):
        net = self.network_graph
        if layer in net.conv_layers:
            return np.moveaxis(weights, [0, 1], [3, 2])
        elif layer in net.fc_layers:
            return np.moveaxis(weights, 1, 0)
        super()._convert_weights_to_internal_form(layer, weights)

    # properties

    @property
    def conv_weights(self):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        convert = self._convert_weights_to_external_form
        return tuple(convert(layer, model.get_weights(layer.W))
                     for layer in conv_layers)

    @conv_weights.setter
    def conv_weights(self, values):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        convert = self._convert_weights_to_internal_form
        for idx in range(len(conv_layers)):
            layer = conv_layers[idx]
            model.set_weights(layer.W, convert(layer, values[idx]))

    @property
    def conv_biases(self):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        return tuple(model.get_weights(layer.b) for layer in conv_layers)

    @conv_biases.setter
    def conv_biases(self, values):
        model, conv_layers = self.network_model, self.network_graph.conv_layers
        for idx in range(len(conv_layers)):
            model.set_weights(conv_layers[idx].b, values[idx])

    @property
    def fc_weights(self):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        convert = self._convert_weights_to_external_form
        return tuple(convert(layer, model.get_weights(layer.W))
                     for layer in fc_layers)

    @fc_weights.setter
    def fc_weights(self, values):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        convert = self._convert_weights_to_internal_form
        for idx in range(len(fc_layers)):
            layer = fc_layers[idx]
            model.set_weights(layer.W, convert(layer, values[idx]))

    @property
    def fc_biases(self):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        return tuple(model.get_weights(layer.b) for layer in fc_layers)

    @fc_biases.setter
    def fc_biases(self, values):
        model, fc_layers = self.network_model, self.network_graph.fc_layers
        for idx in range(len(fc_layers)):
            model.set_weights(fc_layers[idx].b, values[idx])


class Conv2DNetwork(bclasses.NeuralNetwork):

    def __init__(self, conv_layers, fc_layers, output_layer):
        super(Conv2DNetwork, self).__init__(conv_layers + fc_layers,
                                            output_layer)
        self._conv = conv_layers
        self._fc = fc_layers

    @property
    def network_type(self):
        return 'CONV2D'

    @property
    def conv_layers(self):
        return self._conv

    @property
    def fc_layers(self):
        return self._fc

    def get_filter_size(self, conv_layer):
        tensor_shape = conv_layer.W.shape
        return tuple(tensor_shape[idx].value for idx in range(0,3))

    def get_num_filters(self, conv_layer):
        return conv_layer.W.shape[3]

    def get_num_neurons(self, fc_layer):
        return fc_layer.W.shape[1]

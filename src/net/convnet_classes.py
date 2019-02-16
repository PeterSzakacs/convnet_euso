import net.base_classes as bclasses


class Conv2DNetworkModel(bclasses.NetworkModel):

    def __init__(self, neural_network, **model_settings):
        super(Conv2DNetworkModel, self).__init__(
            neural_network, **model_settings)

    # properties

    @property
    def conv_weights(self):
        model, graph = self.network_model, self.network_graph
        conv_layers = graph.conv_layers
        return tuple(model.get_weights(layer.W) for layer in conv_layers)

    @property
    def conv_biases(self):
        model, graph = self.network_model, self.network_graph
        conv_layers = graph.conv_layers
        return tuple(model.get_weights(layer.b) for layer in conv_layers)

    @property
    def conv_weights_snapshot(self):
        return self._conv_w

    @property
    def conv_biases_snapshot(self):
        return self._conv_b

    @property
    def fc_weights(self):
        model, graph = self.network_model, self.network_graph
        fc_layers = graph.fc_layers
        return tuple(model.get_weights(layer.W) for layer in fc_layers)

    @property
    def fc_biases(self):
        model, graph = self.network_model, self.network_graph
        fc_layers = graph.fc_layers
        return tuple(model.get_weights(layer.b) for layer in fc_layers)

    @property
    def fc_weights_snapshot(self):
        return self._fc_w

    @property
    def fc_biases_snapshot(self):
        return self._fc_b

    # interface methods

    def update_snapshots(self):
        super(Conv2DNetworkModel, self).update_snapshots()
        model = self.network_model
        self._conv_w = self.conv_weights
        self._conv_b = self.conv_biases
        self._fc_w = self.fc_weights
        self._fc_b = self.fc_biases


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

    def get_filter_size(self, layer):
        tensor_shape = layer.W.shape
        return tuple(tensor_shape[idx].value for idx in range(0,3))

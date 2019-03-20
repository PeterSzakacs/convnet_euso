import abc

import numpy as np
import tflearn

class NetworkModel():

    def __init__(self, neural_network, **model_settings):
        self._net = neural_network
        self._hidden_models = {}
        self.initialize_model(**model_settings)

    def _update_hidden_models(self):
        hidden_layers = self._net.hidden_layers
        session = self._model.session
        for layer in hidden_layers:
            model = self._hidden_models[layer.name]
            model.session.close()
            model.session = session
            model.trainer.session = session
            model.predictor.session = session

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

    def initialize_model(self, create_hidden_models=False, **model_settings):
        settings = {}
        settings['tensorboard_dir'] = model_settings.get('tb_dir',
                                                         '/tmp/tflearn_logs/')
        settings['tensorboard_verbose'] = tb_verbosity = model_settings.get(
            'tb_verbosity', 0)
        model = tflearn.DNN(self._net.output_layer, **settings)
        self._model = model
        if create_hidden_models:
            hidden_layers = self._net.hidden_layers
            hidden_models = {layer.name: tflearn.DNN(layer)
                             for layer in hidden_layers}
            self._hidden_models = hidden_models

    def get_hidden_layer_activations(self, input_data_seq):
        if len(self._hidden_models) == 0:
            raise Exception('Hidden layer activations not retrievable.')
        hidden_layers = self._net.hidden_layers
        hidden_models = [self._hidden_models[layer.name]
                         for layer in hidden_layers]
        return [[np.squeeze(model.predict([input_data]), axis=0)
                 for model in hidden_models]
                for input_data in input_data_seq]

    def load_from_file(self, model_file, **optargs):
        w_only = optargs.get('weights_only', False)
        self._model.load(model_file, weights_only=w_only)
        if len(self._hidden_models) > 0:
            self._update_hidden_models()


class NeuralNetwork(abc.ABC):

    def __init__(self, inputs, outputs, layers):
        trainable_layers = layers['trainable']
        hidden_layers = layers['hidden']
        self._inputs = inputs
        self._out = outputs
        self._trainable = trainable_layers
        self._hidden = hidden_layers
        self._paths = self._get_data_paths()

    def _get_data_paths(self):
        def _get_next_tensor(curr_tensor):
            consumers = curr_tensor.consumers()
            # 'Switch' is tensorflow Dropout
            if consumers[0].type == 'Switch':
                return consumers[1].outputs[0]
            else:
                return consumers[0].outputs[0]
        input_layers, output_layer = self.input_layers, self.output_layer
        hidden_layers = self.hidden_layers
        paths = {input_layer.name: [] for input_layer in input_layers}
        for input_layer in input_layers:
            input_lst = paths[input_layer.name]
            next_tensor = input_layer
            while next_tensor is not output_layer:
                next_tensor = _get_next_tensor(next_tensor)
                while (next_tensor not in hidden_layers
                    and next_tensor is not output_layer):
                    next_tensor = _get_next_tensor(next_tensor)
                input_lst.append(next_tensor)
        return paths

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

    @property
    def data_paths(self):
        return self._paths

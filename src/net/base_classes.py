import abc
import re

import tflearn

class NetworkModel():

    def __init__(self, neural_network, **model_settings):
        self._net = neural_network
        self._hidden_models = {}
        self.initialize_model(**model_settings)

    def _update_hidden_models(self):
        hidden_layers = self._net.hidden_layers
        session = self._model.session
        for layer_name in hidden_layers.keys():
            model = self._hidden_models[layer_name]
            model.session.close()
            model.session = session
            model.trainer.session = session
            model.predictor.session = session

    def _convert_weights_to_internal_form(self, name, weights):
        return weights

    def _convert_biases_to_internal_form(self, name, biases):
        return biases

    def _convert_weights_to_external_form(self, name, weights):
        return weights

    def _convert_biases_to_external_form(self, name, biases):
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
        return {name: convert(name, model.get_weights(layer.W))
                for name, layer in layers.items()}

    @trainable_layer_weights.setter
    def trainable_layer_weights(self, values):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_weights_to_internal_form
        for name, layer in layers.items():
            model.set_weights(layer.W, convert(name, values[name]))

    @property
    def trainable_layer_biases(self):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_biases_to_external_form
        return {name: convert(name, model.get_weights(layer.b))
                for name, layer in layers.items()}

    @trainable_layer_biases.setter
    def trainable_layer_biases(self, values):
        model, layers = self._model, self._net.trainable_layers
        convert = self._convert_biases_to_internal_form
        for name, layer in layers.items():
            model.set_weights(layer.b, convert(name, values[name]))

    def initialize_model(self, create_hidden_models=False, **model_settings):
        settings = {}
        settings['tensorboard_dir'] = model_settings.get('tb_dir',
                                                         '/tmp/tflearn_logs/')
        settings['tensorboard_verbose'] = tb_verbosity = model_settings.get(
            'tb_verbosity', 0)
        model = tflearn.DNN(self._net.output_layer, **settings)
        self._model = model
        session = self._model.session
        if create_hidden_models:
            hidden_layers = self._net.hidden_layers
            hidden_models = {name: tflearn.DNN(layer, session=session)
                             for name, layer in hidden_layers.items()}
            self._hidden_models = hidden_models

    def get_hidden_layer_activations(self, input_data_dict):
        if len(self._hidden_models) == 0:
            raise Exception('Hidden layer activations not retrievable.')
        hidden_models = self._hidden_models
        return {layer_name: model.predict(input_data_dict)
                for layer_name, model in hidden_models.items()}

    def load_from_file(self, model_file, **optargs):
        w_only = optargs.get('weights_only', False)
        self._model.load(model_file, weights_only=w_only)
        if len(self._hidden_models) > 0:
            self._update_hidden_models()


class NeuralNetwork(abc.ABC):

    # legal names contain alphanumeric characters, underscore, dash and dots
    # (essentially, filesystem friendly)
    __allowable_layer_name_regex__ = re.compile('[a-zA-Z0-9_\\-.]+')
    __default_layer_name_regex__ = re.compile('([^/]+)/.+')

    def __init__(self, inputs, output, layers):
        trainable_layers = layers['trainable']
        hidden_layers = layers['hidden']
        item_type_to_inputs, input_layers = {}, {}
        for item_type_name, layer in inputs.items():
            sanitized_name = self._sanitize_layer_name(layer.name)
            item_type_to_inputs[item_type_name] = sanitized_name
            input_layers[sanitized_name] = layer
        self._item_type_to_input_mappings = item_type_to_inputs
        self._inputs = input_layers
        self._out = output
        self._trainable = {self._sanitize_layer_name(layer.name): layer
                           for layer in trainable_layers}
        self._hidden = {self._sanitize_layer_name(layer.name): layer
                        for layer in hidden_layers}
        self._paths = self._get_data_paths()

    def _sanitize_layer_name(self, layer_name):
        match = self.__default_layer_name_regex__.fullmatch(layer_name)
        if match:
            name = match.groups()[0]
            if self.__allowable_layer_name_regex__.fullmatch(name):
                return name
        raise Exception('Illegal layer name: {}'.format(layer_name))

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
        paths = {}
        for input_name, input_layer in input_layers.items():
            paths[input_name] = []
            input_lst = paths[input_name]
            next_tensor = input_layer
            while next_tensor is not output_layer:
                next_tensor = _get_next_tensor(next_tensor)
                while (next_tensor not in hidden_layers.values()
                    and next_tensor is not output_layer):
                    next_tensor = _get_next_tensor(next_tensor)
                input_lst.append(next_tensor)
        for input_name, input_layer in input_layers.items():
            paths[input_name] = [self._sanitize_layer_name(layer.name)
                                 for layer in paths[input_name]]
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

    @property
    def item_type_to_input_name_mapping(self):
        return self._item_type_to_input_mappings

import abc
import re

import tflearn
import tflearn.layers.core as core
import tflearn.layers.conv as conv
import tflearn.layers.normalization as norm
import tflearn.layers.merge_ops as merge
import tflearn.layers.estimator as est

import net.constants as net_cons


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


class GraphBuilder():

    # legal names contain alphanumeric characters, underscore, dash and dots
    # (essentially, filesystem friendly)
    #__allowable_layer_name_regex__ = re.compile('[a-zA-Z0-9_\\-.]+')
    __default_layer_name_regex__ = re.compile('([^/]+)/.+')

    def __init__(self):
        self._layers = {}
        self._layer_categories = {c: [] for c in net_cons.LAYER_CATEGORIES}
        self._curr_layer_name = None
        self._inputs = {}
        self._output = None

    def _add_layer(self, layer, categories):
        name = self._sanitize_layer_name(layer.name)
        layers, layers_cat = self._layers, self._layer_categories
        layers[name] = layer
        for category_name in categories:
            layers_cat[category_name].append(name)
        self._curr_layer_name = name
        return name

    def _sanitize_layer_name(self, layer_name):
        match = self.__default_layer_name_regex__.fullmatch(layer_name)
        if match:
            return match.groups()[0]
        else:
            raise Exception('Illegal layer name: {}'.format(layer_name))

    @property
    def layer_categories(self):
        return self._layer_categories

    @property
    def layers_dict(self):
        return self._layers

    @property
    def input_to_item_type_mapping(self):
        return self._inputs

    @property
    def output_layer(self):
        return self._output

    def finalize(self, layer_name, trainable=True, **kwargs):
        if self._output is not None:
            raise Exception('Output layer already set')
        try:
            self._layer_categories['hidden'].remove(layer_name)
        except ValueError:
            raise Exception('Unknown layer "{}"'.format(layer_name))
        layer = self._layers[layer_name]
        if trainable:
            # the layer object does not change, only a trainer object is added,
            # no need to reassign to the dict again
            est.regression(layer, **kwargs)
        self._output = layer_name

    def add_input_layer(self, input_shape, input_item_type, **kwargs):
        layer = core.input_data(shape=input_shape, **kwargs)
        name = self._add_layer(layer, ('input', ))
        self._inputs[name] = input_item_type
        return name

    # core layers (fully connected, dropout etc.)

    def add_fc_layer(self, n_units, prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = core.fully_connected(prev, n_units, **kwargs)
        categories = ['FC', 'hidden']
        trainables = tflearn.get_all_trainable_variable()
        if layer.W in trainables:
            categories.append('trainable')
        return self._add_layer(layer, categories)

    def add_dropout_layer(self, dropout_rate, prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = core.dropout(prev, dropout_rate, **kwargs)
        categories = ['Dropout', 'hidden']
        return self._add_layer(layer, categories)

    # convolutional layers

    def add_conv2d_layer(self, n_filters, filter_size, filter_strides=1,
                         prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = conv.conv_2d(prev, n_filters, filter_size,
                             strides=filter_strides, **kwargs)
        categories = ['Conv2D', 'hidden']
        trainables = tflearn.get_all_trainable_variable()
        if layer.W in trainables:
            categories.append('trainable')
        return self._add_layer(layer, categories)

    # max pooling layers

    def add_maxpool2d_layer(self, window_size, window_strides=None,
                            prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = conv.max_pool_2d(prev, window_size, strides=window_strides,
                                 **kwargs)
        categories = ['MaxPool2D', 'hidden']
        return self._add_layer(layer, categories)

    # normalization layers

    def add_lrn_layer(self, prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = norm.local_response_normalization(prev, **kwargs)
        categories = ['LRN', 'hidden']
        return self._add_layer(layer, categories)

    # merge and reshape operations

    def add_flatten_layer(self, prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = core.flatten(prev, **kwargs)
        categories = ['Flatten', 'hidden']
        return self._add_layer(layer, categories)

    def add_reshape_layer(self, new_shape, prev_layer_name=None, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = core.reshape(prev, [-1, *new_shape], **kwargs)
        categories = ['Reshape', 'hidden']
        return self._add_layer(layer, categories)

    def add_merge_layer(self, prev_layer_names, merge_mode, **kwargs):
        prev = [self._layers[name] for name in prev_layer_names]
        layer = merge.merge(prev, merge_mode, **kwargs)
        categories = ['Merge', 'hidden']
        return self._add_layer(layer, categories)

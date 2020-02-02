import re

import tflearn
import tflearn.layers.core as core
import tflearn.layers.conv as conv
import tflearn.layers.normalization as norm
import tflearn.layers.merge_ops as merge
import tflearn.layers.estimator as est

import net.constants as net_cons


class GraphBuilder:

    # legal names contain alphanumeric characters, underscore, dash and dots
    # (essentially, filesystem friendly)
    #__allowable_layer_name_regex__ = re.compile('[a-zA-Z0-9_\\-.]+')
    __default_layer_name_regex__ = re.compile('([^/]+)/.+')

    def __init__(self):
        self._layers = {}
        self._layer_categories = {c: [] for c in net_cons.LAYER_CATEGORIES}
        self._layer_types = {c: [] for c in net_cons.LAYER_TYPES}
        self._curr_layer_name = None
        self._output = None
        self._paths = {}
        self._curr_path = None

    def _add_layer(self, layer, exclude_from_path=False):
        name = self._sanitize_layer_name(layer['layer'].name)
        self._layers[name] = layer

        # add name of the current layer to all of its categories
        categories = self._layer_categories
        for category in layer['categories']:
            categories[category].append(name)

        # add name of the current layer to its type
        self._layer_types[layer['type']].append(name)

        # update current layer name and the newest layer on the current path
        self._curr_layer_name = name
        if not (self._curr_path is None or exclude_from_path):
            self._curr_path.append(name)
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
    def layer_types(self):
        return self._layer_types

    @property
    def layers_dict(self):
        return self._layers

    @property
    def output_layer(self):
        return self._output

    @property
    def data_paths(self):
        return self._paths

    def start_new_path(self, first_layer=None):
        new_path = []
        if first_layer is None:
            first_layer = self._curr_layer_name
        else:
            if first_layer not in self._layers:
                raise Exception('Unknown layer {}'.format(first_layer))
            new_path.append(first_layer)
        self._paths[first_layer] = new_path
        self._curr_path = new_path

    def end_current_path(self):
        self._curr_path = None

    def finalize(self, layer_name, trainable=True, **kwargs):
        if self._output is not None:
            raise Exception('Output layer already set')
        try:
            layer = self._layers[layer_name]
        except ValueError:
            raise Exception('Unknown layer "{}"'.format(layer_name))
        self._layer_categories['hidden'].remove(layer_name)
        for path in self._paths.values():
            if layer_name in path:
                path.remove(layer_name)
        layer['categories'].remove('hidden')
        if trainable:
            # the layer object does not change, only a trainer object is added,
            # no need to reassign to the dict again
            est.regression(layer['layer'], **kwargs)
        self._output = layer_name

    def add_input_layer(self, input_shape, exclude_from_path=False, **kwargs):
        layer = {
            "layer": core.input_data(shape=[None, *input_shape], **kwargs),
            "type": "Input", "categories": ("input", )
        }
        name = self._add_layer(layer, exclude_from_path)
        return name

    # core layers (fully connected, dropout etc.)

    def add_fc_layer(self, n_units, prev_layer_name=None,
                     exclude_from_path=False, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = core.fully_connected(prev, n_units, **kwargs)
        categories = ['hidden']
        if layer.W in tflearn.get_all_trainable_variable():
            categories.append('trainable')
        return self._add_layer(
            {"layer": layer, "type": "FC", "categories": categories},
            exclude_from_path
        )

    def add_dropout_layer(self, dropout_rate, prev_layer_name=None,
                          exclude_from_path=True, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = {
            "layer": core.dropout(prev, dropout_rate, **kwargs),
            "type": "Dropout", "categories": ("hidden", )
        }
        return self._add_layer(layer, exclude_from_path)

    # convolutional layers

    def add_conv2d_layer(self, n_filters, filter_size, filter_strides=1,
                         prev_layer_name=None, exclude_from_path=False,
                         **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = conv.conv_2d(prev, n_filters, filter_size,
                             strides=filter_strides, **kwargs)
        categories = ['hidden']
        if layer.W in tflearn.get_all_trainable_variable():
            categories.append('trainable')
        return self._add_layer(
            {"layer": layer, "type": "Conv2D", "categories": categories},
            exclude_from_path
        )

    # max pooling layers

    def add_maxpool2d_layer(self, window_size, window_strides=None,
                            prev_layer_name=None, exclude_from_path=False,
                            **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = {
            "layer": conv.max_pool_2d(prev, window_size,
                                      strides=window_strides, **kwargs),
            "type": "MaxPool2D", "categories": ("hidden", )
        }
        return self._add_layer(layer, exclude_from_path)

    # normalization layers

    def add_lrn_layer(self, prev_layer_name=None, exclude_from_path=False,
                      **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = {
            "layer": norm.local_response_normalization(prev, **kwargs),
            "type": "LRN", "categories": ("hidden", )
        }
        return self._add_layer(layer, exclude_from_path)

    # merge and reshape operations

    def add_flatten_layer(self, prev_layer_name=None, exclude_from_path=True,
                          **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = {
            "layer": core.flatten(prev, **kwargs),
            "type": "Flatten", "categories": ("hidden", )
        }
        return self._add_layer(layer, exclude_from_path)

    def add_reshape_layer(self, new_shape, prev_layer_name=None,
                          exclude_from_path=True, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]['layer']
        layer = {
            "layer": core.reshape(prev, [-1, *new_shape], **kwargs),
            "type": "Reshape", "categories": ("hidden", )
        }
        return self._add_layer(layer, exclude_from_path)

    def add_merge_layer(self, prev_layer_names, merge_mode,
                        exclude_from_path=True, **kwargs):
        prev = [self._layers[name]['layer'] for name in prev_layer_names]
        layer = {
            "layer": merge.merge(prev, merge_mode, **kwargs),
            "type": "Merge", "categories": ("hidden", )
        }
        return self._add_layer(layer, exclude_from_path)

    def add_upsample2d_layer(self, window_size, prev_layer_name=None,
                             exclude_from_path=True, **kwargs):
        prev_name = prev_layer_name or self._curr_layer_name
        prev = self._layers[prev_name]
        layer = {
            "layer": conv.upsample_2d(prev, window_size, **kwargs),
            "type": "Upsample2D", "categories": ("hidden",)
        }
        return self._add_layer(layer, exclude_from_path)

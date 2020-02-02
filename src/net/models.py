import tflearn

import net.layer.layer_utils as layer_utils


class NetworkModel:

    def __init__(self, neural_network, **model_settings):
        self._net = neural_network
        self._hidden_models = {}
        settings = {
            'tensorboard_dir': model_settings.get('tb_dir',
                                                  '/tmp/tflearn_logs/'),
            'tensorboard_verbose': model_settings.get('tb_verbosity', 0)
        }
        output_layer = next(iter(neural_network.output_layer.values()))
        self._model = tflearn.DNN(output_layer['layer'], **settings)

    # properties

    @property
    def network_graph(self):
        return self._net

    @property
    def network_model(self):
        return self._model

    @property
    def hidden_output_layers(self):
        return set(self._hidden_models.keys())

    # layer weights and biases getters/setters

    def get_layer_weights(self, layer_name):
        layer = self._net.trainable_layers[layer_name]
        converter = layer_utils.weight_converters_external().get(
            layer['type'], lambda w: w)
        return converter(self._model.get_weights(layer['layer'].W))

    def set_layer_weights(self, layer_name, weights):
        layer = self._net.trainable_layers[layer_name]
        converter = layer_utils.weight_converters_internal().get(
            layer['type'], lambda w: w)
        self._model.set_weights(layer['layer'].W, converter(weights))

    def get_layer_biases(self, layer_name):
        layer = self._net.trainable_layers[layer_name]
        return self._model.get_weights(layer['layer'].b)

    def set_layer_biases(self, layer_name, weights):
        layer = self._net.trainable_layers[layer_name]
        self._model.set_weights(layer['layer'].b, weights)

    # load from/save to filesystem

    def load_from_file(self, model_filename, **optargs):
        w_only = optargs.get('weights_only', False)
        self._model.load(model_filename, weights_only=w_only)
        self._update_hidden_models()

    def save_to_file(self, model_filename):
        self._model.save(model_filename)

    # hidden layers output

    def enable_hidden_layer_output(self, layers):
        if isinstance(layers, str):
            new_layers = [layers]
        else:
            new_layers = layers
        hidden_models = self._hidden_models
        hidden_layers = self._net.hidden_layers
        invalid_layers = [layer for layer in new_layers
                          if layer not in hidden_layers]
        if len(invalid_layers) > 0:
            raise ValueError(f'Not a hidden layer: {invalid_layers}')
        new_models = ((layer, tflearn.DNN(hidden_layers[layer]['layer']))
                      for layer in new_layers if layer not in hidden_models)
        hidden_models.update(new_models)
        self._update_hidden_models()

    def get_hidden_layer_activations(self, input_data_dict):
        if len(self._hidden_models) == 0:
            raise Exception('Hidden layer activations not retrievable.')
        hidden_models = self._hidden_models
        return {layer_name: model.predict(input_data_dict)
                for layer_name, model in hidden_models.items()}

    # helper methods

    def _update_hidden_models(self):
        hidden_models = self._hidden_models
        session = self._model.session
        for layer_name, model in hidden_models.items():
            model.session.close()
            model.session = session
            model.trainer.session = session
            model.predictor.session = session

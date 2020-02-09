import itertools
import uuid
import random
import unittest

import numpy.testing as nptest

import net.models as models
import net.test.mocks as mocks


# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# can be rather slow.


class TestNeuralNetworkModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls, model=None):
        cls.graph = mocks.MockNeuralNetwork()
        cls.model = model or models.NetworkModel(cls.graph)
        cls.inputs_dict = cls.graph.example_input

    # helper methods (general)

    def _get_model_and_layers(self, model, layers, num_layers=1):
        model = model or self.model
        graph = model.network_graph
        layers = layers or random.sample(graph.trainable_layers.keys(),
                                         num_layers)
        if num_layers == 1:
            return model, layers[0]
        else:
            return model, layers

    @staticmethod
    def _get_unenabled_hidden_layers(model, num_layers=None):
        graph = model.network_graph
        layers = set(graph.hidden_layers.keys()) - model.hidden_output_layers
        num_layers = num_layers or random.randint(1, len(layers))
        return random.sample(layers, num_layers)

    # tests

    def test_get_weights(self, model=None, layer=None):
        model, layer = self._get_model_and_layers(model, layer)

        weights = model.get_layer_weights(layer)
        self.assertIsNotNone(weights)
        self.assertGreaterEqual(len(weights.shape), 1)

    def test_get_biases(self, model=None, layer=None):
        model, layer = self._get_model_and_layers(model, layer)

        biases = model.get_layer_biases(layer)
        self.assertIsNotNone(biases)

    def test_set_weights(self, model=None, layer=None):
        model, layer = self._get_model_and_layers(model, layer)

        weights = model.get_layer_weights(layer)
        weights.fill(random.randint(0, 100))
        model.set_layer_weights(layer, weights)
        nptest.assert_array_equal(model.get_layer_weights(layer), weights)

    def test_set_biases(self, model=None, layer=None):
        model, layer = self._get_model_and_layers(model, layer)

        biases = model.get_layer_biases(layer)
        biases.fill(random.randint(0, 100))
        model.set_layer_biases(layer, biases)
        nptest.assert_array_equal(model.get_layer_biases(layer), biases)

    def test_enable_hidden_layers(self, model=None, layers=None):
        # for this test, it is best to use a new model instance on every call
        model = model or models.NetworkModel(self.graph)
        layers = layers or self._get_unenabled_hidden_layers(model)
        if len(layers) == 1:
            arg = layers[0]
        else:
            arg = layers

        model.enable_hidden_layer_output(arg)
        self.assertSetEqual(model.hidden_output_layers, set(layers))

    def test_enable_invalid_hidden_layers(self, model=None):
        model = model or self.model
        layers = [uuid.uuid4() for index in range(random.randint(1, 10))]
        if len(layers) == 1:
            arg = layers[0]
        else:
            arg = layers
        self.assertRaises(ValueError, model.enable_hidden_layer_output, arg)

    def test_get_hidden_layer_output(self, model=None, inputs_dict=None):
        model = model or self.model
        inputs = inputs_dict or self.inputs_dict
        layers = model.hidden_output_layers
        if not layers:
            layers = self._get_unenabled_hidden_layers(model)
            model.enable_hidden_layer_output(layers)
            layers = model.hidden_output_layers

        hidden_outputs = model.get_hidden_layer_activations(inputs)
        output = model.network_model.predict(inputs)
        self.assertSetEqual(set(hidden_outputs.keys()), layers)
        for layer_name, hidden_output in hidden_outputs.items():
            self.assertIsNotNone(hidden_output)
            itertools.count()
            self.assertRaises(AssertionError, nptest.assert_array_equal,
                              output, hidden_output)


if __name__ == '__main__':
    unittest.main()

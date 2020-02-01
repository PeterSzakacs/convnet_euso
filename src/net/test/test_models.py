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
        cls.model = model or models.NetworkModel(mocks.MockNeuralNetwork())

    # helper methods (general)

    def _get_model_and_layer(self, model, layer):
        model = model or self.model
        graph = model.network_graph
        layer = layer or random.sample(graph.trainable_layers.keys(), 1)[0]
        return model, layer

    # tests

    def test_get_weights(self, model=None, layer=None):
        model, layer = self._get_model_and_layer(model, layer)

        weights = model.get_layer_weights(layer)
        self.assertIsNotNone(weights)
        self.assertGreaterEqual(len(weights.shape), 1)

    def test_get_biases(self, model=None, layer=None):
        model, layer = self._get_model_and_layer(model, layer)

        biases = model.get_layer_biases(layer)
        self.assertIsNotNone(biases)

    def test_set_weights(self, model=None, layer=None):
        model, layer = self._get_model_and_layer(model, layer)

        weights = model.get_layer_weights(layer)
        weights.fill(random.randint(0, 100))
        model.set_layer_weights(layer, weights)
        nptest.assert_array_equal(model.get_layer_weights(layer), weights)

    def test_set_biases(self, model=None, layer=None):
        model, layer = self._get_model_and_layer(model, layer)

        biases = model.get_layer_biases(layer)
        biases.fill(random.randint(0, 100))
        model.set_layer_biases(layer, biases)
        nptest.assert_array_equal(model.get_layer_biases(layer), biases)


if __name__ == '__main__':
    unittest.main()

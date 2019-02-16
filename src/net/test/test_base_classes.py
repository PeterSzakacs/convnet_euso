import unittest

from numpy.testing import assert_array_equal
from tflearn.layers import regression
from tflearn.layers.core import input_data, fully_connected

import net.base_classes as bclasses

# NOTE: These tests are rather slow for unit tests, but mocking out all of
# tflearn and tensorflow seemed like a really bad idea

class MockNeuralNetwork(bclasses.NeuralNetwork):

    @classmethod
    def get_instance(cls):
        # creating a neural network graph is also expensive, so...
        # just use a single instance (the graph is never changed anyway)
        if not hasattr(cls, '_the_net'):
            cls._the_net = MockNeuralNetwork()
        return cls._the_net

    def __init__(self):
        layers = []
        net = input_data(shape=(None, 3))
        net = fully_connected(net, 10, weights_init='zeros', bias_init='zeros')
        layers.append(net)
        net = fully_connected(net, 3, weights_init='zeros', bias_init='zeros')
        layers.append(net)
        net = regression(net)
        self._exp_num_layers = len(layers)
        self._exp_layers = layers
        self._exp_output = net
        super(MockNeuralNetwork, self).__init__(layers, net)

    def network_type(self):
        return 'test'


class TestNeuralNetwork(unittest.TestCase):

    def test_trainable_layer_order(self):
        network = MockNeuralNetwork.get_instance()
        self.assertEqual(network.trainable_layers, network._exp_layers)

    def test_output_layer(self):
        network = MockNeuralNetwork.get_instance()
        self.assertEqual(network.output_layer, network._exp_output)


class TestNeuralNetworkModel(unittest.TestCase):

    # helper methods (general)

    def _create_static_weights(self, model, value=10):
        tf_model = model.network_model
        tf_layers = model.network_graph.trainable_layers
        weights = tuple(tf_model.get_weights(layer.W) for layer in tf_layers)
        for weight in weights:
            weight.fill(value)
        return weights

    def _create_static_biases(self, model, value=10):
        tf_model = model.network_model
        tf_layers = model.network_graph.trainable_layers
        weights = tuple(tf_model.get_weights(layer.b) for layer in tf_layers)
        for weight in weights:
            weight.fill(value)
        return weights

    def _set_all_weights(self, model, new_weights):
        tf_model = model.network_model
        tf_layers = model.network_graph.trainable_layers
        for idx in range(len(tf_layers)):
            tf_model.set_weights(tf_layers[idx].W, new_weights[idx])

    def _set_all_biases(self, model, new_weights):
        tf_model = model.network_model
        tf_layers = model.network_graph.trainable_layers
        for idx in range(len(tf_layers)):
            tf_model.set_weights(tf_layers[idx].b, new_weights[idx])

    # helper methods (custom asserts)

    def _assert_weights_equal(self, values, exp_values):
        self.assertEqual(len(values), len(exp_values))
        for idx in range(len(values)):
            assert_array_equal(values[idx], exp_values[idx])

    def _assert_biases_equal(self, values, exp_values):
        self.assertEqual(len(values), len(exp_values))
        for idx in range(len(values)):
            assert_array_equal(values[idx], exp_values[idx])

    # test methods

    def test_layer_weights_snapshots_after_update(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        new_weights = self._create_static_weights(model)
        self._set_all_weights(model, new_weights)
        model.update_snapshots()
        self._assert_weights_equal(model.trainable_layer_weights_snapshot,
                                   new_weights)

    def test_layer_biases_snapshots_after_update(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        new_biases = self._create_static_biases(model)
        self._set_all_biases(model, new_biases)
        model.update_snapshots()
        self._assert_biases_equal(model.trainable_layer_biases_snapshot,
                                   new_biases)

    def test_layer_weights_snapshots_untouched_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        prev_snapshot = model.trainable_layer_weights_snapshot
        self._set_all_weights(model, self._create_static_weights(model))
        curr_snapshot = model.trainable_layer_weights_snapshot
        self._assert_weights_equal(prev_snapshot, curr_snapshot)

    def test_layer_biases_snapshots_untouched_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        prev_snapshot = model.trainable_layer_biases_snapshot
        self._set_all_biases(model, self._create_static_biases(model))
        curr_snapshot = model.trainable_layer_biases_snapshot
        self._assert_biases_equal(prev_snapshot, curr_snapshot)

    def test_displayed_layer_weigths_updated_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        prev_weights = model.trainable_layer_weights
        new_weights = self._create_static_weights(model)
        self._set_all_weights(model, new_weights)
        self._assert_weights_equal(model.trainable_layer_weights, new_weights)

    def test_displayed_layer_biases_updated_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        prev_biases = model.trainable_layer_biases
        new_biases = self._create_static_biases(model)
        self._set_all_biases(model, new_biases)
        self._assert_biases_equal(model.trainable_layer_biases, new_biases)

    def test_weights_restored_from_snapshot(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        prev_weights = model.trainable_layer_weights
        self._set_all_weights(model, self._create_static_weights(model))
        model.restore_from_snapshot()
        self._assert_weights_equal(model.trainable_layer_weights, prev_weights)

    def test_biases_restored_from_snapshot(self):
        network = MockNeuralNetwork.get_instance()
        model = bclasses.NetworkModel(network)

        prev_biases = model.trainable_layer_biases
        self._set_all_biases(model, self._create_static_biases(model))
        model.restore_from_snapshot()
        self._assert_biases_equal(model.trainable_layer_biases, prev_biases)


if __name__ == '__main__':
    unittest.main()

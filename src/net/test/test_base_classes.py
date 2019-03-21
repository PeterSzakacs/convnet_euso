import unittest

import numpy as np
from numpy.testing import assert_array_equal
from tflearn.layers import regression
from tflearn.layers.core import input_data, fully_connected

import net.base_classes as bclasses

# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# are rather slow.

class MockNeuralNetwork(bclasses.NeuralNetwork):

    @classmethod
    def get_instance(cls):
        # creating a neural network graph is also expensive, so...
        # just use a single instance (the graph is never changed anyway)
        if not hasattr(cls, '_the_net'):
            cls._the_net = MockNeuralNetwork()
        return cls._the_net

    def __init__(self):
        hidden, trainable = [], []

        net = input_data(shape=(None, 3))
        inputs = [net]
        net = fully_connected(net, 10, weights_init='zeros', bias_init='zeros')
        hidden.append(net); trainable.append(net)
        net = fully_connected(net, 3, weights_init='zeros', bias_init='zeros')
        trainable.append(net)
        net = regression(net)
        layers = {'trainable': trainable, 'hidden': hidden}
        super(MockNeuralNetwork, self).__init__(inputs, net, layers)
        self.all_w = {'FullyConnected': np.zeros((3, 10)),
                      'FullyConnected_1': np.zeros((10, 3))}
        self.all_b = {'FullyConnected': np.zeros(10, ),
                      'FullyConnected_1': np.zeros(3, )}
        self._exp_inputs = {'InputData': inputs[0]}
        self._exp_output = net
        self._exp_trainables = {'FullyConnected': trainable[0],
                                'FullyConnected_1': trainable[1]}
        self._exp_hidden = {'FullyConnected': trainable[0]}

    def network_type(self):
        return 'test'


class TestNeuralNetwork(unittest.TestCase):

    def test_input_layer_order(self, network=None):
        network = network or MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.input_layers, network._exp_inputs)

    def test_hidden_layer_order(self, network=None):
        network = network or MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.hidden_layers, network._exp_hidden)

    def test_trainable_layer_order(self, network=None):
        network = network or MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.trainable_layers, network._exp_trainables)

    def test_output_layer(self, network=None):
        network = network or MockNeuralNetwork.get_instance()
        self.assertEqual(network.output_layer, network._exp_output)


class TestNeuralNetworkModel(unittest.TestCase):

    # helper methods (general)

    def _fill_tensor_weights(self, weights, value=10):
        return {name: np.full(weight.shape, value)
                for name, weight in weights.items()}

    def _create_static_weights(self, weights, value=10):
        return self._fill_tensor_weights(weights, value=value)

    def _create_static_biases(self, biases, value=10):
        return self._fill_tensor_weights(biases, value=value)

    # helper methods (custom asserts)

    def _assert_weights_equal(self, values, exp_values):
        self.assertEqual(len(values), len(exp_values))
        self.assertSetEqual(set(values.keys()), set(exp_values.keys()))
        for key in exp_values.keys():
            assert_array_equal(
                values[key],
                exp_values[key],
                err_msg='"{}" layer weights not equal.'.format(key))

    def _assert_biases_equal(self, values, exp_values):
        self.assertEqual(len(values), len(exp_values))
        self.assertSetEqual(set(values.keys()), set(exp_values.keys()))
        for key in exp_values.keys():
            assert_array_equal(
                values[key],
                exp_values[key],
                err_msg='"{}" layer weights not equal.'.format(key))

    # test methods

    def test_get_initial_weights(self, model=None):
        model = model or bclasses.NetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_weights_equal(model.trainable_layer_weights, net.all_w)

    def test_get_initial_biases(self, model=None):
        model = model or bclasses.NetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_biases_equal(model.trainable_layer_biases, net.all_b)

    def test_set_weights(self, model=None):
        model = model or bclasses.NetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_weights = self._create_static_weights(net.all_w)
        model.trainable_layer_weights = new_weights
        self._assert_weights_equal(model.trainable_layer_weights, new_weights)

    def test_set_biases(self, model=None):
        model = model or bclasses.NetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_biases = self._create_static_biases(net.all_b)
        model.trainable_layer_biases = new_biases
        self._assert_biases_equal(model.trainable_layer_biases, new_biases)


if __name__ == '__main__':
    unittest.main()

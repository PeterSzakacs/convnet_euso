import unittest

import numpy as np
import numpy.testing as nptest

import net.models as models
import net.test.mocks as utils


@unittest.skip("API redesign in progress")
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
            nptest.assert_array_equal(
                values[key],
                exp_values[key],
                err_msg='"{}" layer weights not equal.'.format(key))

    def _assert_biases_equal(self, values, exp_values):
        self.assertEqual(len(values), len(exp_values))
        self.assertSetEqual(set(values.keys()), set(exp_values.keys()))
        for key in exp_values.keys():
            nptest.assert_array_equal(
                values[key],
                exp_values[key],
                err_msg='"{}" layer weights not equal.'.format(key))

    # test methods

    def test_get_initial_weights(self, model=None):
        model = model or models.NetworkModel(
            utils.MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_weights_equal(model.trainable_layer_weights, net.all_w)

    def test_get_initial_biases(self, model=None):
        model = model or models.NetworkModel(
            utils.MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_biases_equal(model.trainable_layer_biases, net.all_b)

    def test_set_weights(self, model=None):
        model = model or models.NetworkModel(
            utils.MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_weights = self._create_static_weights(net.all_w)
        model.trainable_layer_weights = new_weights
        self._assert_weights_equal(model.trainable_layer_weights, new_weights)

    def test_set_biases(self, model=None):
        model = model or models.NetworkModel(
            utils.MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_biases = self._create_static_biases(net.all_b)
        model.trainable_layer_biases = new_biases
        self._assert_biases_equal(model.trainable_layer_biases, new_biases)


if __name__ == '__main__':
    unittest.main()

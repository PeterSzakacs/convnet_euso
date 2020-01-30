import unittest

import net.test.mocks as mocks


@unittest.skip("API redesign in progress")
class TestNeuralNetwork(unittest.TestCase):

    def test_input_layer_order(self, network=None):
        network = network or mocks.MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.input_layers, network._exp_inputs)

    def test_input_mapping(self, network=None):
        network = network or mocks.MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.input_item_types,
                             network._exp_input_mapping)

    def test_hidden_layer_order(self, network=None):
        network = network or mocks.MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.hidden_layers, network._exp_hidden)

    def test_trainable_layer_order(self, network=None):
        network = network or mocks.MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.trainable_layers, network._exp_trainables)

    def test_output_layer(self, network=None):
        network = network or mocks.MockNeuralNetwork.get_instance()
        self.assertEqual(network.output_layer, network._exp_output)


if __name__ == '__main__':
    unittest.main()

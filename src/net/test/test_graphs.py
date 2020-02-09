import unittest

import net.test.mocks as mocks


# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# can be rather slow.


class TestNeuralNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls, network=None):
        cls.network = network or mocks.MockNeuralNetwork()

    def test_input_layers(self, network=None):
        network = network or self.network
        self.assertDictEqual(network.input_layers,
                             network.exp_inputs)

    def test_output_layer(self, network=None):
        network = network or self.network
        self.assertDictEqual(network.output_layer,
                             network.exp_output)

    def test_trainable_layers(self, network=None):
        network = network or self.network
        self.assertDictEqual(network.trainable_layers,
                             network.exp_trainable_layers)

    def test_hidden_layers(self, network=None):
        network = network or self.network
        self.assertDictEqual(network.hidden_layers,
                             network.exp_hidden_layers)

    def text_data_paths(self, network=None):
        network = network or self.network
        self.assertDictEqual(network.data_paths,
                             network.exp_paths)


if __name__ == '__main__':
    unittest.main()

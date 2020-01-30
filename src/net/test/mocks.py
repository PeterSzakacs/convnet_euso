import unittest

import numpy as np

import net.builders as bclasses
import net.graphs as graphs


# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# are rather slow.


class MockNeuralNetwork(graphs.NeuralNetwork):

    @classmethod
    def get_instance(cls):
        # creating a neural network graph is also expensive, so...
        # just use a single instance (the graph is never changed anyway)
        if not hasattr(cls, '_the_net'):
            cls._the_net = MockNeuralNetwork()
        return cls._the_net

    def __init__(self):
        builder = bclasses.GraphBuilder()
        in_name = builder.add_input_layer((3, ), 'test')
        fc1 = builder.add_fc_layer(10, weights_init='zeros', bias_init='zeros')
        fc2 = builder.add_fc_layer(3, weights_init='zeros', bias_init='zeros')
        builder.finalize(fc2)
        super(MockNeuralNetwork, self).__init__(builder)

        layers = builder.layers_dict
        self.all_w = {fc1: np.zeros((3, 10)), fc2: np.zeros((10, 3))}
        self.all_b = {fc1: np.zeros(10, ), fc2: np.zeros(3, )}
        self._exp_inputs = {in_name: layers[in_name]}
        self._exp_input_mapping = {in_name: 'test'}
        self._exp_output = layers[fc2]
        self._exp_trainables = {fc1: layers[fc1],
                                fc2: layers[fc2]}
        self._exp_hidden = {fc1: layers[fc1]}

    def network_type(self):
        return 'test'


if __name__ == '__main__':
    unittest.main()

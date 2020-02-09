import unittest

import numpy.random as nprand

import net.builders as builders
import net.graphs as graphs


class MockNeuralNetwork(graphs.NeuralNetwork):

    def __init__(self):
        builder = builders.GraphBuilder()
        input_layer = builder.add_input_layer((3, 3, 1), 'test')
        conv1 = builder.add_conv2d_layer(10, 3, filter_strides=1,
                                         weights_init='zeros',
                                         bias_init='zeros')
        fc1 = builder.add_fc_layer(10, weights_init='zeros',
                                   bias_init='zeros')
        fc2 = builder.add_fc_layer(3, weights_init='zeros',
                                   bias_init='zeros')
        builder.finalize(fc2)
        super(MockNeuralNetwork, self).__init__(builder)

        all_layers = builder.layers_dict
        self._in, self._out = input_layer, fc2
        self.exp_inputs = {input_layer: all_layers[input_layer]}
        self.exp_output = {fc2: all_layers[fc2]}
        self.exp_trainable_layers = {
            conv1: all_layers[conv1],
            fc1: all_layers[fc1],
            fc2: all_layers[fc2],
        }
        self.exp_hidden_layers = {
            conv1: all_layers[conv1],
            fc1: all_layers[fc1],
        }
        self.exp_paths = {
            input_layer: [conv1, fc1, fc2]
        }
        example_input = nprand.randint(low=0, high=255, size=3*3)
        self.example_input = {self._in: example_input.reshape((1, 3, 3, 1))}

    @property
    def network_type(self):
        return 'test'

    @property
    def input_spec(self):
        return {
            # item_type and location omitted for testing purposes
            self._in: {
                "shape": (3, 3, 1)
            }
        }

    @property
    def output_spec(self):
        return {
            # item_type and location omitted for testing purposes
            self._out: {
                "shape": (3, )
            }
        }


if __name__ == '__main__':
    unittest.main()

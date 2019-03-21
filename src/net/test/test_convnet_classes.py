import unittest

import numpy as np
from tflearn.layers import regression
from tflearn.layers.conv import conv_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.normalization import local_response_normalization

import net.test.test_base_classes as base_test
import net.convnet_classes as conv_classes

# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# are rather slow.

class MockNeuralNetwork(conv_classes.Conv2DNetwork):

    @classmethod
    def get_instance(cls):
        # creating a neural network graph is also expensive, so...
        # just use a single instance (the graph is never changed anyway)
        if not hasattr(cls, '_the_net'):
            cls._the_net = MockNeuralNetwork()
        return cls._the_net

    def __init__(self):
        conv_layers, fc_layers, filter_sizes = [], [], []
        hidden, trainable = [], []

        net = input_data(shape=(None, 10, 10, 1))
        inputs = {'test': net}
        net = conv_2d(net, 2, filter_size=(2, 3), weights_init='zeros',
                      bias_init='zeros')
        conv_layers.append(net); hidden.append(net); trainable.append(net)
        filter_sizes.append((2, 3, 1))
        net = conv_2d(net, 2, filter_size=2, weights_init='zeros',
                      bias_init='zeros')
        conv_layers.append(net); hidden.append(net); trainable.append(net)
        filter_sizes.append((2, 2, 2))
        net = local_response_normalization(net)
        hidden.append(net)

        # num_connections per neuron in fc_layer: 2*10*10 -> 200
        net = fully_connected(net, 3, weights_init='zeros', bias_init='zeros')
        fc_layers.append(net); trainable.append(net)
        net = regression(net)
        layers = {'trainable': trainable, 'hidden': hidden,
                  'conv2d': conv_layers, 'fc': fc_layers}
        super(MockNeuralNetwork, self).__init__(inputs, net, layers)
        self._exp_inputs = {'InputData': inputs['test']}
        self._exp_input_mapping = {'test': 'InputData'}
        self._exp_output = net
        self._exp_trainables = {'Conv2D': trainable[0],
                                'Conv2D_1': trainable[1],
                                'FullyConnected': trainable[2]}
        self._exp_hidden = {'Conv2D': hidden[0],
                            'Conv2D_1': hidden[1],
                            'LocalResponseNormalization': hidden[2]}
        self._exp_conv = {'Conv2D': conv_layers[0],
                          'Conv2D_1': conv_layers[1]}
        self._exp_fc = {'FullyConnected': fc_layers[0]}
        self._exp_filter_sizes = {'Conv2D': filter_sizes[0],
                                  'Conv2D_1': filter_sizes[1]}
        self.conv_w = {'Conv2D': np.zeros((2, 1, 2, 3)),
                       'Conv2D_1': np.zeros((2, 2, 2, 2))}
        self.conv_b = {'Conv2D': np.zeros(2, ), 'Conv2D_1': np.zeros(2, )}
        self.fc_w = {'FullyConnected': np.zeros((3, 200))}
        self.fc_b = {'FullyConnected': np.zeros(3, )}
        self.all_w = {**self.conv_w, **self.fc_w}
        self.all_b = {**self.conv_b, **self.fc_b}


class TestConv2DNetwork(base_test.TestNeuralNetwork):

    def test_input_layer_order(self):
        super().test_input_layer_order(
            network=MockNeuralNetwork.get_instance())

    def test_input_mapping(self):
        super().test_input_mapping(
            network=MockNeuralNetwork.get_instance())

    def test_hidden_layer_order(self):
        super().test_hidden_layer_order(
            network=MockNeuralNetwork.get_instance())

    def test_trainable_layer_order(self):
        super().test_trainable_layer_order(
            network=MockNeuralNetwork.get_instance())

    def test_output_layer(self):
        super().test_output_layer(network=MockNeuralNetwork.get_instance())

    def test_conv_layer_order(self):
        network = MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.conv_layers, network._exp_conv)

    def test_fc_layer_order(self):
        network = MockNeuralNetwork.get_instance()
        self.assertDictEqual(network.fc_layers, network._exp_fc)

    def test_get_filter_size(self):
        network = MockNeuralNetwork.get_instance()
        conv_layers = network.conv_layers
        for layer_name in conv_layers:
            self.assertTupleEqual(network.get_filter_size(layer_name),
                                  network._exp_filter_sizes[layer_name])


class TestConv2DNetworkModel(base_test.TestNeuralNetworkModel):

    # helper methods (general)

    def _create_static_conv_weights(self, weights, value=10):
        return self._fill_tensor_weights(weights, value=value)

    def _create_static_fc_weights(self, weights, value=10):
        return self._fill_tensor_weights(weights, value=value)

    def _create_static_conv_biases(self, biases, value=10):
        return self._fill_tensor_weights(biases, value=value)

    def _create_static_fc_biases(self, biases, value=10):
        return self._fill_tensor_weights(biases, value=value)

    # test methods

    def test_get_initial_weights(self):
        network = MockNeuralNetwork.get_instance()
        super().test_get_initial_weights(
            model=conv_classes.Conv2DNetworkModel(network))

    def test_get_initial_biases(self):
        network = MockNeuralNetwork.get_instance()
        super().test_get_initial_biases(
            model=conv_classes.Conv2DNetworkModel(network))

    def test_set_weights(self):
        network = MockNeuralNetwork.get_instance()
        super().test_set_weights(
            model=conv_classes.Conv2DNetworkModel(network))

    def test_set_biases(self):
        network = MockNeuralNetwork.get_instance()
        super().test_set_biases(
            model=conv_classes.Conv2DNetworkModel(network))

    def test_get_initial_conv_weights(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_weights_equal(model.conv_weights, net.conv_w)

    def test_get_initial_conv_biases(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_weights_equal(model.conv_biases, net.conv_b)

    def test_set_conv_weigths(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_weights = self._create_static_conv_weights(net.conv_w)
        model.conv_weights = new_weights
        self._assert_weights_equal(model.conv_weights, new_weights)

    def test_set_conv_biases(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_biases = self._create_static_conv_biases(net.conv_b)
        model.conv_biases = new_biases
        self._assert_biases_equal(model.conv_biases, new_biases)

    def test_get_initial_fc_weights(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_weights_equal(model.fc_weights, net.fc_w)

    def test_get_initial_fc_biases(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph
        self._assert_biases_equal(model.fc_biases, net.fc_b)

    def test_set_fc_weigths(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_weights = self._create_static_fc_weights(net.fc_w)
        model.fc_weights = new_weights
        self._assert_weights_equal(model.fc_weights, new_weights)

    def test_set_fc_biases(self, model=None):
        model = model or conv_classes.Conv2DNetworkModel(
            MockNeuralNetwork.get_instance())
        net = model.network_graph

        new_biases = self._create_static_fc_biases(net.fc_b)
        model.fc_biases = new_biases
        self._assert_biases_equal(model.fc_biases, new_biases)


# def divide_list_into_chunks(lst, n_chunks):
#     list_len = len(method_list)
#     chunk_sizes = [int(list_len/n_chunks) for idx in range(n_chunks)]
#     chunk_remainder = list_len % n_chunks
#     while chunk_remainder > 0:
#         chunk_sizes[chunk_remainder] += 1
#         chunk_remainder -= 1
#     start_idx = 0
#     lsts = []
#     for chunk_size in chunk_sizes:
#         lsts.append(lst[start_idx:(start_idx + chunk_size)])
#         start_idx += chunk_size
#     return lsts


# def run_suite(methods_list):
#     loader = unittest.TestLoader()
#     suite = loader.loadTestsFromNames(methods_list)
#     unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    import multiprocessing
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestConv2DNetwork)
    # runner = unittest.TextTestRunner().run(suite)

    # module_class = 'net.test.test_convnet_classes.TestConv2DNetworkModel'
    # method_list = [
    #     '{}.{}'.format(module_class, func)
    #     for func in dir(TestConv2DNetworkModel)
    #     if func.startswith('test_') and
    #     callable(getattr(TestConv2DNetworkModel, func))]
    # n_cpu = multiprocessing.cpu_count()
    # chunks = divide_list_into_chunks(method_list, n_cpu)
    # for chunk in chunks:
    #     p = multiprocessing.Process(target=run_suite, args=(chunk, ))
    #     p.start()

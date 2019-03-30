import unittest

import numpy as np

import net.test.test_base_classes as base_test
import net.base_classes as base_classes
import net.convnet_classes as conv_classes

# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# are rather slow.

class MockNeuralNetwork(conv_classes.Conv2DNetwork):

    @classmethod
    def get_instance(cls):
        # creating a neural network graph is also expensive, so...
        # just use a single instance per test process (the graph is
        # never changed anyway)
        if not hasattr(cls, '_the_net'):
            cls._the_net = MockNeuralNetwork()
        return cls._the_net

    def __init__(self):
        builder = base_classes.GraphBuilder()
        in_name = builder.add_input_layer((10, 10, 1), 'test')
        conv1 = builder.add_conv2d_layer(2, (2, 3), weights_init='zeros',
                                         bias_init='zeros')
        conv2 = builder.add_conv2d_layer(2, 2, weights_init='zeros',
                                         bias_init='zeros')
        lrn = builder.add_lrn_layer()
        # num_connections per neuron in fc_layer: 2*10*10 -> 200
        out = builder.add_fc_layer(3, weights_init='zeros', bias_init='zeros')
        builder.finalize(out)
        super(MockNeuralNetwork, self).__init__(builder)

        layers = builder.layers_dict

        self._exp_inputs = {in_name: layers[in_name]}
        self._exp_input_mapping = {in_name: 'test'}
        self._exp_output = layers[out]
        self._exp_trainables = {conv1: layers[conv1],
                                conv2: layers[conv2],
                                out: layers[out]}
        self._exp_hidden = {conv1: layers[conv1],
                            conv2: layers[conv2],
                            lrn: layers[lrn]}
        self._exp_conv = {conv1: layers[conv1],
                          conv2: layers[conv2]}
        self._exp_fc = {out: layers[out]}
        self._exp_filter_sizes = {conv1: (2, 3, 1),
                                  conv2: (2, 2, 2)}
        self.conv_w = {conv1: np.zeros((2, 1, 2, 3)),
                       conv2: np.zeros((2, 2, 2, 2))}
        self.conv_b = {conv1: np.zeros(2, ), conv2: np.zeros(2, )}
        self.fc_w = {out: np.zeros((3, 200))}
        self.fc_b = {out: np.zeros(3, )}
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


def divide_list_into_chunks(lst, n_chunks):
    list_len = len(method_list)
    chunk_sizes = [int(list_len/n_chunks) for idx in range(n_chunks)]
    chunk_remainder = list_len % n_chunks
    while chunk_remainder > 0:
        chunk_sizes[chunk_remainder] += 1
        chunk_remainder -= 1
    start_idx = 0
    lsts = []
    for chunk_size in chunk_sizes:
        lsts.append(lst[start_idx:(start_idx + chunk_size)])
        start_idx += chunk_size
    return lsts


def run_suite(methods_list):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(methods_list)
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    import multiprocessing
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConv2DNetwork)
    runner = unittest.TextTestRunner().run(suite)

    module_class = 'net.test.test_convnet_classes.TestConv2DNetworkModel'
    method_list = [
        '{}.{}'.format(module_class, func)
        for func in dir(TestConv2DNetworkModel)
        if func.startswith('test_') and
        callable(getattr(TestConv2DNetworkModel, func))]
    n_cpu = multiprocessing.cpu_count()
    chunks = divide_list_into_chunks(method_list, n_cpu)
    for chunk in chunks:
        p = multiprocessing.Process(target=run_suite, args=(chunk, ))
        p.start()

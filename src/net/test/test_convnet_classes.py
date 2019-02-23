import unittest

from tflearn.layers import regression
from tflearn.layers.conv import conv_2d
from tflearn.layers.core import input_data, fully_connected

import net.test.test_base_classes as base_test
import net.convnet_classes as conv_classes

# NOTE: Since the classes under test here are 3rd party library wrappers, these
# tests are much more integration tests than unit tests. Because of this, they
# are rather slow (multiple minutes without multi-process execution).

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
        net = input_data(shape=(None, 10, 10, 1))

        net = conv_2d(net, 2, filter_size=(2, 3), weights_init='zeros',
                      bias_init='zeros')
        conv_layers.append(net)
        filter_sizes.append((2, 3, 1))
        net = conv_2d(net, 2, filter_size=2, weights_init='zeros',
                      bias_init='zeros')
        conv_layers.append(net)
        filter_sizes.append((2, 2, 2))

        net = fully_connected(net, 3, weights_init='zeros', bias_init='zeros')
        fc_layers.append(net)
        net = regression(net)
        self._exp_layers = conv_layers + fc_layers
        self._exp_conv = conv_layers
        self._exp_fc = fc_layers
        self._exp_output = net
        self._exp_filter_sizes = filter_sizes
        super(MockNeuralNetwork, self).__init__(conv_layers, fc_layers, net)


class TestConv2DNetwork(base_test.TestNeuralNetwork):

    def test_conv_layer_order(self):
        network = MockNeuralNetwork.get_instance()
        self.assertEqual(network.conv_layers, network._exp_conv)

    def test_fc_layer_order(self):
        network = MockNeuralNetwork.get_instance()
        self.assertEqual(network.fc_layers, network._exp_fc)

    def test_get_filter_size(self):
        network = MockNeuralNetwork.get_instance()
        conv_layers = network.conv_layers
        for idx in range(len(conv_layers)):
            self.assertTupleEqual(network.get_filter_size(conv_layers[idx]),
                                  network._exp_filter_sizes[idx])


class TestConv2DNetworkModel(base_test.TestNeuralNetworkModel):

    # helper methods (general)

    def _create_static_conv_weights(self, model, value=10):
        tf_model = model.network_model
        tf_layers = model.network_graph.conv_layers
        weights = tuple(tf_model.get_weights(layer.W) for layer in tf_layers)
        return self._fill_tensor_weights(weights, value=value)

    def _create_static_fc_weights(self, model, value=10):
        tf_model = model.network_model
        tf_layers = model.network_graph.fc_layers
        weights = tuple(tf_model.get_weights(layer.W) for layer in tf_layers)
        return self._fill_tensor_weights(weights, value=value)

    def _create_static_conv_biases(self, model, value=10):
        tf_model = model.network_model
        tf_layers = model.network_graph.conv_layers
        weights = tuple(tf_model.get_weights(layer.b) for layer in tf_layers)
        return self._fill_tensor_weights(weights, value=value)

    def _create_static_fc_biases(self, model, value=10):
        tf_model = model.network_model
        tf_layers = model.network_graph.fc_layers
        weights = tuple(tf_model.get_weights(layer.b) for layer in tf_layers)
        return self._fill_tensor_weights(weights, value=value)

    # test methods

    def test_conv_weights_snapshots_after_update(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        new_weights = self._create_static_conv_weights(model)
        model.conv_weights = new_weights
        model.update_snapshots()
        self._assert_weights_equal(model.conv_weights_snapshot,
                                   new_weights)

    def test_conv_biases_snapshots_after_update(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        new_biases = self._create_static_conv_biases(model)
        model.conv_biases = new_biases
        model.update_snapshots()
        self._assert_biases_equal(model.conv_biases_snapshot,
                                  new_biases)

    def test_conv_weights_snapshots_untouched_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_snapshot = model.conv_weights_snapshot
        model.conv_weights = self._create_static_conv_weights(model)
        curr_snapshot = model.conv_weights_snapshot
        self._assert_weights_equal(prev_snapshot, curr_snapshot)

    def test_conv_biases_snapshots_untouched_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_snapshot = model.conv_biases_snapshot
        model.conv_biases = self._create_static_conv_biases(model)
        curr_snapshot = model.conv_biases_snapshot
        self._assert_biases_equal(prev_snapshot, curr_snapshot)

    def test_displayed_conv_weigths_updated_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_weights = model.trainable_layer_weights
        new_weights = self._create_static_conv_weights(model)
        model.conv_weights = new_weights
        self._assert_weights_equal(model.conv_weights, new_weights)

    def test_displayed_conv_biases_updated_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_biases = model.trainable_layer_biases
        new_biases = self._create_static_conv_biases(model)
        model.conv_biases = new_biases
        self._assert_biases_equal(model.conv_biases, new_biases)

    def test_conv_weights_restored_from_snapshot(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_weights = model.conv_weights
        model.conv_weights = self._create_static_conv_weights(model)
        model.restore_from_snapshot()
        self._assert_weights_equal(model.conv_weights, prev_weights)

    def test_conv_biases_restored_from_snapshot(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_biases = model.conv_biases
        model.conv_biases = self._create_static_conv_biases(model)
        model.restore_from_snapshot()
        self._assert_biases_equal(model.conv_biases, prev_biases)

    def test_fc_weights_snapshots_after_update(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        new_weights = self._create_static_fc_weights(model)
        model.fc_weights = new_weights
        model.update_snapshots()
        self._assert_weights_equal(model.fc_weights_snapshot,
                                   new_weights)

    def test_fc_biases_snapshots_after_update(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        new_biases = self._create_static_fc_biases(model)
        model.fc_biases = new_biases
        model.update_snapshots()
        self._assert_biases_equal(model.fc_biases_snapshot,
                                   new_biases)

    def test_fc_weights_snapshots_untouched_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_snapshot = model.fc_weights_snapshot
        model.fc_weights = self._create_static_fc_weights(model)
        curr_snapshot = model.fc_weights_snapshot
        self._assert_weights_equal(prev_snapshot, curr_snapshot)

    def test_fc_biases_snapshots_untouched_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_snapshot = model.fc_biases_snapshot
        model.fc_biases = self._create_static_fc_biases(model)
        curr_snapshot = model.fc_biases_snapshot
        self._assert_biases_equal(prev_snapshot, curr_snapshot)

    def test_displayed_fc_weigths_updated_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_weights = model.fc_weights
        new_weights = self._create_static_fc_weights(model)
        model.fc_weights = new_weights
        self._assert_weights_equal(model.fc_weights, new_weights)

    def test_displayed_fc_biases_updated_after_changing_model(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_biases = model.fc_biases
        new_biases = self._create_static_fc_biases(model)
        model.fc_biases = new_biases
        self._assert_biases_equal(model.fc_biases, new_biases)

    def test_fc_weights_restored_from_snapshot(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_weights = model.fc_weights
        model.fc_weights = self._create_static_fc_weights(model)
        model.restore_from_snapshot()
        self._assert_weights_equal(model.fc_weights, prev_weights)

    def test_fc_biases_restored_from_snapshot(self):
        network = MockNeuralNetwork.get_instance()
        model = conv_classes.Conv2DNetworkModel(network)

        prev_biases = model.fc_biases
        model.fc_biases = self._create_static_fc_biases(model)
        model.restore_from_snapshot()
        self._assert_biases_equal(model.fc_biases, prev_biases)


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

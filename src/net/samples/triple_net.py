
# first triple-input convolutional network, derived from github_net3b

import net.builders as builders
import net.graphs as graphs
import net.models as models


def create_model(input_shapes, **optsettings):
    network = TripleNet(input_shapes, **optsettings)
    return models.NetworkModel(network, **optsettings)


class TripleNet(graphs.NeuralNetwork):

    def __init__(self, input_shapes, **optsettings):
        lr = optsettings.get('learning_rate') or 0.001
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        builder = builders.GraphBuilder()

        shape = input_shapes['yx']
        builder.add_input_layer(shape, name='input_yx')
        builder.start_new_path()
        builder.add_reshape_layer((*shape, 1))
        builder.add_conv2d_layer(32, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        builder.add_conv2d_layer(64, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        yx_out = builder.add_flatten_layer()
        builder.end_current_path()

        shape = input_shapes['gtux']
        builder.add_input_layer(shape, name='input_gtux')
        builder.start_new_path()
        builder.add_reshape_layer((*shape, 1))
        builder.add_conv2d_layer(32, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        builder.add_conv2d_layer(64, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        gtux_out = builder.add_flatten_layer()
        builder.end_current_path()

        shape = input_shapes['gtuy']
        builder.add_input_layer(shape, name='input_gtuy')
        builder.start_new_path()
        builder.add_reshape_layer((*shape, 1))
        builder.add_conv2d_layer(32, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        builder.add_conv2d_layer(64, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        gtuy_out = builder.add_flatten_layer()
        builder.end_current_path()

        builder.add_merge_layer((yx_out, gtux_out, gtuy_out),
                                'concat')
        builder.start_new_path()
        builder.add_fc_layer(128, activation='relu')
        builder.add_dropout_layer(0.5)
        builder.add_fc_layer(50, activation='relu')
        builder.add_dropout_layer(0.5)
        out_name = builder.add_fc_layer(2, activation='softmax')
        builder.end_current_path()
        builder.finalize(out_name, name='target', learning_rate=lr,
                         optimizer=optimizer, loss=loss_fn)
        super(self.__class__, self).__init__(builder)
        self.input_shapes = input_shapes.copy()
        self.out_name = out_name

    @property
    def network_type(self):
        return 'classifier'

    @property
    def input_spec(self):
        return {
            "input_yx": {
                "shape": self.input_shapes['yx'],
                "item_type": 'yx',
                "location": "data"
            },
            "input_gtux": {
                "shape": self.input_shapes['gtux'],
                "item_type": 'gtux',
                "location": "data"
            },
            "input_gtuy": {
                "shape": self.input_shapes['gtuy'],
                "item_type": 'gtuy',
                "location": "data"
            }
        }

    @property
    def output_spec(self):
        return {
            self.out_name: {
                "item_type": "classification",
                "location": "targets"
            }
        }

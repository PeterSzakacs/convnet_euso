
# Original MNIST network changed (fully connected layers, different number of
# filters in convolutional layers)

import net.builders as builders
import net.graphs as graphs
import net.models as models


def create_model(input_shapes, **optsettings):
    network = MNISTNet2(input_shapes, **optsettings)
    return models.NetworkModel(network, **optsettings)


class MNISTNet2(graphs.NeuralNetwork):

    def __init__(self, input_shapes, input_type='yx', **optsettings):
        lr = optsettings.get('learning_rate') or 0.01
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        builder = builders.GraphBuilder()
        shape = input_shapes[input_type]
        builder.add_input_layer(shape, input_type, name='input')
        builder.start_new_path()
        builder.add_reshape_layer((*shape, 1))
        builder.add_conv2d_layer(64, 3, activation='relu', regularizer="L2")
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        builder.add_conv2d_layer(64, 3, activation='relu', regularizer="L2")
        builder.add_maxpool2d_layer(2)
        builder.add_lrn_layer()
        builder.add_fc_layer(256, activation='relu')
        builder.add_dropout_layer(0.8)
        builder.add_fc_layer(256, activation='relu')
        builder.add_dropout_layer(0.8)
        out_name = builder.add_fc_layer(2, activation='softmax')
        builder.end_current_path()
        builder.finalize(out_name, name='target', learning_rate=lr,
                         optimizer=optimizer, loss=loss_fn)
        super(self.__class__, self).__init__(builder)
        self.input_type = input_type
        self.input_shape = input_shapes[input_type]
        self.out_name = out_name

    @property
    def network_type(self):
        return 'classifier'

    @property
    def input_spec(self):
        return {
            "input": {
                "shape": self.input_shape,
                "item_type": self.input_type,
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

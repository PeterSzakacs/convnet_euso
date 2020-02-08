import net.builders as builders
import net.graphs as graphs
import net.models as models


def create_model(input_shapes, **optsettings):
    network = TestAutoencoder(input_shapes, **optsettings)
    return models.AutoEncoderModel(network, **optsettings)


class TestAutoencoder(graphs.AutoEncoder):

    def __init__(self, input_shapes, input_type='yx', **optsettings):
        lr = optsettings.get('learning_rate') or 0.001
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        builder = builders.GraphBuilder()
        shape = input_shapes[input_type]
        in_name = builder.add_input_layer(shape, input_type, name='input')
        builder.start_new_path()
        builder.add_reshape_layer((*shape, 1))
        builder.add_conv2d_layer(10, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        encoder = builder.add_conv2d_layer(10, 3, filter_strides=1,
                                           activation='relu', regularizer='L2',
                                           padding='same')
        builder.add_conv2d_layer(1, 3, filter_strides=1, activation='relu',
                                 regularizer='L2', padding='same')
        decoder = builder.add_reshape_layer(shape)
        builder.end_current_path()
        builder.finalize(decoder, name='target', learning_rate=lr,
                         optimizer=optimizer, loss=loss_fn)
        super(self.__class__, self).__init__(builder, encoder)
        self.input_type = input_type
        self.input_shape = input_shapes[input_type]
        self.in_name, self.out_name = in_name, decoder

    @property
    def input_spec(self):
        return {
            self.in_name: {
                "shape": self.input_shape,
                "item_type": self.input_type,
                "location": "data"
            }
        }

    @property
    def output_spec(self):
        return {
            self.out_name: {
                "item_type": self.input_type,
                "location": "data"
            }
        }

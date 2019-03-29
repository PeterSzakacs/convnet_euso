
# Original MNIST network changed (fully connected layers, different number of
# filters in convolutional layers)

import net.base_classes as base_classes
import net.convnet_classes as conv_classes


def create(input_shapes, **optsettings):
    network = MNISTNet2(input_shapes, **optsettings)
    return network.output_layer, network.conv_layers, network.fc_layers


def create_model(input_shapes, **optsettings):
    network = MNISTNet2(input_shapes, **optsettings)
    return conv_classes.Conv2DNetworkModel(network, **optsettings)


class MNISTNet2(conv_classes.Conv2DNetwork):

    def __init__(self, input_shapes, input_type='yx', **optsettings):
        lr = optsettings.get('learning_rate') or 0.01
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        builder = base_classes.GraphBuilder()
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

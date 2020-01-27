
# based on: https://github.com/nlinc1905/Particle-Identification-Neural-Net
# (conv_classifier_2.py)

import net.base_classes as base_classes


def create_model(input_shapes, **optsettings):
    network = GithubNet(input_shapes, **optsettings)
    return base_classes.NetworkModel(network, **optsettings)


class GithubNet(base_classes.NeuralNetwork):

    def network_type(self):
        return 'classifier'

    def __init__(self, input_shapes, input_type='yx', **optsettings):
        lr = optsettings.get('learning_rate') or 0.001
        optimizer = optsettings.get('optimizer') or 'adam'
        loss_fn = optsettings.get('loss_fn') or 'categorical_crossentropy'
        builder = base_classes.GraphBuilder()
        shape = input_shapes[input_type]
        builder.add_input_layer(shape, input_type, name='input')
        builder.start_new_path()
        builder.add_reshape_layer((*shape, 1))
        builder.add_conv2d_layer(64, 3, filter_strides=3, activation='relu',
                                 padding='valid')
        builder.add_maxpool2d_layer(2)
        builder.add_conv2d_layer(64, 3, filter_strides=3, activation='relu',
                                 padding='valid')
        builder.add_maxpool2d_layer(2)
        builder.add_dropout_layer(0.3)
        builder.add_flatten_layer()
        builder.add_fc_layer(128, activation='relu')
        builder.add_dropout_layer(0.5)
        builder.add_fc_layer(50, activation='relu')
        out_name = builder.add_fc_layer(2, activation='softmax')
        builder.end_current_path()
        builder.finalize(out_name, name='target', learning_rate=lr,
                         optimizer=optimizer, loss=loss_fn)
        super(self.__class__, self).__init__(builder)

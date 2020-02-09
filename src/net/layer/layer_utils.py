import net.layer.conv2d as conv2d
import net.layer.fc as fc


def weight_converters_external():
    return {
        "Conv2D": conv2d.convert_weights_external,
        "FC": fc.convert_weights_external
    }


def weight_converters_internal():
    return {
        "Conv2D": conv2d.convert_weights_internal,
        "FC": fc.convert_weights_internal
    }

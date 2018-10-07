import numpy as np

"""Class providing commonly used functionality for handling datasets of neural networks in the form
of packets and their various transformed forms (all represented currently as numpy arrays).

Currentyl, there are 2 categories of functionality supported

-- packet conversion --
Converts packets into projections of maximum values along a certain axis or creates subpackets
(continuous range of frames from the original packet ordered with respect to time or GTU). All
methods support the use of start_idx and end_idx that have similar semantics to the 'start' and
'stop' attributes of a Python range(). They select a range of frames along the GTU axis to be
used in creating subpackets and projections (this also alters the size of arrays along the GTU
axis of GTU X and GTU Y projections as well as the)

-- packet holder creation --
A 'holder' represents a data structure (specifically a numpy.ndarray) used for storing (sub)packets
or packet projections. The second argument in the relevant methods represents the expected dimensions
of packets (in the form of an integer tuple with 3 elements) from which packet projections are created
or which are stored directly as subpackets. This is in case that the items to be stored in the dataset
are created from a subrange of frames on the source packets.

-- properties --

Properties named output_* are boolean flags representing which types of dataset items the caller wishes
to work with. In the case of packet conversions, these flags say which packet conversions are to be
applied when calling convert_packet on a given packet, and in the case of packet holder creation, they
specify the type of packet holders to create.

Multiple types of items are possible, in which case, a tuple of items is returned, ordered the following way:

packet_conversion:
((sub)packet, y x projection, gtu x projection, gtu y projection)
holder_creation:
((sub) packet holder, y x projection holder, gtu x projection holder, gtu y projection holder)

If a flag for a particular output type is set to False, the item is ommited from this tuple entirely,
e.g. for output_raw and output_gtux set to False, and others set to True, the result of calling
convert_packet on a given packet would be:

(y x projection, gtu y projection)
"""
class numpy_dataset_helper:

    def __init__(self, output_raw=True, output_yx=False,
                    output_gtux=False, output_gtuy=False):
        self._raw = output_raw
        self._yx = output_yx
        self._gtux = output_gtux
        self._gtuy = output_gtuy
        self._holder_creators = (self.create_subpacket_holder, self.create_y_x_projection_holder,
                                    self.create_gtu_x_projection_holder, self.create_gtu_y_projection_holder)
        self._packet_converters = (self.create_subpacket, self.create_y_x_projection,
                                    self.create_gtu_x_projection, self.create_gtu_y_projection)
        self._set_current_holder_creators()
        self._set_current_packet_converters()

    def _set_current_holder_creators(self):
        outputs = (self._raw, self._yx, self._gtux, self._gtuy)
        self._curr_creators = tuple(self._holder_creators[idx] for idx in range(4) if outputs[idx] == True)

    def _set_current_packet_converters(self):
        outputs = (self._raw, self._yx, self._gtux, self._gtuy)
        self._curr_converters = tuple(self._packet_converters[idx] for idx in range(4) if outputs[idx] == True)

    # properties (output types)

    @property
    def output_raw(self):
        return self._raw

    @output_raw.setter
    def output_raw(self, value):
        self._raw = value
        self._set_current_holder_creators()
        self._set_current_packet_converters()

    @property
    def output_yx(self):
        return self._yx

    @output_yx.setter
    def output_yx(self, value):
        self._yx = value
        self._set_current_holder_creators()
        self._set_current_packet_converters()

    @property
    def output_gtux(self):
        return self._gtux

    @output_gtux.setter
    def output_gtux(self, value):
        self._gtux = value
        self._set_current_holder_creators()
        self._set_current_packet_converters()

    @property
    def output_gtuy(self):
        return self._gtuy

    @output_gtuy.setter
    def output_gtuy(self, value):
        self._gtuy = value
        self._set_current_holder_creators()
        self._set_current_packet_converters()

    # projection and subpacket creation from packet

    def create_subpacket(self, packet, start_idx=0, end_idx=None):
        return packet[start_idx:end_idx]

    def create_y_x_projection(self, packet, start_idx=0, end_idx=None):
        return np.max(packet[start_idx:end_idx], axis=0)

    def create_gtu_x_projection(self, packet, start_idx=0, end_idx=None):
        return np.max(packet[start_idx:end_idx], axis=1)

    def create_gtu_y_projection(self, packet, start_idx=0, end_idx=None):
        return np.max(packet[start_idx:end_idx], axis=2)

    def convert_packet(self, packet, start_idx=0, end_idx=None):
        return tuple(converter(packet, start_idx=start_idx, end_idx=end_idx)
                        for converter in self._curr_converters)

    # holder creation

    def create_subpacket_holder(self, num_packets, packet_shape):
        n_f, f_h, f_w = packet_shape[0:3]
        return np.empty((num_packets, n_f, f_h, f_w))

    def create_y_x_projection_holder(self, num_packets, packet_shape):
        n_f, f_h, f_w = packet_shape[0:3]
        return np.empty((num_packets, f_h, f_w))

    def create_gtu_x_projection_holder(self, num_packets, packet_shape):
        n_f, f_h, f_w = packet_shape[0:3]
        return np.empty((num_packets, n_f, f_w))

    def create_gtu_y_projection_holder(self, num_packets, packet_shape):
        n_f, f_h, f_w = packet_shape[0:3]
        return np.empty((num_packets, n_f, f_h))

    def create_converted_packets_holders(self, num_packets, packet_shape):
        return tuple(creator(num_packets, packet_shape) for creator in self._curr_creators)


"""Shuffle dataset data and their targets in unison for a given number of times.

Parameters
----------
num_shuffles :  int
                number of times the data and targets are to be shuffled
dataset :       tuple of numpy.ndarray
                the dataset in the form of a tuple of numpy arrays
targets :       numpy.ndarray
                the expected classification outputs (targets) for a neural network
"""
def shuffle_dataset(num_shuffles, dataset, targets):
    for idx in range(num_shuffles):
        rng_state = np.random.get_state()
        for idx in range(len(dataset)):
            np.random.shuffle(dataset[idx])
            np.random.set_state(rng_state)
        np.random.shuffle(targets)

# save dataset data and targets
def save_dataset(dataset, targets, dataset_filenames, targets_filename):
    for idx in range(len(dataset)):
        np.save(dataset_filenames[idx], dataset[idx])
    np.save(targets_filename, targets)
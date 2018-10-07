import sys
import operator
import random as rand

import numpy as np

import utils.common_utils as cutils
import utils.dataset_utils as ds
import utils.synth_data_utils as sdutils


class simulated_data_generator():

    def __init__(self, shower_template, bg_lambda=1, bad_ECs_range=(0, 0),
                 dataset_helper=ds.numpy_dataset_helper()):
        self.shower_template = shower_template
        self.bg_lambda = bg_lambda
        self.bad_ECs_range = bad_ECs_range
        self._helper = dataset_helper

    # generator properties

    """Template for simulated shower.

    Determines the range of values of start coordinates, number of frames
    the shower line spans and its intensity relative to the background.
    """
    @property
    def shower_template(self):
        return self._template

    @shower_template.setter
    def shower_template(self, value):
        if not isinstance(value, sdutils.simulated_shower_template):
            raise TypeError(('Wrong template type, cannot use {} as shower '
                            ' template').format(type(value)))
        self._template = value

    """Helper class to create appropriate output types as well as data
    structures to store them in (holders).

    Determines which type of items (projections or packets) to create and
    store for the dataset. In case the caller wants to change output types,
    just get this property and set the flags on the object as appropriate.
    """
    @property
    def dataset_helper(self):
        return self._helper

    """Tuple of 2 integers (MIN, MAX) representing how many EC units within
    the data should haved simulated malfunctions.

    The effect of EC malfunction is simulated for half of all items in the
    dataset, where the actual number of these units per frame is from MIN
    to MAX inclusive. MIN == MAX impies a constant number of malfunctioned
    EC units per frame. MAX can not bet more than the total number of ECs
    per frame. Note also that in the case of packets with simulated showers,
    some EC units are not allowed to be malfunctioned. Therefore, even if
    this property were set to (num ECs, num ECs), data items with smulated
    showers shall never have every EC unit malfunctioned.
    """
    @property
    def bad_ECs_range(self):
        return self._bad_ECs

    @bad_ECs_range.setter
    def bad_ECs_range(self, value):
        interval = cutils.check_and_convert_value_to_tuple(value,
                                                           'bad_ECs_range')
        cutils.check_interval_tuple(interval, 'bad_ECs_range', 0,
                                    self._template.packet_template.num_EC)
        self._bad_ECs = interval
        EC_min, EC_max = interval[0:2]
        self._bad_ECs_gen = ((lambda: EC_min) if EC_min == EC_max else
                             (lambda: rand.randint(EC_min, EC_max)))

    # methods

    # TODO: might want to break up these methods and possibly move them
    # to different modules as well
    def create_shower_packet(self, yx_angle, max_EC_malfunctions=0):
        # create the actual packet
        packet_template = self._template.packet_template
        packet = np.random.poisson(lam=self.bg_lambda,
                                   size=packet_template.packet_shape)
        GTU, Y, X, vals = sdutils.create_simu_shower_line_from_template(
            self._template, yx_angle
        )
        packet[GTU, Y, X] += vals

        # get the sum of shower pixel values in all EC modules
        ECs_used = [packet_template.xy_to_ec_idx(x, y) for (x, y) in zip(X, Y)]
        ECs_dict = dict(zip(ECs_used, [0] * len(ECs_used)))
        for idx in range(len(ECs_used)):
            EC = ECs_used[idx]
            ECs_dict[EC] += packet[GTU[idx], Y[idx], X[idx]]
        # get the EC containing the maximum sum of pixel values
        maxval_EC = max(ECs_dict.items(), key=operator.itemgetter(1))[0]
        # zero-out pixels to simulate random EC failures
        X, Y, indices = sdutils.select_random_ECs(packet_template,
                                                  max_EC_malfunctions,
                                                  excluded_ECs=[maxval_EC])
        for idx in range(len(indices)):
            packet[:, Y[idx], X[idx]] = 0
        return packet

    def create_noise_packet(self, max_EC_malfunctions=0):
        packet_template = self._template.packet_template
        packet = np.random.poisson(lam=self.bg_lambda,
                                   size=packet_template.packet_shape)
        X, Y, indices = sdutils.select_random_ECs(packet_template,
                                                  max_EC_malfunctions)
        for idx in range(len(indices)):
            packet[:, Y[idx], X[idx]] = 0
        return packet

    """Generates and returns a set of data containing simulated shower lines
    and corresponding targets, both as numpy arrays, for use in training neural
    networks for classifiction tasks.

    Individual data items are created by first creating a packet for each data
    item and then creating projections which are the actual data items returned
    along with targets for them.

    The data returned is divided into equal-sized quarters as follows:

    1/4: shower data (possibly with malfunctioned EC units)
    2/4: shower data (without malfunctioned EC units)
    3/4: noise data (possibly with malfunctioned EC units)
    4/4: noise data (without malfunctioned EC units)

    Whether there are any data items with malfunctioning ECs depends on the
    property bad_ECs_range.

    Parameters
    ----------
    num_data :          int
                        The number of data items to create in total
    Returns
    -------
    data :      tuple of np.ndarray
                A tuple where each item is a numpy array containing packets or
                projections of a given type (raw, xy, xgtu, ygtu) numbering
                num_data projections for each type.
    targets :   np.ndarray
                Numpy array containing classification targets for each item at
                the same index (for any projection type)
    """
    def create_dataset(self, num_data):
        # create output data holders as needed
        template_shape = self._template.packet_template.packet_shape
        data = self._helper.create_converted_packets_holders(num_data,
                                                             template_shape)
        targets = np.empty((num_data, 2), dtype=np.uint8)

        # output and target generation
        ec_gen = self._bad_ECs_gen
        num_showers = int(num_data / 2)
        shower_creator = self.create_shower_packet
        noise_creator = self.create_noise_packet
        iteration_handlers = (
            {'target': [1, 0], 'start': 0, 'stop': int(num_showers / 2),
             'packet_handler': lambda angle: shower_creator(angle, ec_gen())},
            {'target': [1, 0], 'start': int(num_showers / 2),
             'stop': num_showers,
             'packet_handler': lambda angle: shower_creator(angle)},
            {'target': [0, 1], 'start': num_showers,
             'stop': num_data - int(num_showers / 2),
             'packet_handler': lambda angle: noise_creator(ec_gen())},
            {'target': [0, 1], 'start': num_data - int(num_showers / 2),
             'stop': num_data,
             'packet_handler': lambda angle: noise_creator()}
        )
        # main loop
        for handler in iteration_handlers:
            start, stop = handler['start'], handler['stop']
            packet_handler = handler['packet_handler']
            target = handler['target']
            # idx serves as both an index into targets and data, as well as
            # shower angle in xy projection
            for idx in range(start, stop):
                packet = packet_handler(idx)
                outputs = self._helper.convert_packet(packet)
                for data_idx in range(len(data)):
                    data[data_idx][idx] = outputs[data_idx]
                np.put(targets[idx], [0, 1], target)
        return data, targets


if __name__ == '__main__':
    import cmdint.cmd_interface_datagen as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    helper = args.dataset_helper
    data_generator = simulated_data_generator(args.shower_template,
                                              dataset_helper=helper,
                                              bg_lambda=args.bg_lambda,
                                              bad_ECs_range=args.bad_ECs)
    data, targets = data_generator.create_dataset(args.num_data)

    ds.shuffle_dataset(args.num_shuffles, data, targets)
    ds.save_dataset(data, targets, args.outfiles, args.targetfile)

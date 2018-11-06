import sys
import operator
import random as rand

import numpy as np

import utils.common_utils as cutils
import utils.dataset_utils as ds
import utils.synth_data_utils as sdutils


class simulated_data_generator():

    def __init__(self, shower_template, bg_lambda=(1, 1),
                 bad_ECs_range=(0, 0)):
        self.shower_template = shower_template
        self.bg_lambda_range = bg_lambda
        self.bad_ECs_range = bad_ECs_range

    # generator properties

    @property
    def shower_template(self):
        """
            Template for simulated shower.

            Determines the range of values of start coordinates, number of
            frames the shower line spans and its intensity relative to the
            background.
        """
        return self._template

    @shower_template.setter
    def shower_template(self, value):
        if not isinstance(value, sdutils.simulated_shower_template):
            raise TypeError(('Wrong template type, cannot use {} as shower '
                            ' template').format(type(value)))
        self._template = value

    @property
    def bad_ECs_range(self):
        """
            Tuple of 2 integers (MIN, MAX) representing how many EC units
            within the data should haved simulated malfunctions.

            The effect of EC malfunction is simulated for half of all items in
            the dataset, where the actual number of these units per frame is
            from MIN to MAX inclusive. MIN == MAX impies a constant number of
            malfunctioned EC units per frame. MAX can not be more than the
            total number of ECs per frame. Note also that in case of packets
            with simulated showers, some EC cannot be malfunctioned. Therefore,
            even if this property were set to (num ECs, num ECs), data items
            with smulated showers shall never have every EC unit malfunctioned.
        """
        return self._bad_ECs

    @property
    def bg_lambda_range(self):
        """
            Tuple of 2 integers (MIN, MAX) representing the average background
            value of pixels in the data items.

            The actual average value is different from item to item but always
            from MIN to MAX inclusive. MIN == MAX impies a constant value for
            all items. MIN can not be less than 0.
        """
        return self._bg_lambda

    @bg_lambda_range.setter
    def bg_lambda_range(self, value):
        interval = cutils.check_and_convert_value_to_tuple(value,
                                                           'bg_lambda_range')
        cutils.check_interval_tuple(interval, 'bg_lambda_range', lower_limit=0)
        self._bg_lambda = interval
        lam_min, lam_max = interval[0:2]
        self._bg_lambda_gen = ((lambda: lam_min) if lam_min == lam_max else
                               (lambda: rand.uniform(lam_min, lam_max)))

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
        lam = self._bg_lambda_gen()
        packet = np.random.poisson(lam=lam, size=packet_template.packet_shape)
        GTU, Y, X, vals, meta = sdutils.create_simu_shower_line_from_template(
            self._template, yx_angle, return_metadata=True
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
        num_bad_ECs = len(indices)
        meta['bg_lambda'] = lam
        meta['shower'] = True
        meta['num_bad_ECs'] = num_bad_ECs
        for idx in range(num_bad_ECs):
            packet[:, Y[idx], X[idx]] = 0
        return packet, meta

    def create_noise_packet(self, max_EC_malfunctions=0):
        packet_template = self._template.packet_template
        lam = self._bg_lambda_gen()
        packet = np.random.poisson(lam=lam, size=packet_template.packet_shape)
        X, Y, indices = sdutils.select_random_ECs(packet_template,
                                                  max_EC_malfunctions)
        num_bad_ECs = len(indices)
        meta = dict()
        meta['bg_lambda'] = lam
        meta['shower'] = False
        meta['num_bad_ECs'] = num_bad_ECs
        for idx in range(len(indices)):
            packet[:, Y[idx], X[idx]] = 0
        return packet, meta

    def create_dataset(self, name, num_data, item_types):
        """
            Generate and return a numpy dataset containing simulated showers
            and corresponding targets for them, for use in training neural
            networks for classifiction tasks.

            The data returned is divided into equal-sized quarters as follows:

            1/4: shower data (possibly with malfunctioned EC units)
            2/4: shower data (without malfunctioned EC units)
            3/4: noise data (possibly with malfunctioned EC units)
            4/4: noise data (without malfunctioned EC units)

            Whether there are any data items with malfunctioning ECs depends on
            the property bad_ECs_range.

            Parameters
            ----------
            num_data :          int
                                The number of data items to create in total.
            item_types :        dict of str to bool
                                The requested item types, where the keys are
                                from the utils.dataset_utils.item_types
                                module-level constant.
            Returns
            -------
            dataset :   utils.dataset_utils.numpy_dataset
                        A numpy dataset with capacity and num_items both equal
                        to num_data.
        """
        # create output data holders as needed
        template_shape = self._template.packet_template.packet_shape
        dataset = ds.numpy_dataset(name, template_shape, capacity=num_data,
                                   item_types=item_types)

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
                packet, meta = packet_handler(idx)
                dataset.add_data_item(packet, target, meta)
        return dataset


if __name__ == '__main__':
    import cmdint.cmd_interface_datagen as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    data_generator = simulated_data_generator(args.shower_template,
                                              bg_lambda=args.bg_lambda,
                                              bad_ECs_range=args.bad_ECs)
    dataset = data_generator.create_dataset(args.name, args.num_data,
                                            args.item_types)
    dataset.shuffle_dataset(args.num_shuffles)
    dataset.save(args.outdir)

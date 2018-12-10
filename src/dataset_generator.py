import sys
import operator

import numpy as np

import utils.dataset_utils as ds
import utils.synth_data_utils as sdutils


class simulated_data_generator():

    def __init__(self, shower_template, bg_template):
        if shower_template.packet_template != bg_template.packet_template:
            raise ValueError(("Shower and background templates do not share"
                             " the same packet template."))
        self._shower_template = shower_template
        self._bg_template = bg_template

    # generator properties

    @property
    def shower_template(self):
        """Template for simulated shower parameters."""
        return self._shower_template

    @property
    def bg_template(self):
        """Template for background parameters"""
        return self._bg_template

    # methods

    # TODO: might want to break up these methods and possibly move them
    # to different modules as well
    def create_shower_packet(self, yx_angle, max_EC_malfunctions=0):
        # create the actual packet
        packet_template = self._bg_template.packet_template
        lam = self._bg_template.get_new_bg_lambda()
        packet = np.random.poisson(lam=lam, size=packet_template.packet_shape)
        GTU, Y, X, vals, meta = sdutils.create_simu_shower_line_from_template(
            self._shower_template, yx_angle, return_metadata=True
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
        packet_template = self._bg_template.packet_template
        lam = self._bg_template.get_new_bg_lambda()
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
        template_shape = self._bg_template.packet_template.packet_shape
        dataset = ds.numpy_dataset(name, template_shape, capacity=num_data,
                                   item_types=item_types)

        # output and target generation
        ec_gen = self._bg_template.get_new_bad_ECs
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
    import cmdint.cmd_interface_generator as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    data_generator = simulated_data_generator(
        args.shower_template, args.bg_template
    )
    dataset = data_generator.create_dataset(args.name, args.num_data,
                                            args.item_types)
    dataset.shuffle_dataset(args.num_shuffles)
    dataset.save(args.outdir)

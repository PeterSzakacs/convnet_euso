import sys
import operator

import skimage.filters as filters
import numpy as np

import dataset.constants as cons
import dataset.dataset_utils as ds
import dataset.io.fs_io as io_utils
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

    def _apply_antialias(self, line, vals, sigma=0.7):
        packet_template = self._bg_template.packet_template
        packet_shape = packet_template.packet_shape
        packet = np.zeros(packet_shape, dtype=np.uint8)
        GTU, Y, X = line[:]
        # draw the actual line values into the packet
        packet[GTU, Y, X] = vals
        # blur the line using a gaussian filter
        packet = filters.gaussian(packet, sigma=sigma, mode='constant', cval=0)
        # remove the center of the line
        packet[GTU, Y, X] = 0
        # make all antialiased values "around" the line have at least
        # 1 positive decimal place
        packet *= 1000
        # restore the center of the line
        packet[GTU, Y, X] = vals
        return packet

    # methods

    # TODO: might want to break up these methods and possibly move them
    # to different modules as well
    def create_shower_packet(self, yx_angle, max_EC_malfunctions=0):
        # create the actual packet
        packet_template = self._bg_template.packet_template
        packet_shape = packet_template.packet_shape
        lam = self._bg_template.get_new_bg_lambda()
        GTU, Y, X, vals, meta = sdutils.create_simu_shower_line_from_template(
            self._shower_template, yx_angle, return_metadata=True
        )
        pure_shower_packet = self._apply_antialias((GTU, Y, X), vals)
        final_packet = np.random.poisson(lam=lam, size=packet_shape)
        final_packet = final_packet.astype('uint8')
        final_packet += pure_shower_packet.astype('uint8')

        # get the sum of shower pixel values in all EC modules
        ECs_used = [packet_template.xy_to_ec_idx(x, y) for (x, y) in zip(X, Y)]
        ECs_dict = dict(zip(ECs_used, [0] * len(ECs_used)))
        for idx in range(len(ECs_used)):
            EC = ECs_used[idx]
            ECs_dict[EC] += final_packet[GTU[idx], Y[idx], X[idx]]
        # get the EC containing the maximum sum of pixel values
        maxval_EC = max(ECs_dict.items(), key=operator.itemgetter(1))[0]
        # zero-out pixels to simulate random EC failures
        X, Y, indices = sdutils.select_random_ECs(packet_template,
                                                  max_EC_malfunctions,
                                                  excluded_ECs=[maxval_EC])
        num_bad_ECs = len(indices)
        meta['bg_lambda'] = lam
        meta['num_bad_ECs'] = num_bad_ECs
        for idx in range(num_bad_ECs):
            final_packet[:, Y[idx], X[idx]] = 0
        return final_packet, meta

    def create_noise_packet(self, max_EC_malfunctions=0):
        packet_template = self._bg_template.packet_template
        lam = self._bg_template.get_new_bg_lambda()
        packet = np.random.poisson(lam=lam, size=packet_template.packet_shape)
        X, Y, indices = sdutils.select_random_ECs(packet_template,
                                                  max_EC_malfunctions)
        num_bad_ECs = len(indices)
        meta = {}
        meta['bg_lambda'] = lam
        meta['num_bad_ECs'] = num_bad_ECs
        for idx in range(len(indices)):
            packet[:, Y[idx], X[idx]] = 0
        return packet, meta

    def create_dataset(self, name, num_data, item_types, dtype='uint8'):
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
            dataset :   utils.dataset_utils.NumpyDataset
                        A numpy dataset with capacity and num_items both equal
                        to num_data.
        """
        # create output data holders as needed
        template_shape = self._bg_template.packet_template.packet_shape
        dataset = ds.NumpyDataset(name, template_shape, item_types=item_types,
                                  dtype=dtype)

        # output and target generation
        ec_gen = self._bg_template.get_new_bad_ECs
        num_showers = int(num_data / 2)
        shower_creator = self.create_shower_packet
        noise_creator = self.create_noise_packet
        shower_target =  cons.CLASSIFICATION_TARGETS['shower']
        noise_target = cons.CLASSIFICATION_TARGETS['noise']
        iteration_handlers = (
            {'target': shower_target, 'start': 0, 'stop': int(num_showers / 2),
             'packet_handler': lambda angle: shower_creator(angle, ec_gen())},
            {'target': shower_target, 'start': int(num_showers / 2),
             'stop': num_showers,
             'packet_handler': lambda angle: shower_creator(angle)},
            {'target': noise_target, 'start': num_showers,
             'stop': num_data - int(num_showers / 2),
             'packet_handler': lambda angle: noise_creator(ec_gen())},
            {'target': noise_target, 'start': num_data - int(num_showers / 2),
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
    handler = io_utils.DatasetFsPersistencyHandler(save_dir=args.outdir)
    dataset = data_generator.create_dataset(args.name, args.num_data,
                                            item_types=args.item_types,
                                            dtype=args.dtype)
    handler.save_dataset(dataset, metafields_order=cons.SYNTH_METADATA)

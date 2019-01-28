import sys
import csv

import utils.dataset_utils as ds
import utils.io_utils as iout
import utils.metadata_utils as meta
import cmdint.cmd_interface_condenser as cmd

SRCFILE_KEY = 'source_file_acquisition_full'
REQUIRED_FILELIST_COLUMNS = [SRCFILE_KEY, 'packet_id', 'gtu_in_packet',
                             'event_id']

# for processing simu data:
# - frames 0-20 can be used as examples of bg noise
# - frames 27-47 typically contain the shower
# for processing raw flight data:
# - frames 27-47 are the rule of thumb

class default_event_transformer:

    def __init__(self, target, packet_id, start_gtu, stop_gtu):
        self._target = target
        self._packet_id = packet_id
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def event_to_dataset_items(self, packets, metadata):
        idx = self._packet_id
        start, stop = self._start_gtu, self._stop_gtu
        meta_dict = {SRCFILE_KEY: metadata[SRCFILE_KEY], 'packet_id': idx,
                     'start_gtu': start, 'end_gtu': stop}
        return [(packets[idx][start:stop], self._target, meta_dict), ]


class all_packets_event_transformer():

    def __init__(self, target, start_gtu, stop_gtu):
        self._target = target
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def event_to_dataset_items(self, packets, metadata):
        items   = []
        start, stop = self._start_gtu, self._stop_gtu
        target = self._target
        for idx in range(len(packets)):
            packet = packets[idx]
            meta_dict = {SRCFILE_KEY: metadata[SRCFILE_KEY], 'packet_id': idx,
                         'start_gtu': start, 'end_gtu': stop}
            items.append((packet[start:stop], target, meta_dict))
        return items


class gtu_in_packet_event_transformer:

    def __init__(self, target, num_gtu_before=None, num_gtu_after=None,
                 adjust_if_out_of_bounds=True):
        self._target = target
        self._gtu_before = num_gtu_before or 4
        self._gtu_after = num_gtu_after or 15
        self._gtu_after = self._gtu_after + 1
        self._adjust = adjust_if_out_of_bounds

    @property
    def num_frames(self):
        return self._gtu_after + self._gtu_before

    def event_to_dataset_items(self, packets, metadata):
        idx = int(metadata['packet_id'])
        packet = packets[idx]
        packet_gtu = int(metadata['gtu_in_packet'])
        start = packet_gtu - self._gtu_before
        stop = packet_gtu + self._gtu_after
        if (start < 0 or stop > packet.shape[0]) and not self._adjust:
            idx = metadata['event_id']
            raise Exception('Frame range for event id {} ({}:{}) is out of '
                            'packet bounds'.format(idx, start, stop))
        else:
            while start < 0:
                start += 1
                stop += 1
            while stop > packet.shape[0]:
                start -= 1
                stop -= 1
        meta_dict = {SRCFILE_KEY: metadata[SRCFILE_KEY], 'packet_id': idx,
                     'start_gtu': start, 'end_gtu': stop}
        return [(packet[start:stop], self._target, meta_dict), ]


# script to coallesce (simulated or real) data from several files
# into a single larger .npy file to use as a dataset for training
# or evaluating a network.

# currently no usage for L1 trigger file, so it is just ignored

if __name__ == "__main__":
    # command line parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    packet_template = args.template
    extractor = iout.packet_extractor(packet_template=packet_template)

    target = [0, 1] if args.target == 'noise' else [1, 0]
    if args.converter == 'gtupack':
        before, after = args.num_gtu_around[0:2]
        data_transformer = gtu_in_packet_event_transformer(
            target, num_gtu_before=before, num_gtu_after=after,
            adjust_if_out_of_bounds=(not args.no_bounds_adjust))
    elif args.converter == 'allpack':
        start, stop = args.gtu_range[0:2]
        data_transformer = all_packets_event_transformer(target, start, stop)
    else:
        start, stop = args.gtu_range[0:2]
        data_transformer = default_event_transformer(target, args.packet_idx,
                                                     start, stop)

    extracted_packet_shape = list(packet_template.packet_shape)
    extracted_packet_shape[0] = data_transformer.num_frames

    output_handler = iout.dataset_fs_persistency_handler(save_dir=args.outdir)
    dataset = ds.numpy_dataset(args.name, extracted_packet_shape,
                               item_types=args.item_types, dtype=args.dtype)
    input_tsv = args.filelist
    total_num_data = 0
    cache = {}


    # main loop
    rows = iout.load_TSV(input_tsv, selected_columns=REQUIRED_FILELIST_COLUMNS)
    for row in rows:
        srcfile, triggerfile = row[SRCFILE_KEY], None
        print("Processing file {}".format(srcfile))
        packets = cache.get(srcfile, None)
        if packets is None:
            if srcfile.endswith('.npy'):
                packets = extractor.extract_packets_from_npyfile(
                    srcfile, triggerfile=triggerfile)
                cache[srcfile] = packets
            elif srcfile.endswith('.root'):
                packets = extractor.extract_packets_from_rootfile(
                    srcfile, triggerfile=triggerfile)
                cache[srcfile] = packets
            else:
                raise Exception('Unknown file type: {}'.format(srcfile))
        items = data_transformer.event_to_dataset_items(packets, row)
        num_items = len(items)
        print("Extracted: {} data items".format(num_items))
        for item in items:
            dataset.add_data_item(item[0], item[1], metadata=item[2])
        total_num_data += num_items
        print("Dataset current total data items count: {}".format(
            total_num_data
        ))
        if len(cache.items()) == args.max_cache_size:
            cache.popitem()

    print('Creating dataset "{}" containing {} items'.format(
        args.name, total_num_data))
    output_handler.save_dataset(dataset)

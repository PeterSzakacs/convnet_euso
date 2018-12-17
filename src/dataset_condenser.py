import sys
import csv

import utils.dataset_utils as ds
import utils.io_utils as iout
import cmdint.cmd_interface_condenser as cmd


def simu_transformer(packets, metadata):
    items   = []
    srcfile_key = 'source_file_acquisition_full'
    packet  = packets[1]
    gtus    = [27, 47]
    metadata_dict = {srcfile_key: metadata[srcfile_key], 'packet_idx': 1, 
                     'start_gtu': gtus[0], 'end_gtu': gtus[1]}
    items.append((packet[gtus[0]:gtus[1]], [1, 0], metadata_dict))
    gtus = [0, 20]
    metadata_dict = {srcfile_key: metadata[srcfile_key], 'packet_idx': 1, 
                     'start_gtu': gtus[0], 'end_gtu': gtus[1]}
    items.append((packet[gtus[0]:gtus[1]], [0, 1], metadata_dict))
    return items


def flight_transformer(packets, metadata):
    items   = []
    srcfile_key = 'source_file_acquisition_full'
    gtus    = [27, 47]
    for idx in len(packets):
        packet = packets[idx]
        metadata_dict = {srcfile_key: metadata[srcfile_key], 'packet_idx': idx, 
                         'start_gtu': gtus[0], 'end_gtu': gtus[1]}
        items.append((packet[gtus[0]:gtus[1]], [0, 1], metadata_dict))
    return items


class custom_transformer:

    def __init__(self, target, gtus=(None, None)):
        self.target = target
        self.gtus_range = gtus

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    @property
    def gtus_range(self):
        return self._gtus

    @gtus_range.setter
    def gtus_range(self, value):
        self._gtus = value

    def process_event(self, packets, metadata):
        items   = []
        srcfile_key = 'source_file_acquisition_full'
        idx = int(metadata['packet_id'])
        packet = packets[idx]
        packet_gtu = int(metadata['gtu_in_packet'])
        start_gtu = self._gtus[0] or packet_gtu - 4
        end_gtu = self._gtus[1] or packet_gtu + 16
        while start_gtu < 0:
            start_gtu += 1
            end_gtu += 1
        while end_gtu > packet.shape[0]:
            start_gtu -= 1
            end_gtu -= 1

        subpacket = packet[start_gtu: end_gtu]
        metadata_dict = {srcfile_key: metadata[srcfile_key], 'packet_idx': idx, 
                         'start_gtu': start_gtu, 'end_gtu': end_gtu}
        items.append((subpacket, self._target, metadata_dict))
        return items


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

    if args.simu:
        num_frames = 20
        data_transformer = simu_transformer
    elif args.flight:
        num_frames = 20
        data_transformer = flight_transformer
    else:
        GTUs_range = (args.start_gtu, args.end_gtu)
        transformer = custom_transformer(args.target, gtus=GTUs_range)
        data_transformer = transformer.process_event
        if args.start_gtu == None and args.end_gtu == None:
            num_frames = 20
        else:
            num_frames = args.end_gtu - args.start_gtu

    extracted_packet_shape = list(packet_template.packet_shape)
    extracted_packet_shape[0] = num_frames

    dataset = ds.numpy_dataset(args.name, extracted_packet_shape, item_types=args.item_types)
    input_tsv = args.filelist
    total_num_data = 0
    cache = {}


    # main loop
    with open(input_tsv, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            srcfile, triggerfile = row['source_file_acquisition_full'], None
            print("Processing file {}".format(srcfile))
            packets = cache.get(srcfile, None)
            if packets is None:
                if srcfile.endswith('.npy'):
                    packets = extractor.extract_packets_from_npyfile(
                        srcfile, triggerfile=triggerfile
                    )
                    cache[srcfile] = packets
                elif srcfile.endswith('.root'):
                    packets = extractor.extract_packets_from_rootfile(
                        srcfile, triggerfile=triggerfile
                    )
                    cache[srcfile] = packets
                else:
                    raise Exception('Unknown file type: {}'.format(srcfile))
            items = data_transformer(packets, row)
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

    print('Creating dataset "{}" containing {} items'.format(args.name, total_num_data))
    dataset.save(args.outdir)

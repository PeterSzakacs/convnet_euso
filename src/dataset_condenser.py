import collections

import dataset.constants as cons
import dataset.dataset_utils as ds
import dataset.io.fs_io as fs_io
import utils.io_utils as io_utils

SRCFILE_KEY = 'source_file_acquisition_full'

# for processing simu data:
# - frames 0-20 can be used as examples of bg noise
# - frames 27-47 typically contain the shower
# for processing raw flight data:
# - frames 27-47 are the rule of thumb

class default_event_transformer:

    REQUIRED_FILELIST_COLUMNS = ('packet_id', )

    def __init__(self, packet_id, start_gtu, stop_gtu):
        self._packet_id = packet_id
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def event_to_packets(self, event_packets, event_metadata):
        idx = self._packet_id
        start, stop = self._start_gtu, self._stop_gtu
        result = {'packet': packets[idx][start:stop], 'packet_id': idx,
                  'start_gtu': start, 'end_gtu': stop, }
        return [result, ]


class all_packets_event_transformer():

    REQUIRED_FILELIST_COLUMNS = ()

    def __init__(self, start_gtu, stop_gtu):
        self._start_gtu = start_gtu
        self._stop_gtu = stop_gtu

    @property
    def num_frames(self):
        return self._stop_gtu - self._start_gtu

    def event_to_packets(self, event_packets, event_metadata):
        results = []
        start, stop = self._start_gtu, self._stop_gtu
        for idx in range(len(packets)):
            packet = event_packets[idx][start:stop]
            result = {'packet': packet, 'packet_id': idx,
                      'start_gtu': start, 'end_gtu': stop, }
            results.append(result)
        return results


class gtu_in_packet_event_transformer:

    REQUIRED_FILELIST_COLUMNS = ('packet_id', 'gtu_in_packet')

    def __init__(self, num_gtu_before=None, num_gtu_after=None,
                 adjust_if_out_of_bounds=True):
        self._gtu_before = num_gtu_before or 4
        self._gtu_after = num_gtu_after or 15
        self._gtu_after = self._gtu_after + 1
        self._adjust = adjust_if_out_of_bounds

    @property
    def num_frames(self):
        return self._gtu_after + self._gtu_before

    def event_to_packets(self, event_packets, event_metadata):
        idx = int(event_metadata['packet_id'])
        packet = event_packets[idx]
        packet_gtu = int(event_metadata['gtu_in_packet'])
        start = packet_gtu - self._gtu_before
        stop = packet_gtu + self._gtu_after
        if (start < 0 or stop > packet.shape[0]) and not self._adjust:
            idx = event_metadata.get(['event_id'], event_metadata[SRCFILE_KEY])
            raise Exception('Frame range for event id {} ({}:{}) is out of '
                            'packet bounds'.format(idx, start, stop))
        else:
            while start < 0:
                start += 1
                stop += 1
            while stop > packet.shape[0]:
                start -= 1
                stop -= 1
        result = {'packet': packet[start:stop], 'packet_id': idx,
                  'start_gtu': start, 'end_gtu': stop}
        return [result, ]


class MetadataCreator:

    MANDATORY_PACKET_ATTRS = ('packet_id', 'start_gtu', 'end_gtu')
    MANDATORY_EVENT_META = (SRCFILE_KEY, )

    def __init__(self, extra_fields=None):
        self._extra = set(extra_fields or [])

    @property
    def extra_metafields(self):
        return self._extra

    def create_metadata(self, packet_attrs, event_metadata):
        metadata = []
        metafields = self._extra.union(self.MANDATORY_EVENT_META)
        meta_dict = {field: event_metadata.get(field) for field in metafields}
        for packet_attr in packet_attrs:
            meta = meta_dict.copy()
            for fieldname in self.MANDATORY_PACKET_ATTRS:
                meta[fieldname] = packet_attr[fieldname]
            metadata.append(meta)
        return metadata


class PacketCache:

    def __init__(self, max_size, packet_extractors, num_evict_on_full=10):
        if max_size < num_evict_on_full:
            raise ValueError('Number of evicted items must be less than the '
                             'cache size')
        self._maxsize = max_size
        self._extractors = {}
        for key in ('NPY', 'ROOT'):
            self._extractors[key] = packet_extractors[key]
        self._num_evict = num_evict_on_full
        self._packets = {}
        self._file_queue = collections.deque([], max_size)

    def get(self, filename):
        all_packets, queue = self._packets, self._file_queue
        packets = all_packets.get(filename, None)
        if packets is None:
            extractors = self._extractors
            if filename.endswith('.npy'):
                extractor = extractors['NPY']
            elif filename.endswith('.root'):
                extractor = extractors['ROOT']
            else:
                raise Exception('Unknown file type: {}'.format(filename))
            packets = extractor(filename)
            all_packets[filename] = packets
            queue.append(filename)
            if len(queue) == self._maxsize:
                for idx in range(self._num_evict):
                    filename = queue.popleft()
                    all_packets.pop(filename)
        return packets


# script to coallesce (simulated or real) data from several files
# into a single larger .npy file to use as a dataset for training
# or evaluating a network.

# currently no usage for L1 trigger file, so it is just ignored

if __name__ == "__main__":
    import sys
    import cmdint.cmd_interface_condenser as cmd

    # command line parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    packet_template = args.template
    extractor = io_utils.packet_extractor(packet_template=packet_template)
    extractors = {'NPY': extractor.extract_packets_from_npyfile,
                  'ROOT': extractor.extract_packets_from_rootfile}
    cache = PacketCache(args.max_cache_size, extractors,
                        num_evict_on_full=args.num_evicted)

    if args.converter == 'gtupack':
        before, after = args.num_gtu_around[0:2]
        data_transformer = gtu_in_packet_event_transformer(
            num_gtu_before=before, num_gtu_after=after,
            adjust_if_out_of_bounds=(not args.no_bounds_adjust))
    elif args.converter == 'allpack':
        start, stop = args.gtu_range[0:2]
        data_transformer = all_packets_event_transformer(start, stop)
    else:
        packet_id, (start, stop) = args.packet_idx, args.gtu_range
        data_transformer = default_event_transformer(packet_id, start, stop)
    target = cons.CLASSIFICATION_TARGETS[args.target]
    meta_creator = MetadataCreator(args.extra_metafields)

    extracted_packet_shape = list(packet_template.packet_shape)
    extracted_packet_shape[0] = data_transformer.num_frames
    output_handler = fs_io.dataset_fs_persistency_handler(save_dir=args.outdir)
    dataset = ds.numpy_dataset(args.name, extracted_packet_shape,
                               item_types=args.item_types, dtype=args.dtype)


    # main loop
    input_tsv = args.filelist
    fields = set(data_transformer.REQUIRED_FILELIST_COLUMNS)
    fields = fields.union(meta_creator.MANDATORY_EVENT_META)
    fields = fields.union(meta_creator.extra_metafields)
    rows = io_utils.load_TSV(input_tsv, selected_columns=fields)
    for row in rows:
        srcfile = row[SRCFILE_KEY]
        print("Processing file {}".format(srcfile))
        packets = cache.get(srcfile)
        dataset_packets = data_transformer.event_to_packets(packets, row)
        print("Extracted: {} data items".format(len(dataset_packets)))
        metadata = meta_creator.create_metadata(dataset_packets, row)
        for idx in range(len(dataset_packets)):
            packet, meta = dataset_packets[idx]['packet'], metadata[idx]
            dataset.add_data_item(packet, target, metadata=meta)
        print("Dataset current total data items count: {}".format(
            dataset.num_data
        ))

    print('Creating dataset "{}" containing {} items'.format(
        dataset.name, dataset.num_data))
    output_handler.save_dataset(dataset)

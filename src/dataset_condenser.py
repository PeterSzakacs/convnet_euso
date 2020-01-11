import logging

import dataset.dataset_utils as ds
import dataset.io.fs_io as fs_io
import dataset.tck.constants as tck_cons
import dataset.tck.event_transformers as event_tran
import dataset.tck.io_utils as tck_io_utils
import dataset.tck.metadata_handlers as meta
import dataset.tck.target_handlers as targ
import utils.io_utils as io_utils


# script to coallesce (simulated or real) data from several files
# into a single larger .npy file to use as a dataset for training
# or evaluating a network.

# currently no usage for L1 trigger file, so it is just ignored

# for processing simu data:
# - frames 0-20 can be used as examples of bg noise
# - frames 27-47 typically contain the shower
# for processing raw flight data:
# - frames 27-47 are the rule of thumb


class DatasetCondenser:

    def __init__(self, packets_handler, metadata_handler, targets_handler,
                 logging_level=None):
        self.packets_handler = packets_handler
        self.metadata_handler = metadata_handler
        self.targets_handler = targets_handler
        self.logger = logging.getLogger(self.__class__.__name__)
        self.set_logging_level(logging_level=logging_level)

    def set_logging_level(self, logging_level=None):
        logger = self.logger
        level = logging_level or logging.INFO
        logger.level = level
        stdout_handler = logging.StreamHandler()
        stdout_handler.level = level
        logger.addHandler(stdout_handler)

    def add_to_dataset(self, event_stream, dataset):
        events = self.packets_handler.process_events(event_stream)
        events = self.metadata_handler.process_events(events)
        events = self.targets_handler.process_events(events)
        logger = self.logger
        for event_list in events:
            event_meta = event_list[0][2]
            logger.info(f"Processing {len(event_list)} packets "
                        f"from {event_meta[tck_cons.SRCFILE_KEY]}")
            for event in event_list:
                packet, target, meta = event[:]
                dataset.add_data_item(packet, target, metadata=meta)
            logger.info(f"Dataset current total data items count: "
                        f"{dataset.num_data}")


def get_data_handler(args):
    packet_template = args.template
    extractor = tck_io_utils.PacketExtractor(packet_template=packet_template)
    extractors = {'NPY': extractor.extract_packets_from_npyfile,
                  'ROOT': extractor.extract_packets_from_rootfile}
    cache = tck_io_utils.PacketCache(args.max_cache_size, extractors,
                                     num_evict_on_full=args.num_evicted)
    if args.converter == 'gtupack':
        before, after = args.num_gtu_around[0:2]
        handler = event_tran.GtuInPacketEventTransformer(
            cache.get, num_gtu_before=before, num_gtu_after=after,
            adjust_if_out_of_bounds=(not args.no_bounds_adjust))
    elif args.converter == 'allpack':
        start, stop = args.gtu_range[0:2]
        handler = event_tran.AllPacketsEventTransformer(cache.get, start, stop)
    else:
        packet_id, (start, stop) = args.packet_idx, args.gtu_range
        handler = event_tran.DefaultEventTransformer(cache.get, packet_id,
                                                     start, stop)
    return handler


def main(args):
    data_handler = get_data_handler(args)
    target_handler = targ.get_target_handler(args.target_handler_type,
                                             **args.target_handler_args)
    meta_creator = meta.MetadataCreator(args.extra_metafields)

    condenser = DatasetCondenser(data_handler, meta_creator, target_handler)

    extracted_packet_shape = list(args.template.packet_shape)
    extracted_packet_shape[0] = data_handler.num_frames
    dataset = ds.NumpyDataset(args.name, extracted_packet_shape,
                              item_types=args.item_types, dtype=args.dtype)

    input_tsv = args.filelist
    fields = set(data_handler.REQUIRED_FILELIST_COLUMNS)
    fields = fields.union(meta_creator.MANDATORY_EVENT_META)
    fields = fields.union(meta_creator.extra_metafields)
    rows = io_utils.load_TSV(input_tsv, selected_columns=fields)
    condenser.add_to_dataset(rows, dataset)
    output_handler = fs_io.DatasetFsPersistencyHandler(save_dir=args.outdir)
    print(f"Creating dataset \"{dataset.name}\" containing "
          f"{dataset.num_data} items")
    output_handler.save_dataset(dataset)


if __name__ == "__main__":
    import sys
    import cmdint.cmd_interface_condenser as cmd

    # command line parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    main(args)

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


def main(**kwargs):
    # get conversion class from events to dataset items
    packet_template = kwargs['packet_template']
    cache = get_packet_cache(packet_template, **kwargs['cache'])
    condenser = get_condenser(cache.get, **kwargs)

    # create output dataset
    data_handler = condenser.packets_handler
    output_packet_shape = list(packet_template.packet_shape)
    output_packet_shape[0] = data_handler.num_frames
    dataset, handler = get_output_dataset_and_handler(output_packet_shape,
                                                      **kwargs['output_dataset'])

    # load events from filelist and add them to output dataset as items
    meta_creator = condenser.metadata_handler
    input_tsv = kwargs['filelist']
    fields = set(data_handler.REQUIRED_FILELIST_COLUMNS)
    fields = fields.union(meta_creator.MANDATORY_EVENT_META)
    fields = fields.union(meta_creator.extra_metafields)
    rows = io_utils.load_TSV(input_tsv, selected_columns=fields)
    condenser.add_to_dataset(rows, dataset)

    # save dataset
    print(f"Creating dataset \"{dataset.name}\" containing "
          f"{dataset.num_data} items")
    handler.save_dataset(dataset)


def get_packet_cache(packet_template, **cache_args):
    extractor = tck_io_utils.PacketExtractor(packet_template=packet_template)
    extractors = {'NPY': extractor.extract_packets_from_npyfile,
                  'ROOT': extractor.extract_packets_from_rootfile}
    cache = tck_io_utils.PacketCache(
        cache_args['max_size'], extractors,
        num_evict_on_full=cache_args['num_evict_on_full'])
    return cache


def get_condenser(packet_extraction_fn, **kwargs):
    event_transformer = kwargs['event_transformer']
    data_handler = event_tran.get_event_transformer(
        event_transformer['name'], packet_extraction_fn,
        **event_transformer['args'])

    target_handler = kwargs['target_handler']
    target_handler = targ.get_target_handler(
        target_handler['name'], **target_handler['args'])

    meta_creator = meta.MetadataCreator(kwargs['extra_metafields'])
    return DatasetCondenser(data_handler, meta_creator, target_handler)


def get_output_dataset_and_handler(output_packet_shape, **dataset_args):
    dataset = ds.NumpyDataset(dataset_args['name'], output_packet_shape,
                              item_types=dataset_args['item_types'],
                              dtype=dataset_args['dtype'])
    output_handler = fs_io.DatasetFsPersistencyHandler(
        save_dir=dataset_args['outdir'])
    return dataset, output_handler


if __name__ == "__main__":
    import sys
    import cmdint.cmd_interface_condenser as cmd

    # command line parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    main(**args)

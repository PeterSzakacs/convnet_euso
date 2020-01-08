import dataset.constants as cons
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

if __name__ == "__main__":
    import sys
    import cmdint.cmd_interface_condenser as cmd

    # command line parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    packet_template = args.template
    extractor = tck_io_utils.PacketExtractor(packet_template=packet_template)
    extractors = {'NPY': extractor.extract_packets_from_npyfile,
                  'ROOT': extractor.extract_packets_from_rootfile}
    cache = tck_io_utils.PacketCache(args.max_cache_size, extractors,
                                     num_evict_on_full=args.num_evicted)

    if args.converter == 'gtupack':
        before, after = args.num_gtu_around[0:2]
        data_transformer = event_tran.GtuInPacketEventTransformer(cache.get,
            num_gtu_before=before, num_gtu_after=after,
            adjust_if_out_of_bounds=(not args.no_bounds_adjust))
    elif args.converter == 'allpack':
        start, stop = args.gtu_range[0:2]
        data_transformer = event_tran.AllPacketsEventTransformer(cache.get,
                                                                 start, stop)
    else:
        packet_id, (start, stop) = args.packet_idx, args.gtu_range
        data_transformer = event_tran.DefaultEventTransformer(cache.get,
                                                              packet_id,
                                                              start, stop)
    target_handler = targ.StaticTargetHandler(
        cons.CLASSIFICATION_TARGETS[args.target])
    meta_creator = meta.MetadataCreator(args.extra_metafields)

    extracted_packet_shape = list(packet_template.packet_shape)
    extracted_packet_shape[0] = data_transformer.num_frames
    output_handler = fs_io.DatasetFsPersistencyHandler(save_dir=args.outdir)
    dataset = ds.NumpyDataset(args.name, extracted_packet_shape,
                              item_types=args.item_types, dtype=args.dtype)


    # main loop
    input_tsv = args.filelist
    fields = set(data_transformer.REQUIRED_FILELIST_COLUMNS)
    fields = fields.union(meta_creator.MANDATORY_EVENT_META)
    fields = fields.union(meta_creator.extra_metafields)
    rows = io_utils.load_TSV(input_tsv, selected_columns=fields)
    events = data_transformer.process_events(rows)
    events = meta_creator.process_events(events)
    events = target_handler.process_events(events)
    for event_list in events:
        event_meta = event_list[0][2]
        print("Processing {} packets from {}".format(
            len(event_list), event_meta[tck_cons.SRCFILE_KEY]))
        for event in event_list:
            packet, target, meta = event[:]
            dataset.add_data_item(packet, target, metadata=meta)
        print("Dataset current total data items count: {}".format(
            dataset.num_data
        ))

    print('Creating dataset "{}" containing {} items'.format(
        dataset.name, dataset.num_data))
    output_handler.save_dataset(dataset)

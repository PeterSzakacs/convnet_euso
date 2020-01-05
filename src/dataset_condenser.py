import dataset.constants as cons
import dataset.tck.constants as tck_cons
import dataset.dataset_utils as ds
import dataset.io.fs_io as fs_io
import dataset.tck.event_transformers as event_tran
import dataset.tck.io_utils as tck_io_utils
import dataset.tck.metadata_handlers as meta
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
        data_transformer = event_tran.GtuInPacketEventTransformer(
            num_gtu_before=before, num_gtu_after=after,
            adjust_if_out_of_bounds=(not args.no_bounds_adjust))
    elif args.converter == 'allpack':
        start, stop = args.gtu_range[0:2]
        data_transformer = event_tran.AllPacketsEventTransformer(start, stop)
    else:
        packet_id, (start, stop) = args.packet_idx, args.gtu_range
        data_transformer = event_tran.DefaultEventTransformer(packet_id,
                                                              start, stop)
    target = cons.CLASSIFICATION_TARGETS[args.target]
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
    for row in rows:
        srcfile = row[tck_cons.SRCFILE_KEY]
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

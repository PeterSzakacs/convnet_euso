# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime as dt

import visualization.classification_report_writer as cwriter
import utils.dataset_utils as ds
import utils.network_utils as netutils


if __name__ == '__main__':

    import sys
    import cmdint.cmd_interface_checker as cmd
    import visualization.event_visualization as eviz

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    logdir = args.logdir or netutils.DEFAULT_CHECKING_LOGDIR
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # do not use the GPU
    if args.usecpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load input data, targets and metadata
    dataset = ds.numpy_dataset.load_dataset(args.srcdir, args.name,
                                            item_types=args.item_types)

    # deprecated feature, as this was never really used.
    # else:
        # args.infile = args.acqfile
        # args.targetfile = 'None (implicitly assumed_noise in all packets)'
        # extractor = io_utils.packet_extractor()
        # manipulator = dat.packet_manipulator(extractor.packet_template)
        # X_all = []
        # proj_creator = lambda packet, packet_idx, srcfile: X_all.append(
        #                             manipulator.create_x_y_projection(packet, start_idx=27, end_idx=47))
        # extractor.extract_packets_from_rootfile_and_process(args.acqfile, triggerfile=args.triggerfile, on_packet_extracted=proj_creator)
        # # implicitly assuming that all packets will contain noise
        # X_all = np.array(X_all, dtype=np.uint8)
        # Y_all = np.array([[0, 1] for idx in range(len(X_all))], dtype=np.uint8)

    if args.flight:
        meta_transformer = eviz.flight_metadata_to_text
    elif args.simu:
        meta_transformer = eviz.simu_metadata_to_text
    elif args.synth:
        meta_transformer = eviz.synth_metadata_to_text

    # main loop
    writer = cwriter.report_writer(dataset.item_types, "/tmp",
                                   max_table_size=args.tablesize,
                                   metadata_text_transformer=meta_transformer)
    for network in args.networks:
        network_module_name, model_file = network[0], network[1]
        network_module_name = "net." + network_module_name
        run_id = netutils.get_default_run_id(network_module_name);
        tb_dir = os.path.join(logdir, run_id)
        os.mkdir(tb_dir)
        model, net, conv, fc = netutils.import_convnet(
            network_module_name, tb_dir, dataset.item_shapes,
            model_file=model_file
        )
        log_data, hits, classes_count = netutils.evaluate_classification_model(
            model, dataset, slice(args.eval_numframes), onlyerr=args.onlyerr
        )
        # sort ascending in the direction of greater noise probability
        log_data.sort(key=lambda item: item[1][1])

        # Misc statistics
        shower_count, noise_count = classes_count[0], classes_count[1]
        num_items = shower_count + noise_count
        acc = (hits * 100 / dataset.num_data)
        err = 100 - acc

        print('acc: {}, err: {}, showers: {}, noise: {}'.format(acc, err,
                                                                shower_count,
                                                                noise_count))

        logs_dict = {'logs': log_data, 'net_arch': network_module_name,
                     'model_file': os.path.abspath(model_file),
                     'time': dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                     'showers': shower_count, 'noise': noise_count,
                     'hits': hits, 'misses': num_items - hits}
        savedir = os.path.join(tb_dir, 'eval_report')
        os.mkdir(savedir)
        writer.savedir = savedir
        writer.write_reports(logs_dict, dataset)

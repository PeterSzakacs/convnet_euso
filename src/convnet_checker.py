# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import csv

import utils.dataset_utils as ds
import utils.io_utils as io_utils
import utils.metadata_utils as meta
import utils.network_utils as netutils


if __name__ == '__main__':

    import sys
    import cmdint.cmd_interface_checker as cmd

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])

    name, srcdir = args.name, args.srcdir
    item_types = args.item_types

    logdir = args.logdir or netutils.DEFAULT_CHECKING_LOGDIR
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # do not use the GPU
    if args.usecpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load input data, targets and metadata
    input_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types)

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

    # main code
    network_module_name, model_file = args.network[0:2]
    network_module_name = "net." + network_module_name
    run_id = netutils.get_default_run_id(network_module_name)
    tb_dir = os.path.join(logdir, run_id)
    os.mkdir(tb_dir)
    model, net, conv, fc = netutils.import_convnet(
        network_module_name, tb_dir, dataset.item_shapes,
        model_file=model_file
    )
    log_data = netutils.evaluate_classification_model(
        model, dataset, items_slice=slice(args.eval_numframes)
    )

    # output results
    meta_fields = dataset.metadata_fields
    meta_order = args.meta_order
    extra_fields = meta_fields.difference(meta_order)
    extra_fields = list(extra_fields)
    extra_fields.sort()
    headers = netutils.CLASSIFICATION_FIELDS + meta_order + extra_fields
    writer = csv.DictWriter(args.outfile, headers, delimiter='\t')
    writer.writeheader()
    writer.writerows(log_data)
    if args.outfile is not sys.stdout:
        print('Saved report to {}'.format(args.outfile.name))
    args.outfile.close()

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

    # do not use the GPU
    if args.usecpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load input dataset
    input_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    #load trained network model
    logdir = netutils.DEFAULT_CHECKING_LOGDIR
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    network_module_name, model_file = args.network[0:2]
    network_module_name = "net." + network_module_name
    run_id = netutils.get_default_run_id(network_module_name)
    tb_dir = os.path.join(logdir, run_id)
    os.mkdir(tb_dir)
    model, net, conv, fc = netutils.import_convnet(
        network_module_name, tb_dir, dataset.item_shapes,
        model_file=model_file
    )

    # check (evaluate) model
    log_data = netutils.evaluate_classification_model(
        model, dataset, items_slice=slice(args.start_item, args.stop_item)
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

import csv
import os

import dataset.io.fs_io as io_utils
import net.network_utils as netutils
import utils.config_utils as cutils


if __name__ == '__main__':

    import sys
    import cmdint.cmd_interface_checker as cmd

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])

    name, srcdir = args.name, args.srcdir
    item_types = args.item_types

    # do not use the GPU
    if args.usecpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load input dataset
    input_handler = io_utils.DatasetFsPersistencyHandler(load_dir=srcdir)
    dataset = input_handler.load_dataset(name, item_types=item_types)

    #load trained network model
    logdir = cutils.get_config_for_module("model_checker")['default']['logdir']
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    network_module_name, model_file = args.network, args.model_file
    network_module_name = "net." + network_module_name
    run_id = netutils.get_default_run_id(network_module_name)
    tb_dir = os.path.join(logdir, run_id)
    os.mkdir(tb_dir)
    model = netutils.import_model(network_module_name, dataset.item_shapes,
                                  model_file=model_file, tb_dir=tb_dir)

    # check (evaluate) model
    log_data = netutils.evaluate_classification_model(
        model,
        dataset,
        items_slice=slice(args.start_item, args.stop_item)
    )

    # output results
    meta_fields = dataset.metadata_fields
    extra_fields = list(meta_fields)
    extra_fields.sort()
    headers = netutils.CLASSIFICATION_FIELDS + extra_fields
    writer = csv.DictWriter(args.outfile, headers, delimiter='\t')
    writer.writeheader()
    writer.writerows(log_data)
    if args.outfile is not sys.stdout:
        print('Saved report to {}'.format(args.outfile.name))
    args.outfile.close()

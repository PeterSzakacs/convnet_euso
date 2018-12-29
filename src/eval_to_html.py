import os
import csv

import utils.metadata_utils as meta
import utils.network_utils as netutils
import visualization.classification_report_writer as cwriter
import visualization.event_visualization as eviz


DEFAULT_REPORT_LOGDIR = '/run/user/{}/eval_html'.format(os.getuid())


if __name__ == "__main__":
    import sys
    import cmdint.cmd_interface_eval2html as cmd

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    network_module_name, model_file = args.network[0:2]
    network_module_name = "net." + network_module_name
    model_file = os.path.abspath(model_file)

    run_id = netutils.get_default_run_id(network_module_name)
    tb_dir = os.path.join(DEFAULT_REPORT_LOGDIR, run_id)
    logdir = args.logdir or tb_dir
    os.makedirs(logdir, exist_ok=True)

    if args.flight:
        meta_fields = meta.FLIGHT_METADATA
    elif args.simu:
        meta_fields = meta.SIMU_METADATA
    elif args.synth:
        meta_fields = meta.SYNTH_METADATA
    headers = meta_fields + meta.CLASS_METADATA

    in_reader = csv.DictReader(args.infile, fieldnames=headers, delimiter='\t')
    log_data = []

    # skip header
    next(in_reader)
    for row in in_reader:
        log_data.append(row)
    args.infile.close()
    context = {'net_arch': network_module_name, 'model_file': model_file,
               'dataset': args.name, 'item_types': args.item_types}

    writer = cwriter.report_writer(logdir, table_size=args.tablesize,
                                   extra_fields=meta_fields)
    writer.write_reports(log_data, context)

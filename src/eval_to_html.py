import os
import csv

import net.network_utils as netutils
import visualization.classification_report_writer as cwriter
import visualization.event_visualization as eviz


DEFAULT_REPORT_LOGDIR = '/run/user/{}/eval_html'.format(os.getuid())


if __name__ == "__main__":
    import sys
    import cmdint.cmd_interface_eval2html as cmd

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    network_module_name = args.network
    network_module_name = "net." + network_module_name
    model_file = args.model_file
    model_file = os.path.abspath(model_file)

    run_id = netutils.get_default_run_id(network_module_name)
    tb_dir = os.path.join(DEFAULT_REPORT_LOGDIR, run_id)
    logdir = args.logdir or tb_dir
    os.makedirs(logdir, exist_ok=True)

    in_reader = csv.DictReader(args.infile, delimiter='\t')
    all_fields = in_reader.fieldnames
    mandatory_fields = netutils.CLASSIFICATION_FIELDS
    extra_fields = [field for field in all_fields
                    if field not in mandatory_fields]
    log_data = []

    for row in in_reader:
        log_data.append(row)
    args.infile.close()
    item_types = args.item_types
    context = {'network': network_module_name, 'model_file': model_file,
               'dataset': args.name}

    writer = cwriter.ReportWriter(logdir, table_size=args.tablesize,
                                  extra_fields_order=extra_fields)
    writer.write_reports(log_data, item_types, **context)

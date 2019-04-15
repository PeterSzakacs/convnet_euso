import pandas as pd
import sklearn.metrics as metrics

import libs.data_analysis as dutils
dutils.use('svg')
import utils.analysis_utils as autils


def _cm_iter(group_df, all_targets, cm_func=metrics.confusion_matrix):
    for val, rows in group_df:
        yield val, cm_func(rows['target'], rows['output'], labels=all_targets)


def main(**args):
    infiles, outfile = args['infiles'], args['outfile']
    column, column_type = args['column'], args['column_type']
    target, all_targets = args['class_target'], args['all_targets']

    # set visualization settings
    err_def = dutils.EFFICIENCY_STAT_ERRORBAR_DEFAULTS
    fill_def = dutils.EFFICIENCY_STAT_FILL_BETWEEN_DEFAULTS

    num_lines = len(infiles)
    plot_labels = args['plot_labels']
    if plot_labels is None:
        plot_labels = [''] * num_lines
        add_legend = False
    else:
        add_legend = True
        legend_props = {'loc': 'center',
                        'prop': {'size': (args['legend_fontsize'] or 6)}}
    fill_colors = args['plot_colors'] or ([err_def['color']] * num_lines)

    xscale, yerr = args['xscale'], args['plot_yerr']
    if yerr:
        sensitivity_err = 'sensitivity_err_mario'
    else:
        sensitivity_err = None

    # main loop
    ax=None
    target_idx = all_targets.index(target)
    fields = ['output', 'target', column]
    for idx in range(num_lines):
        # load data
        log_data = autils.get_classification_logs_from_file(
            infiles[idx], fields=fields, target=target)
        for log in log_data:
            log[column] = column_type(log[column])

        # get classification stats binned
        group_df = pd.DataFrame(log_data).groupby(column)
        cm_iter = _cm_iter(group_df, all_targets)
        out_dict = autils.get_target_stats_binned(cm_iter, target_idx)

        # draw plot
        err_attrs = err_def.copy()
        err_attrs['color'] = fill_colors[idx]
        fill_attrs = fill_def.copy()
        # fill_attrs['edgecolor'] = fill_colors[idx]
        fig, ax, err = dutils.plot_efficiency_stat(
            out_dict, ax=ax, label=plot_labels[idx],
            plotted_stat='sensitivity', plotted_yerr_stat=sensitivity_err,
            errorbar_attrs=err_attrs, fill_between_attrs=fill_attrs,
            xlabel=column, xscale=xscale,
            ylabel='Sensitivity', ylim=(0,1.2),
            num_steps=20, figsize=(10,6),
            show=False)
    if add_legend:
        fig.legend(**legend_props)
    filename = '{}.svg'.format(outfile)
    dutils.save_figure(fig, filename)

if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_eval2sens as cmd

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    main(**args)

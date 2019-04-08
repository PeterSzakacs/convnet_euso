import collections
import os

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import libs.data_analysis as dutils
dutils.use('svg')
import utils.io_utils as io_utils


def get_target_stats(confusion_matrix, target_idx):
    num_classes = len(confusion_matrix)
    actual_targ_axis, predicted_targ_axis = 0, 1
    # sum the whole matrix
    total = np.sum(confusion_matrix)
    # sum along the diagonal
    hits = np.sum(confusion_matrix[x, x] for x in range(num_classes))

    tp = confusion_matrix[target_idx, target_idx]
    tn = hits - tp
    fp = np.sum(confusion_matrix, axis=actual_targ_axis)[target_idx] - tp
    fn = np.sum(confusion_matrix, axis=predicted_targ_axis)[target_idx] - tp

    return {
        'num_positive': tp + fn, 'num_negative': tn + fp,
        'num_true_positive': tp, 'num_true_negative': tn,
        'num_false_positive': fp, 'num_false_negative': fn,
    }


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
        # load data and get efficiency stats binned
        filename = infiles[idx]
        log_data = io_utils.load_TSV(filename, selected_columns=fields)
        log_data = [l for l in filter(
            lambda l: l['target'] == target, log_data)]
        for log in log_data:
            log[column] = column_type(log[column])

        # get efficiency stats binned
        group_df = pd.DataFrame(log_data).groupby(column)
        out_dict = collections.OrderedDict()
        for val, rows in group_df:
            cm = metrics.confusion_matrix(rows['target'], rows['output'],
                                        labels=all_targets)
            stats = get_target_stats(cm, target_idx)
            out_dict[val] = stats

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

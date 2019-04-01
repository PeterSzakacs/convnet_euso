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
    infile, logdir = args['infile'], args['logdir']
    column, column_type = args['column'], args['column_type']
    fields = ['output', 'target', column]
    target, all_targets = args['class_target'], args['all_targets']
    xscale = args['xscale']

    log_data = io_utils.load_TSV(infile, selected_columns=fields)
    log_data = [l for l in filter(lambda l: l['target'] == target, log_data)]

    for log in log_data:
        log[column] = column_type(log[column])
    group_df = pd.DataFrame(log_data).groupby(column)
    targ_idx = all_targets.index(target)
    out_dict = collections.OrderedDict()
    for val, rows in group_df:
        cm = metrics.confusion_matrix(rows['target'], rows['output'],
                                      labels=all_targets)
        stats = get_target_stats(cm, targ_idx)
        out_dict[val] = stats
    fig, ax, err = dutils.plot_efficiency_stat(
        out_dict,
        plotted_stat='sensitivity',
        plotted_yerr_stat='sensitivity_err_mario',
        num_steps=20,
        xscale=xscale,
        xlabel=column,
        ylabel='Sensitivity',
        figsize=(10,6),
        ylim=(0,1.2),
        show=False)
    filename = '{}_prediction_sensitivity_per_{}-{}.svg'.format(
        target, column, xscale)
    dutils.save_figure(fig, os.path.join(logdir, filename))

if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_eval2sens as cmd

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    main(**args)

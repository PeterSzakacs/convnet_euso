import pandas as pd
import sklearn.metrics as metrics

import dataset.constants as cons
import libs.data_analysis as dutils
dutils.use('svg')
import utils.analysis_utils as autils


def _cm_iter(group_df, all_targets, cm_func=metrics.confusion_matrix):
    for val, rows in group_df:
        yield val, cm_func(rows['target'], rows['output'], labels=all_targets)


def _get_fill_attrs_list(num_plots, **settings):
    fill_def = dutils.EFFICIENCY_STAT_FILL_BETWEEN_DEFAULTS
    plt_colors = settings.get('plot_colors')
    if plt_colors is None:
        return [fill_def] * num_plots
    else:
        return [{**fill_def, 'edgecolor': color} for color in plt_colors]


def _get_err_attrs_list(num_plots, **settings):
    err_def = dutils.EFFICIENCY_STAT_ERRORBAR_DEFAULTS
    plt_colors = settings.get('plot_colors')
    if plt_colors is None:
        return [err_def] * num_plots
    else:
        return [{**err_def, 'color': color} for color in plt_colors]


def _get_fontsizes(**settings):
    fs = {}
    fontsize = settings.get('fontsize') or 20
    fs['fontsize'] = fontsize
    fs['label_fontsize'] = settings.get('label_fontsize') or fontsize
    fs['legend_fontsize'] = settings.get('legend_fontsize') or fontsize
    ticks_fs = settings.get('ticks_fontsize')
    if ticks_fs is None:
        fs['ticks_fontsize'] = (fontsize - 3, fontsize - 5)
    else:
        fs['ticks_fontsize'] = (int(ticks_fs[0]), int(ticks_fs[1]))
    return fs


def _create_sensitivity_ax(attr_name, **settings):
    plt = dutils.plt
    xlabel = settings.get('xlabel') or attr_name
    ylabel = settings.get('ylabel') or 'Sensitivity'
    xscale = settings.get('xscale') or 'linear'
    plt.rcParams.update({'font.size': settings['fontsize']})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale(xscale); ax.set_ylim((0, 1.2))
    ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    label_size = settings['label_fontsize']
    ax.xaxis.label.set_size(label_size); ax.yaxis.label.set_size(label_size)
    ticks_sizes = settings['ticks_fontsize']
    ax.tick_params(axis='both', which='major', labelsize=ticks_sizes[0])
    ax.tick_params(axis='both', which='minor', labelsize=ticks_sizes[1])
    return ax


def _add_plot_legend(ax, handles, **settings):
    labels = settings.get('plot_labels')
    if labels is not None:
        loc = settings.get('legend_loc') or 'best'
        fontsize = settings.get('legend_fontsize') or settings['fontsize']
        ax.legend(handles, labels, loc=loc, fontsize=fontsize)


def main(**args):
    infiles, outfile = args['infiles'], args['outfile']
    column, column_type = args['column'], args['column_type']
    target = args['class_target']
    yerr = args.get('add_yerr')
    if yerr:
        sensitivity_err = 'sensitivity_err_mario'
    else:
        sensitivity_err = None
    num_plotlines = len(infiles)

    # set visualization settings
    err_attrs = _get_err_attrs_list(num_plotlines, **args)
    fill_attrs = _get_fill_attrs_list(num_plotlines, **args)
    fontsizes = _get_fontsizes(**args)

    # main loop
    ax = _create_sensitivity_ax(column, **{**args, **fontsizes})
    errorbars = []
    all_targets = list(cons.CLASSIFICATION_TARGETS)
    target_idx = all_targets.index(target)
    fields = ['output', 'target', column]
    for idx in range(num_plotlines):
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
        # fill_attrs['edgecolor'] = fill_colors[idx]
        fig, ax, err = dutils.plot_efficiency_stat(
            out_dict, ax=ax,
            plotted_stat='sensitivity', plotted_yerr_stat=sensitivity_err,
            errorbar_attrs=err_attrs[idx], fill_between_attrs=fill_attrs[idx],
            xscale=None, num_steps=20, show=False)
        errorbars.append(err)
    _add_plot_legend(ax, errorbars, **args)
    filename = '{}.svg'.format(outfile)
    dutils.save_figure(fig, filename)


if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_eval2sens as cmd

    # command line argument parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    main(**args)

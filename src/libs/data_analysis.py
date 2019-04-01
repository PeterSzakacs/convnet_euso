import collections
import functools
import math
import sys

import matplotlib
import numpy as np
import pandas as pd


def use(mpl_backend='svg'):
    """
    Specify matplotlib render backend to use (default: svg).

    This function must be called upon import of this module, otherwise pyplot
    will not be available for creating figures.

    Parameters
    ----------
    :param mpl_backend: matplotlib backend to use
    :type mpl_backend:  str
    """
    global plt
    matplotlib.use(mpl_backend)
    import matplotlib.pyplot as plt


def save_figure(figure, save_pathname):
    figure.savefig(save_pathname)
    plt.close(figure)


def calc_cls_stats_from_numbers(n):
    num_predicted_positive = (n['num_true_positive'] + n['num_false_positive'])
    num_predicted_negative = (n['num_true_negative'] + n['num_false_negative'])

    sensitivity = n['num_true_positive'] / n['num_positive'] if n['num_positive'] > 0 else None
    specificity = n['num_true_negative'] / n['num_negative'] if n['num_negative'] > 0 else None
    precision = n['num_true_positive'] / num_predicted_positive if num_predicted_positive > 0 else None
    negative_predictive_value = n['num_true_negative'] / num_predicted_negative if num_predicted_negative > 0 else None

    fallout = n['num_false_positive'] / n['num_negative'] if n['num_negative'] > 0 else None
    miss_rate = n['num_false_negative'] / n['num_positive'] if n['num_positive'] > 0 else None

    tot_num = n['num_positive'] + n['num_negative']

    accuracy = (n['num_true_positive'] + n['num_true_negative']) / tot_num if tot_num > 0 else None

    sensitivity_err_mario = (
                np.sqrt(((1 - sensitivity) / n['num_true_positive']) + (1 / n['num_positive'])) * sensitivity) \
        if sensitivity is not None and n['num_true_positive'] > 0 and n['num_positive'] > 0 else 1

    # positive_normal_approximation_interval_dict = dict()
    #
    # for confidence_interval in (0.68, 0.95, 0.997):
    #     alpha = 1 - confidence_interval
    #     # actual alpha
    #     # number of samples is important for typical confidence interval
    #     # needs to be looked up in a table to find a z score - area under bell curve for this value
    #     1 - (alpha / 2)
    #
    #     # this just simply is wrong
    #     # z = 1 - (1 - confidence_interval)/2
    #
    #     positive_normal_approximation_interval_dict[confidence_interval] = \
    #         ((z / n['num_positive']) * np.sqrt((n['num_true_positive']*n['num_false_negative'])/n['num_positive'])) \
    #             if n['num_positive'] > 0 else 1

    return {
        'sensitivity': sensitivity,
        'sensitivity_err_mario': sensitivity_err_mario,
        'specificity': specificity,
        'precision': precision,
        'negative_predictive_value': negative_predictive_value,
        'fallout': fallout,
        'miss_rate': miss_rate,
        'accuracy': accuracy,
        # **{'positive_normal_approx_interval_{:d}'.format(round(confidence_interval*100)): v \
        #     for confidence_interval, v in positive_normal_approximation_interval_dict.items()}
    }


def lod_to_dol(list_of_dicts, do_secondary=True, dict_class=collections.OrderedDict, list_class=list,
               skip_none_dict=True, skip_none_val=True,
               secondary_save_different=False, apply_func=None):
    od = dict_class()
    for d in list_of_dicts:
        if d is None and skip_none_dict:
            continue
        for k, v in d.items():
            if k not in od:
                od[k] = list_class()
            if v is not None or not skip_none_val:
                od[k].append(v)
    # what is the purpose of secondary?
    if do_secondary:
        #         print('>doing secondary')
        #         od_n = dict_class()
        for k in list(od.keys()):
            l = od[k]
            #             print('>',k,len(l))
            if len(l) > 0 and isinstance(l[0], dict):
                sec_od = dict_class()
                for d in l:
                    if d is None and skip_none_dict:
                        continue

                    #                     print('*'*30)
                    #                     print(d);
                    #                     print('*'*30)

                    for d_k, d_v in d.items():
                        if d_k not in sec_od:
                            sec_od[d_k] = list_class()
                        if v is not None or not skip_none_val:
                            sec_od[d_k].append(d_v)
                #                             print('sec_od[{}].append({})'.format(d_k,d_v))

                #                 print('od[{}] = sec_od'.format(k))
                #                 print('> len(sec_od)=',len(sec_od))
                #                 print(sec_od)

                od[k] = sec_od
            elif secondary_save_different:
                od[k] = l

    if callable(apply_func):
        for k, l in od.items():
            od[k] = apply_func(l)

    return od


def get_x_y_vals(table):
    x_vals = []
    y_vals = []
    for x_val, y_val in table:
        if y_val is not None and (not isinstance(y_val, (list, tuple, np.ndarray, pd.Series)) or len(y_val) > 0):
            x_vals.append(x_val)
            y_vals.append(y_val)
    return x_vals, y_vals


def get_thinned_datapoints_linear(x_vals, y_vals, num_steps, min_x=None, max_x=None):
    if min_x is None:
        min_x = x_vals[0]
    if max_x is None:
        max_x = x_vals[-1]

    x_val_step = (max_x - min_x) / num_steps

    r_i = 0
    cur_x_val_step = min_x
    cur_x_val = cur_x_val_step

    bins_x_vals = []
    bins_y_vals = []
    x_ranges_low = []
    x_ranges_high = []
    bin_ranges_low = []
    bin_ranges_high = []

    while cur_x_val_step <= max_x:
        bin_x_vals = []
        bin_y_vals = []

        while cur_x_val_step <= cur_x_val < cur_x_val_step + x_val_step:

            #             print(cur_x_val_step, cur_x_val, x_val_step, cur_x_val_step+x_val_step)

            bin_x_vals.append(cur_x_val)
            bin_y_vals.append(y_vals[r_i])

            if r_i + 1 >= len(x_vals):
                break
            r_i += 1
            cur_x_val = x_vals[r_i]

        if bin_y_vals:
            x_ranges_low.append(bin_x_vals[0])
            x_ranges_high.append(bin_x_vals[-1])

            bins_x_vals.append(bin_x_vals)
            bins_y_vals.append(bin_y_vals)

        #         print('>',cur_x_val_step)

        bin_ranges_low.append(cur_x_val_step)

        cur_x_val_step += x_val_step

        bin_ranges_high.append(cur_x_val_step)

    #         print('>',cur_x_val_step)

    return bins_x_vals, bins_y_vals, x_ranges_low, x_ranges_high, bin_ranges_low, bin_ranges_high


def get_thinned_datapoints_log(x_vals, y_vals, num_steps, log_base=10, min_x=None, max_x=None):
    if min_x is None:
        min_x = x_vals[0]
    if max_x is None:
        max_x = x_vals[-1]

    x_val_step = (max_x - min_x) / num_steps  # TODO dynamically

    max_exponent = math.log(max_x, log_base)
    min_exponent = math.log(min_x, log_base)
    exponent_step = (max_exponent - min_exponent) / num_steps

    #     print('->',min_x, max_x, max_exponent, exponent_step)

    r_i = 0

    cur_x_val_step = min_x
    cur_x_val = cur_x_val_step

    #     cur_exponent_step = math.log(min_x, log_base)
    #     cur_exponent = cur_exponent_step

    bins_x_vals = []
    bins_y_vals = []
    binned_x_ranges_low = []
    binned_x_ranges_high = []
    bin_ranges_low = []
    bin_ranges_high = []

    while cur_x_val_step <= max_x:
        bin_x_vals = []
        bin_y_vals = []

        next_exponent_step = math.log(cur_x_val_step, log_base) + exponent_step

        # while cur_x_val_step <= cur_x_val < cur_x_val_step+x_val_step:
        while cur_x_val_step <= cur_x_val \
                and math.log(cur_x_val, log_base) < next_exponent_step:

            #             print(
            #                 cur_x_val_step , cur_x_val ,
            #                 math.log(cur_x_val, log_base) ,
            #                 next_exponent_step,
            #             )

            bin_x_vals.append(cur_x_val)
            bin_y_vals.append(y_vals[r_i])

            if r_i + 1 >= len(x_vals):
                break
            r_i += 1
            cur_x_val = x_vals[r_i]

        #             time.sleep(1)

        if bin_y_vals:
            binned_x_ranges_low.append(bin_x_vals[0])
            binned_x_ranges_high.append(bin_x_vals[-1])

            bins_x_vals.append(bin_x_vals)
            bins_y_vals.append(bin_y_vals)

        #         print('>', cur_x_val_step)

        bin_ranges_low.append(cur_x_val_step)

        cur_x_val_step = log_base ** next_exponent_step

        bin_ranges_high.append(cur_x_val_step)

    #         print('>', cur_x_val_step)
    #         time.sleep(1)

    return bins_x_vals, bins_y_vals, binned_x_ranges_low, binned_x_ranges_high, bin_ranges_low, bin_ranges_high


def get_efficiency_stat_plot_data(
        stats_by_attr_dict, plotted_stat='sensitivity', plotted_yerr_stat='sensitivity_err_mario',
        num_steps=36, default_yerr=1,
        xscale='linear', yscale='linear', xscale_binning=None, xtranslate_func=None,
        do_xaxis_weights=True, xaxis_weight_reduce_func=np.sum, xaxis_weight_stat='num_positive',
        concat_dicts=True, dict_stats_reduce_func=np.mean, dict_stats_yerr_reduce=('y', np.std),
        bin_y_vals_reduce=np.sum,
        calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers):
    '''
    dict_stats_yerr_mode: string|callable|tuple
        'std_y'  or  'minmax_y'  or   callable (applied to yerr)   or tuple('y' or 'yerr', callable)
    '''

    if xscale_binning is None:
        xscale_binning = xscale

    if concat_dicts and isinstance(stats_by_attr_dict, (list, tuple)):
        stats_by_attr_dict = lod_to_dol(stats_by_attr_dict)

    if isinstance(stats_by_attr_dict, (list, tuple)):
        stats_by_attr_dict_list = stats_by_attr_dict
    else:
        stats_by_attr_dict_list = [stats_by_attr_dict]

    plot_entries_list = []

    #     plot_x_list = []
    #     plot_y_list = []
    #     plot_xerr_list = []
    #     plot_yerr_list = []

    plot_xbin_entries_dict = collections.OrderedDict()

    min_x = None
    max_x = None

    for stats_by_attr_dict in stats_by_attr_dict_list:
        t_min_x = np.min(list(stats_by_attr_dict.keys()))
        t_max_x = np.max(list(stats_by_attr_dict.keys()))
        if min_x is None or min_x > t_min_x:
            min_x = t_min_x
        if max_x is None or max_x < t_max_x:
            max_x = t_max_x

    #print('min_x=',min_x)
    #print('max_x=',max_x)
    #print('len(stats_by_attr_dict_list)=',len(stats_by_attr_dict_list))

    # following definitions are mostly just to prevent IDE complains
    plot_x = []
    plot_y = []
    plot_xerr = [[], []]
    plot_yerr = [[], []]
    stats_dict = {}
    x_vals_weights = None  # functionally important

    for stats_by_attr_dict in stats_by_attr_dict_list:

        x_vals, y_vals = get_x_y_vals(stats_by_attr_dict.items())

        if xscale_binning == 'log':
            bins_x_vals, bins_y_vals, binned_x_ranges_low, binned_x_ranges_high, bin_ranges_low, bin_ranges_high = \
                get_thinned_datapoints_log(x_vals, y_vals, num_steps, min_x=min_x, max_x=max_x)
        else:
            if xscale_binning != 'linear':
                print('Using linear xscale binning!', file=sys.stderr)
            bins_x_vals, bins_y_vals, binned_x_ranges_low, binned_x_ranges_high, bin_ranges_low, bin_ranges_high = \
                get_thinned_datapoints_linear(x_vals, y_vals, num_steps, min_x=min_x, max_x=max_x)

        plot_x = []
        plot_y = []
        plot_xerr = [[], []]
        plot_yerr = [[], []]

        #     _tp = []
        #     _p = []

        #print('len(bins_x_vals)',len(bins_x_vals))
        #print('len(bins_y_vals)',len(bins_y_vals))

        for bin_x_vals, bin_y_vals, x_low, x_high, bin_low, bin_high in zip(bins_x_vals, bins_y_vals,
                                                                            binned_x_ranges_low, binned_x_ranges_high,
                                                                            bin_ranges_low, bin_ranges_high):

            #print('>len(bin_x_vals)',len(bin_x_vals))
            #print('>len(bin_y_vals)',len(bin_y_vals))

            sum_num_dict = lod_to_dol(bin_y_vals, apply_func=bin_y_vals_reduce, do_secondary=False)

            #print(sum_num_dict['num_true_positive'])
            #print(sum_num_dict['num_positive'])
            #         _tp.append(sum_num_dict['num_true_positive'])
            #         _p.append(sum_num_dict['num_positive'])

            if plotted_stat not in sum_num_dict or (
                    plotted_yerr_stat is not None and plotted_yerr_stat not in sum_num_dict):
                stats_dict = calc_cls_stats_from_numbers_func(sum_num_dict)

            if plotted_stat not in sum_num_dict:
                stat_val = stats_dict[plotted_stat]
            else:
                stat_val = sum_num_dict[plotted_stat]

            if stat_val is not None:

                x_vals_weights = None
                if do_xaxis_weights:
                    assert (xaxis_weight_stat is not None and xaxis_weight_reduce_func is not None)
                    y_vals = [d[xaxis_weight_stat] for d in bin_y_vals]
                    x_vals_weights = []
                    for y_vals_list in y_vals:
                        x_vals_weights.append(
                            xaxis_weight_reduce_func(y_vals_list) \
                                if y_vals_list is not None else 0)

                yerr_val = default_yerr

                if plotted_yerr_stat is not None:
                    if plotted_yerr_stat not in sum_num_dict:
                        yerr_val = stats_dict[plotted_yerr_stat]
                    else:
                        yerr_val = sum_num_dict[plotted_yerr_stat]

                x_low_high_tuple = (bin_low, bin_high)  # Intentionally using bin range

                if x_low_high_tuple not in plot_xbin_entries_dict:
                    plot_xbin_entries_dict[x_low_high_tuple] = []

                if len(stats_by_attr_dict_list) > 1:
                    plot_xbin_entries_dict[x_low_high_tuple].append(
                        (bin_x_vals, stat_val, x_vals_weights, yerr_val)
                    )

                else:
                    if x_vals_weights is not None and np.sum(x_vals_weights) != 0:
                        plot_x_val = np.average(bin_x_vals, weights=x_vals_weights)
                    else:
                        plot_x_val = np.average(bin_x_vals)

                    xerr_low = plot_x_val - x_low
                    xerr_high = x_high - plot_x_val

                    plot_x.append(plot_x_val)
                    plot_xerr[0].append(xerr_low)
                    plot_xerr[1].append(xerr_high)

                    plot_y.append(stat_val)
                    # plot_yerr[0].append(yerr_val)
                    # plot_yerr[1].append(yerr_val)

                    if isinstance(yerr_val, (list, tuple, np.ndarray)):
                        yerr_val_low = yerr_val[0]
                        yerr_val_high = yerr_val[1]
                    else:
                        yerr_val_low = yerr_val_high = yerr_val

                    plot_yerr[0].append(yerr_val_low)
                    plot_yerr[1].append(yerr_val_high)

            # endif  (stat_val is not None)
        # endfor

    #     print('len(plot_xbin_entries_dict)=',len(plot_xbin_entries_dict))
    #     pprint.pprint(list(plot_xbin_entries_dict.keys()))

    if len(stats_by_attr_dict_list) > 1 and len(plot_xbin_entries_dict) > 0:
        plot_x = []
        plot_y = []
        plot_xerr = [[], []]
        plot_yerr = [[], []]

        plot_xbin_entries_dict_items_sorted = sorted(plot_xbin_entries_dict.items())

        for (bin_low, bin_high), bin_entries in plot_xbin_entries_dict_items_sorted:

            # (bin_x_vals, stat_val, x_vals_weights, yerr_val)

            bin_x_vals_all = functools.reduce(lambda a, b: a + b, [entry[0] for entry in bin_entries])

            x_vals_weights_all = None
            if x_vals_weights is not None and np.sum(x_vals_weights) != 0:
                x_vals_weights_all = functools.reduce(lambda a, b: a + b, [entry[2] for entry in bin_entries])
                #                 print('bin_x_vals_all')
                #                 pprint.pprint(bin_x_vals_all)
                #                 print('x_vals_weights_all')
                #                 pprint.pprint(x_vals_weights_all)
                plot_x_val = np.average(bin_x_vals_all, weights=x_vals_weights_all)
            #                 print('+'*50)
            else:
                plot_x_val = np.average(bin_x_vals_all)

            plot_y_list = [entry[1] for entry in bin_entries]
            plot_y_val = dict_stats_reduce_func(plot_y_list)

            plot_xerr_val_0 = plot_x_val - np.min(bin_x_vals_all)
            plot_xerr_val_1 = np.max(bin_x_vals_all) - plot_x_val

            plot_yerr_val_0 = plot_yerr_val_1 = default_yerr

            if dict_stats_yerr_reduce == 'std_y':
                plot_yerr_val_0 = np.std(plot_y_list)
                plot_yerr_val_1 = plot_yerr_val_0
            elif dict_stats_yerr_reduce == 'minmax_y':
                plot_yerr_val_0 = plot_y_val - np.min(plot_y_list)
                plot_yerr_val_1 = np.max(plot_y_list) - plot_y_val
            elif isinstance(dict_stats_yerr_reduce, tuple) and dict_stats_yerr_reduce[0] == 'y':
                r = dict_stats_yerr_reduce[1](plot_y_list)
                if isinstance(r, tuple):
                    plot_yerr_val_0, plot_yerr_val_1 = r
                else:
                    plot_yerr_val_0 = r
                    plot_yerr_val_1 = plot_yerr_val_0

            else:
                # plot_yerr_list = [entry[3] for entry in bin_entries]

                plot_yerr_val_0_list = []
                plot_yerr_val_1_list = []

                for entry in bin_entries:
                    if isinstance(entry[3], (list, tuple, np.ndarray)):
                        plot_yerr_val_0_list.append(entry[3][0])
                        plot_yerr_val_1_list.append(entry[3][1])
                    else:
                        plot_yerr_val_0_list.append(entry[3])
                        plot_yerr_val_1_list.append(entry[3])

                if plotted_yerr_stat is not None:
                    if dict_stats_yerr_reduce == 'avg_yerr_weighted':
                        yerr_weights = [np.sum(entry[2]) for entry in bin_entries]
                        plot_yerr_val_0 = np.average(plot_yerr_val_0_list, weights=yerr_weights)
                        plot_yerr_val_1 = np.average(plot_yerr_val_1_list, weights=yerr_weights)
                    elif dict_stats_yerr_reduce == 'minmax_yerr':
                        plot_yerr_val_0 = np.min(plot_yerr_val_0_list)
                        plot_yerr_val_1 = np.max(plot_yerr_val_1_list)
                    else:
                        yerr_reduce_func = None
                        if callable(dict_stats_yerr_reduce):
                            yerr_reduce_func = dict_stats_yerr_reduce
                        elif isinstance(dict_stats_yerr_reduce, tuple) and dict_stats_yerr_reduce[0] == 'yerr':
                            yerr_reduce_func = dict_stats_yerr_reduce[1]
                        if yerr_reduce_func is not None:
                            plot_yerr_val_0 = yerr_reduce_func(plot_yerr_val_0_list)
                            plot_yerr_val_1 = yerr_reduce_func(plot_yerr_val_1_list)

                            # r = yerr_reduce_func(plot_yerr_list)
                            # if isinstance(r, tuple):
                            #     plot_yerr_val_0, plot_yerr_val_1 = r
                            # else:
                            #     plot_yerr_val_0 = r
                            #     plot_yerr_val_1 = plot_yerr_val_0
                        else:
                            plot_yerr_val_0 = plot_yerr_val_1 = None
                else:
                    plot_yerr_val_0 = plot_yerr_val_1 = None

            plot_x.append(plot_x_val)
            plot_y.append(plot_y_val)
            plot_xerr[0].append(plot_xerr_val_0)
            plot_xerr[1].append(plot_xerr_val_1)
            plot_yerr[0].append(plot_yerr_val_0)
            plot_yerr[1].append(plot_yerr_val_1)

    if plotted_yerr_stat is not None:
        plot_yerr = [(v if v is not None else default_yerr) for v in plot_yerr]
    else:
        plot_yerr = None

    if callable(xtranslate_func):
        plot_x = xtranslate_func(plot_x)
        plot_xerr[0] = xtranslate_func(plot_xerr[0])
        plot_xerr[1] = xtranslate_func(plot_xerr[1])

    return plot_x, plot_y, plot_xerr, plot_yerr


_EFFICIENCY_STAT_ERRORBAR_DEFAULTS = dict(
        marker='.',
        linestyle='-',
        color='black',
        ecolor='silver'
    )

if 'EFFICIENCY_STAT_ERRORBAR_DEFAULTS' in globals():
    EFFICIENCY_STAT_ERRORBAR_DEFAULTS = {**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, **_EFFICIENCY_STAT_ERRORBAR_DEFAULTS}
else:
    EFFICIENCY_STAT_ERRORBAR_DEFAULTS = _EFFICIENCY_STAT_ERRORBAR_DEFAULTS


EFFICIENCY_STAT_FILL_BETWEEN_DEFAULTS = dict(
    facecolor='#dddddd'
)

def plot_efficiency_stat(
        stats_by_attr_dict, plotted_stat='sensitivity', plotted_yerr_stat='sensitivity_err_mario',
        num_steps=36, default_yerr=1,
        xscale='linear', yscale='linear', xscale_binning=None, xtranslate_func=None,
        do_xaxis_weights=True, xaxis_weight_reduce_func=np.sum, xaxis_weight_stat='num_positive',
        concat_dicts=True, dict_stats_reduce_func=np.mean, dict_stats_yerr_reduce=('y', np.std),
        bin_y_vals_reduce=np.sum,
        xlabel=None, ylabel=None, label='',
        ylim=None, xlim=None,
        figsize=(15, 8),
        errorbar_attrs=EFFICIENCY_STAT_ERRORBAR_DEFAULTS,
        fill_between_attrs=EFFICIENCY_STAT_FILL_BETWEEN_DEFAULTS,
        show=True, ax=None,
        calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers,
        show_fill_between=True, show_yerr=False, show_xerr=True,
        ret_fill_between=False
):
    '''
    dict_stats_yerr_mode: string|callable|tuple
        'std_y'  or  'minmax_y'  or   callable (applied to yerr)   or tuple('y' or 'yerr', callable)
    '''

    plot_x, plot_y, plot_xerr, plot_yerr = \
        get_efficiency_stat_plot_data(
            stats_by_attr_dict, plotted_stat, plotted_yerr_stat,
            num_steps, default_yerr,
            xscale, yscale, xscale_binning, xtranslate_func,
            do_xaxis_weights, xaxis_weight_reduce_func, xaxis_weight_stat,
            concat_dicts, dict_stats_reduce_func, dict_stats_yerr_reduce,
            bin_y_vals_reduce, calc_cls_stats_from_numbers_func)


    #         print(plot_y)
    #         print(plot_xerr)
    #         print(plot_yerr)

    #     print(plot_x[:3])
    #     print(plot_y[:3])
    #     print(_tp[:3])
    #     print(_p[:3])
    #     print(plot_yerr[:3])
    #     print('-'*50)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    #     print('plot_x')
    #     print(plot_x)
    #     print('plot_y')
    #     print(plot_y)
    #     print('plot_xerr')
    #     print(plot_xerr)
    #     print('plot_yerr')
    #     print(plot_yerr)

    if show_fill_between and plotted_yerr_stat is not None:
        fbtwn = ax.fill_between(plot_x, np.array(plot_y) - plot_yerr[0], np.array(plot_y) + plot_yerr[1],
                                **fill_between_attrs)

    errbr_params_dict = dict()
    if show_yerr:
        errbr_params_dict['yerr'] = plot_yerr
    if show_xerr:
        errbr_params_dict['xerr'] = plot_xerr

    errbr = ax.errorbar(plot_x, plot_y, label=label, **errbr_params_dict, **errorbar_attrs)

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if figsize is not None:
        fig.set_size_inches(figsize)
    if show:
        plt.show()

    if ret_fill_between:
        return fig, ax, errbr, fbtwn
    else:
        return fig, ax, errbr

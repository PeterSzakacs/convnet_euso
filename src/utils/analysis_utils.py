import collections

import numpy as np

import utils.io_utils as io_utils


def get_target_stats_binned(confusion_matrices_iter, target_idx):
    out_dict = collections.OrderedDict()
    for label, cm in confusion_matrices_iter:
        stats = get_target_stats(cm, target_idx)
        out_dict[label] = stats
    return out_dict


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


def get_classification_logs_from_file(filename, fields=None, target=None):
    # load data and get efficiency stats binned
    log_data = io_utils.load_TSV(filename, selected_columns=fields)
    if target is not None:
        log_data = [log for log in filter(
            lambda log: log['target'] == target, log_data)]
    return log_data


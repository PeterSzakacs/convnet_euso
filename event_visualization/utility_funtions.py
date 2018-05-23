import numpy as np
import os
from argparse import ArgumentTypeError
import collections


def translate_struct(struct, trans_func):
    """
    Maps all Tasks in a structured data object to their .output().
    """
    if isinstance(struct, (dict, collections.OrderedDict)):
        r = type(struct)()
        for k, v in struct.items():
            r[k] = translate_struct(v, trans_func)
        return r
    elif isinstance(struct, (list, tuple)):
        try:
            s = list(struct)
        except TypeError:
            raise Exception('Cannot map %s to list' % str(struct))
        return [translate_struct(r, trans_func) for r in s]

    return trans_func(struct)


def translate_dict_keys(d, trans_func):
    od = {}
    for k,v in d.items():
        od[trans_func(k)] = v
    return od


def get_field_positions(arr, cond_func):
    o = []
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        if cond_func(arr[it.multi_index]):
            o.append(it.multi_index)
        it.iternext()
    return o


def find_n_max_values(arr, n):
    # inefficient !!!
    if n <= 0:
        raise Exception("Parameter value n has to be larger than 0")
    max_value_positions = [None]*n #collections.deque(maxlen=n)
    it = np.nditer(arr, flags=['multi_index'])
    i = 0
    while i < n and not it.finished:
        max_value_positions[i] = it.multi_index
        it.iternext()
        i += 1
    it.reset()
    while not it.finished:
        append_index = False
        smallest_val_pos = None
        for i, pos in enumerate(max_value_positions):
            if arr[it.multi_index] > arr[pos]:
                append_index = True
                if smallest_val_pos is None or arr[pos] < arr[max_value_positions[smallest_val_pos]]:
                    smallest_val_pos = i
        if append_index:
            max_value_positions[smallest_val_pos] = it.multi_index
        it.iternext()
    return max_value_positions


def split_all_filed_values_to_groups(arr, ignored_values=[]):
    groups = {}

    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        if arr[it.multi_index] not in ignored_values:
            if arr[it.multi_index] not in groups:
                groups[arr[it.multi_index]] = []
            groups[arr[it.multi_index]].append(it.multi_index)
        it.iternext()

    return groups


def split_values_to_groups(points, arr):
    groups = {}

    for point in points:
        if arr[point] not in groups:
            groups[arr[point]] = []
        groups[arr[point]].append(point)

    return groups


def key_vals2val_keys(in_dict, exclusive=False):
    out_dict = {}
    for k,l in in_dict.items():
        for v in l:
            if exclusive:
                out_dict[v] = k
            else:
                if v not in out_dict:
                    out_dict[v] = []
                out_dict[v].append(v)
    return out_dict


def prepare_pathname(format_str, name, exist_ok=True, **kwargs):
    if format_str is None:
        return None
    formatted_pathname = format_str.format(name=name, **kwargs)
    dirpath = os.path.realpath(os.path.dirname(formatted_pathname))
    os.makedirs(dirpath, exist_ok=exist_ok)
    return os.path.join(dirpath,os.path.basename(formatted_pathname))


def get_str2bool_func(exc_type=Exception):
    def str2bool(v):
        if isinstance(v, (int,bool,float)) or v is None:
            return bool(v)
        if v.strip().lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.strip().lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise exc_type('Boolean value expected.')
    return str2bool


str2bool_argparse = get_str2bool_func(ArgumentTypeError)
str2bool = get_str2bool_func(Exception)


def bool2yesno(bool_val):
    return 'yes' if bool_val else 'no'


def bool2yn(bool_val):
    return 'y' if bool_val else 'n'

def merge_config_and_run_args(run_args, params, arg_names, arg_name_prefix):
    for arg_entry in arg_names:
        if isinstance(arg_entry, tuple) and len(arg_entry) > 1:
            arg_name = arg_entry[0]
            param_name = arg_entry[1]
        else:
            arg_name = str(arg_entry)
            param_name = arg_name
        if arg_name_prefix:
            arg_name = arg_name_prefix + arg_name
        arg_val = getattr(run_args, arg_name)
        if arg_val is not None:
            setattr(params, param_name, arg_val)
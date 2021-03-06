import importlib
import datetime as dt
import random

import numpy as np

import dataset.target_utils as targ
import net.constants as net_cons

CLASSIFICATION_FIELDS   = ['item_idx', 'output', 'target', 'shower_prob',
                           'noise_prob']

def _classification_fields_handler(raw_output, target, item_idx,
                                   old_dict=None):
    if old_dict == None:
        old_dict = {}
    probs = targ.get_target_probabilities(raw_output)
    old_dict['shower_prob'] = probs['shower']
    old_dict['noise_prob']  = probs['noise']
    rnd_output = np.round(raw_output).astype(np.uint8)
    old_dict['output'] = targ.get_target_name(rnd_output)
    old_dict['target'] = targ.get_target_name(target)
    old_dict['item_idx'] = item_idx
    return old_dict


def get_default_run_id(network_module_name):
    current_time = dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return '{}_{}'.format(network_module_name, current_time)


# dataset input functions (bind data to inputs, etc.)


def convert_dataset_items_to_model_inputs(model, data_dict,
                                          create_getter=False):
    inputs_dict = {}
    input_spec = model.network_graph.input_spec
    for input_name, spec in input_spec.items():
        inputs_dict[input_name] = data_dict[spec['item_type']]
    item_getter = lambda data, i_slice: {k: d[i_slice]
                                         for k, d in data.items()}
    if create_getter:
        return inputs_dict, item_getter
    else:
        return inputs_dict


def convert_to_model_inputs_dict(model, available_data):
    inputs_dict = {}
    input_spec = model.network_graph.input_spec
    for input_name, spec in input_spec.items():
        item_type = spec['item_type']
        inputs_dict[input_name] = available_data['data'][item_type]
    return inputs_dict


def convert_to_model_outputs_dict(model, available_data):
    targets = {}
    output_spec = model.network_graph.output_spec
    for output_name, spec in output_spec.items():
        if spec['location'] == 'targets':
            targets = available_data['targets']
        elif spec['location'] == 'data':
            item_type = spec['item_type']
            targets = available_data['data'][item_type]
    return targets


# model functions (import, train, evaluate, save, etc.)


def import_model(module_name, input_shapes, model_file=None, **optsettings):
    network_module = importlib.import_module(module_name)
    model = network_module.create_model(input_shapes, **optsettings)
    if model_file != None:
        model.load_from_file(model_file, **optsettings)
    return model


class DatasetSplitter:

    ALLOWED_OUTPUT_FORMATS = ('FLAT', 'PER_SET', 'PER_TYPE', )

    def __init__(self, split_mode, items_fraction=0.1, num_items=None):
        self.split_mode = split_mode
        self.test_items_fraction = items_fraction
        self.test_items_count = num_items

    @property
    def split_mode(self):
        return self._mode

    @split_mode.setter
    def split_mode(self, value):
        val = value.upper()
        if val not in net_cons.DATASET_SPLIT_MODES:
            raise ValueError('Invalid split mode {}, choose one of {}'.format(
                value, net_cons.DATASET_SPLIT_MODES
            ))
        self._mode = val

    @property
    def test_items_count(self):
        return self._count

    @test_items_count.setter
    def test_items_count(self, value):
        if value is not None:
            self._count = int(value)
        else:
            self._count = value

    @property
    def test_items_fraction(self):
        return self._frac

    @test_items_fraction.setter
    def test_items_fraction(self, value):
        frac = float(value)
        if frac >= 1:
            raise ValueError('Invalid fraction {}, must be less than 1'.format(
                frac
            ))
        if frac < 0:
            raise ValueError('Invalid fraction {}, cannot be negative'.format(
                frac
            ))
        self._frac = frac

    def get_train_test_indices(self, n_data):
        n_test = self.test_items_count or round(
            self.test_items_fraction * n_data)
        n_train = n_data - n_test
        mode = self.split_mode
        if mode == 'FROM_START':
            test_idx, train_idx = range(n_test), range(n_test, n_data)
        elif mode == 'FROM_END':
            test_idx, train_idx = range(n_train, n_data), range(n_train)
        elif mode == 'RANDOM':
            test_idx, next_idx = [], None
            all_idx = set(range(n_data))
            for idx in range(n_test):
                while next_idx not in all_idx:
                    next_idx = random.randrange(0, n_data)
                all_idx.remove(next_idx)
                test_idx.append(next_idx)
            train_idx = list(all_idx)
        return list(train_idx), list(test_idx)

    def get_data_and_targets(self, train_dset, test_dset=None,
                             dict_format='FLAT'):
        if (not isinstance(dict_format, str) or
                dict_format.upper() not in self.ALLOWED_OUTPUT_FORMATS):
            raise ValueError(f"Unknown dict format: {dict_format}, "
                             f"allowed values: {self.ALLOWED_OUTPUT_FORMATS}")
        dict_format = dict_format.upper()
        train_idx, test_idx = None, None
        if test_dset is None:
            test_dset = train_dset
            n_data = train_dset.num_data
            train_idx, test_idx = self.get_train_test_indices(n_data)
        train_data = train_dset.get_data_as_dict(train_idx)
        train_targets = train_dset.get_targets(train_idx)
        test_data = test_dset.get_data_as_dict(test_idx)
        test_targets = test_dset.get_targets(test_idx)
        if dict_format == 'FLAT':
            return {
                'train_data': train_data, 'train_targets': train_targets,
                'test_data': test_data, 'test_targets': test_targets,
            }
        elif dict_format == 'PER_SET':
            return {
                "train": {
                    "data": train_data, "targets": train_targets,
                },
                "test": {
                    "data": test_data, "targets": test_targets,
                }
            }
        elif dict_format == 'PER_TYPE':
            return {
                "data": {
                    "train": train_data, "test": test_data,
                },
                "targets": {
                    "train": train_targets, "test": test_targets,
                }
            }

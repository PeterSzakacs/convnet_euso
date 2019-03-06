import os
import importlib
import datetime as dt
import random

import tflearn
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


# dataset functions (reshape data, etc.)


def reshape_data_for_convnet(data, num_channels=1, create_getter=False):
    data_reshaped = []
    for item in data:
        item_np = np.array(item)
        item_shape = item_np[0].shape
        data_reshaped.append(item_np.reshape(-1, *item_shape, num_channels))
    item_getter = lambda data, i_slice: tuple(d[i_slice] for d in data)
    # tflearn does not seem to like single element sequences,
    # (the single element within is the actual input data).
    if len(data_reshaped) == 1:
        data_reshaped = data_reshaped[0]
        item_getter = lambda data, i_slice: data[i_slice]
    if create_getter:
        return data_reshaped, item_getter
    else:
        return data_reshaped


def convert_item_shapes_to_convnet_input_shapes(dataset, batch_size=None):
    item_shapes = dataset.item_shapes
    return {k:[batch_size, *v, 1] for k,v in item_shapes.items()
            if v is not None}


# model functions (import, train, evaluate, save, etc.)


def import_model(module_name, input_shapes, model_file=None, **optsettings):
    network_module = importlib.import_module(module_name)
    model = network_module.create_model(input_shapes, **optsettings)
    if model_file != None:
        model.load_from_file(model_file, **optsettings)
    return model


def evaluate_classification_model(model, dataset, items_slice=None,
                                  batch_size=128):
    items_slice = items_slice or slice(0, None)
    data = dataset.get_data_as_arraylike(items_slice)
    targets = dataset.get_targets(items_slice)
    metadata = dataset.get_metadata(items_slice)
    data, item_getter = reshape_data_for_convnet(data, create_getter=True)
    log_data = []

    # TODO: might want to simplify these indexes or at least give better names
    start, stop = items_slice.start, items_slice.stop
    stop = stop or dataset.num_data
    for idx in range(start, stop, batch_size):
        rel_idx = idx - start
        items_slice = slice(rel_idx, rel_idx + batch_size)
        data_batch = item_getter(data, items_slice)
        predictions = model.predict(data_batch)
        for pred_idx in range(len(predictions)):
            prediction = predictions[pred_idx]
            abs_idx = rel_idx + pred_idx
            log_item = _classification_fields_handler(
                prediction, targets[abs_idx], idx + pred_idx,
                old_dict=metadata[abs_idx].copy())
            log_data.append(log_item)
    return log_data


def save_model(model, save_pathname):
    model.save(save_pathname)


class DatasetSplitter():

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

    def get_data_and_targets(self, train_dset, test_dset=None):
        train_idx, test_idx = None, None
        if test_dset is None:
            test_dset = train_dset
            n_data = train_dset.num_data
            train_idx, test_idx = self.get_train_test_indices(n_data)
        return {'train_data': train_dset.get_data_as_arraylike(train_idx),
                'train_targets': train_dset.get_targets(train_idx),
                'test_data': test_dset.get_data_as_arraylike(test_idx),
                'test_targets': test_dset.get_targets(test_idx)}


class TfModelTrainer():

    DEFAULT_OPTIONAL_SETTINGS = {
        'batch_size': None, 'validation_batch_size': None, 'show_metric': True,
        'snapshot_step': 100,
    }

    def __init__(self, data_dict, num_epochs=11, **optsettings):
        self.train_test_data = data_dict
        self.default_num_epochs = num_epochs
        self._settings = self.DEFAULT_OPTIONAL_SETTINGS.copy()
        self.optional_settings = optsettings

    def _validate_setting(self, key, value):
        if key.endswith('batch_size') or key == 'snapshot_step':
            if value is None:
                return None
            else:
                return int(value)
        elif key == 'show_metric':
            return bool(value)

    def _get_new_settings_dict(self, **settings):
        old_settings = self._settings
        new_settings = {}
        for key, val in old_settings.items():
            try:
                # cannot use optsettings.get(key, default_value=None), or we
                # could not set some settings to None
                new_settings[key] = self._validate_setting(key, settings[key])
            except KeyError:
                new_settings[key] = val
        return new_settings

    @property
    def train_test_data(self):
        return self._data_dict

    @train_test_data.setter
    def train_test_data(self, values):
        new_data_dict = {}
        for k in net_cons.TRAIN_DATA_DICT_KEYS:
            new_data_dict[k] = values[k]
        self._data_dict = new_data_dict

    @property
    def default_num_epochs(self):
        return self._epochs

    @default_num_epochs.setter
    def default_num_epochs(self, value):
        self._epochs = int(value)

    @property
    def optional_settings(self):
        return self._settings

    @optional_settings.setter
    def optional_settings(self, values):
        self._settings = self._get_new_settings_dict(**values)

    def train_model(self, model, data_dict=None, num_epochs=None, run_id=None,
                    **optsettings):
        data = data_dict or self._data_dict
        tr_data, tr_targets = data['train_data'], data['train_targets']
        te_data, te_targets = data['test_data'], data['test_targets']
        epochs = num_epochs or self.default_num_epochs
        settings = self._get_new_settings_dict(**optsettings)
        run_id = run_id or get_default_run_id(model.network_graph.__module__)

        tf_model = model.network_model
        tf_model.fit(tr_data, tr_targets, n_epoch=epochs, run_id=run_id,
                     validation_set=(te_data, te_targets), **settings)

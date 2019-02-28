import os
import importlib
import datetime as dt
import random

import tflearn
import numpy as np

import utils.dataset_utils as ds
import utils.metadata_utils as meta
import utils.target_utils as targ

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


DEFAULT_CHECKING_LOGDIR = '/run/user/{}/convnet_checker'.format(os.getuid())
DEFAULT_TRAINING_LOGDIR = '/run/user/{}/convnet_trainer'.format(os.getuid())


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


# model functions (import, train, evaluate, save, etc.)


def import_convnet(module_name, tb_dir, input_shapes, model_file=None,
                   optimizer=None, loss_fn=None, learning_rate=None,
                   tb_verbosity=0):
    network_module = importlib.import_module(module_name)
    shapes = {k:[None, *v, 1] for k,v in input_shapes.items() if v is not None}
    network, conv_layers, fc_layers = network_module.create(
        inputShape=shapes, learning_rate=learning_rate, optimizer=optimizer,
        loss_fn=loss_fn
    )
    model = tflearn.DNN(network, tensorboard_verbose=tb_verbosity,
                        tensorboard_dir=tb_dir)
    if model_file != None:
        model.load(model_file)
    return model, network, conv_layers, fc_layers


def import_model(module_name, input_shapes, model_file=None, **optsettings):
    network_module = importlib.import_module(module_name)
    shapes = {k:[None, *v, 1] for k,v in input_shapes.items() if v is not None}
    model = network_module.create_model(shapes, **optsettings)
    if model_file != None:
        model.load_from_file(model_file, **optsettings)
    return model


def train_model(model, data_dict, run_id=None, num_epochs=11, step=100,
                metric=True):
    tr_data, tr_targets = data_dict['train_data'], data_dict['train_targets']
    te_data, te_targets = data_dict['test_data'], data_dict['test_targets']

    run_id = run_id or get_default_run_id(model.network_graph.__module__)
    tf_model = model.network_model
    tf_model.fit(tr_data, tr_targets, n_epoch=num_epochs, run_id=run_id,
                 validation_set=(te_data, te_targets), snapshot_step=step,
                 show_metric=metric)


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

    DATASET_SPLIT_MODES = ('FROM_START', 'FROM_END', 'RANDOM', )

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
        if val not in self.DATASET_SPLIT_MODES:
            raise ValueError('Invalid split mode {}, choose one of {}'.format(
                value, self.DATASET_SPLIT_MODES
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

    def get_data_and_targets(self, train_dset, test_dset=None):
        train_idx, test_idx = None, None
        if test_dset is None:
            test_dset = train_dset
            n_data = train_dset.num_data
            n_items = self.test_items_count or round(
                self.test_items_fraction * train_dset.num_data)
            mode = self.split_mode
            if mode == 'FROM_START':
                test_idx, train_idx = slice(n_items), slice(n_items, n_data)
            elif mode == 'FROM_END':
                test_idx, train_idx = slice(n_items, n_data), slice(n_items)
            elif mode == 'RANDOM':
                test_idx, next_idx = [], None
                all_idx = set(range(n_data))
                for idx in range(n_items):
                    while next_idx not in all_idx:
                        next_idx = random.randrange(0, n_data)
                    all_idx.remove(next_idx)
                    test_idx.append(next_idx)
                train_idx = list(all_idx)
        return {'train_data': train_dset.get_data_as_arraylike(train_idx),
                'train_targets': train_dset.get_targets(train_idx),
                'test_data': test_dset.get_data_as_arraylike(test_idx),
                'test_targets': test_dset.get_targets(test_idx)}

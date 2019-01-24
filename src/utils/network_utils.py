import os
import importlib
import datetime as dt

import tflearn
import numpy as np

import utils.dataset_utils as ds
import utils.metadata_utils as meta

CLASSIFICATION_FIELDS   = ['item_idx', 'output', 'target', 'shower_prob',
                           'noise_prob']

def _classification_fields_handler(raw_output, target, item_idx,
                                   old_dict=None):
    if old_dict == None:
        old_dict = {}
    rnd_output = np.round(raw_output).astype(np.uint8)
    old_dict['shower_prob'] = round(raw_output[0], 6)
    old_dict['noise_prob']  = round(raw_output[1], 6)
    old_dict['output'] = ('shower' if np.array_equal(rnd_output, [1, 0])
                          else 'noise')
    old_dict['target'] = ('shower' if np.array_equal(target, [1, 0])
                          else 'noise')
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


def train_model(model, train_dataset, run_id, num_epochs=11, eval_dataset=None,
                eval_num=None, eval_fraction=0.1, step=100, metric=True):
    num_data = train_dataset.num_data
    if eval_dataset == None:
        eval_dataset = train_dataset
        eval_size = eval_num or round(num_data*eval_fraction)
        train_slice, eval_slice = slice(eval_size, num_data), slice(eval_size)
    else:
        eval_size = eval_dataset.num_data
        train_slice, eval_slice = slice(num_data), slice(eval_size)
    train_data = reshape_data_for_convnet(
        train_dataset.get_data_as_arraylike(train_slice)
    )
    eval_data = reshape_data_for_convnet(
        eval_dataset.get_data_as_arraylike(eval_slice)
    )
    train_targets = train_dataset.get_targets(train_slice)
    eval_targets = eval_dataset.get_targets(eval_slice)

    model.fit(train_data, train_targets, n_epoch=num_epochs, run_id=run_id,
              validation_set=(eval_data, eval_targets), snapshot_step=step,
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

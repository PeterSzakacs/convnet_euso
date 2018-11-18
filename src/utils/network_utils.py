import os
import importlib
from datetime import datetime as dt

import tflearn
import numpy as np

import utils.dataset_utils as ds


DEFAULT_CHECKING_LOGDIR = '/run/user/{}/convnet_checker'.format(os.getuid())
DEFAULT_TRAINING_LOGDIR = '/run/user/{}/convnet_trainer'.format(os.getuid())


def get_default_run_id(network_module_name):
    current_time = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
    return '{}_{}'.format(network_module_name, current_time)


# dataset functions (reshape data, etc.)


def reshape_data_for_convnet(data, num_channels=1, create_getter=False):
    data_reshaped = []
    for item in data:
        item_shape = item[0].shape
        data_reshaped.append(item.reshape(-1, *item_shape, num_channels))
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


def evaluate_classification_model(model, dataset, items_slice, batch_size=128,
                                  onlyerr=False):
    data = dataset.get_data_as_arraylike(items_slice)
    targets = dataset.get_targets(items_slice)
    num_data = len(data[0])
    data, item_getter = reshape_data_for_convnet(data, create_getter=True)
    log_data, classes_count, hits = [], [0, 0], 0

    miss_handler = lambda log_data, item: log_data.append(item)
    if onlyerr:
        hit_handler = lambda log_data, item: None
    else:
        hit_handler = lambda log_data, item: log_data.append(item)

    for idx in range(0, batch_size, num_data):
        items_slice = slice(idx, idx + batch_size)
        data_batch = item_getter(data, items_slice)
        predictions = model.predict(data_batch)
        for pred_idx in range(len(predictions)):
            prediction = predictions[pred_idx]
            rounded_prediction = np.round(prediction).astype(np.uint8)
            classes_count[0] += rounded_prediction[0]
            classes_count[1] += rounded_prediction[1]
            abs_idx = idx + pred_idx
            if np.array_equal(rounded_prediction, targets[abs_idx]):
                print("correct prediction at item {}".format(abs_idx))
                hit_handler(log_data, (abs_idx, prediction))
                hits += 1
            else:
                print("prediction error at item {}".format(abs_idx))
                miss_handler(log_data, (abs_idx, prediction))
    return log_data, hits, classes_count

def save_model(model, save_pathname):
    model.save(save_pathname);
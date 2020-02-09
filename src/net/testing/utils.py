import numpy as np
import dataset.target_utils as targ
import net.network_utils as netutils


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


def evaluate_classification_model(model, dataset, items_slice=None,
                                  batch_size=128):
    items_slice = items_slice or slice(0, None)
    data = dataset.get_data_as_dict(items_slice)
    targets = dataset.get_targets(items_slice)
    metadata = dataset.get_metadata(items_slice)
    data, item_getter = netutils.convert_dataset_items_to_model_inputs(
        model, data, create_getter=True)

    # TODO: might want to simplify these indexes or at least give better names
    start, stop = items_slice.start, items_slice.stop
    stop = stop or dataset.num_data
    tf_model = model.network_model
    for idx in range(start, stop, batch_size):
        rel_idx = idx - start
        items_slice = slice(rel_idx, rel_idx + batch_size)
        data_batch = item_getter(data, items_slice)
        predictions = tf_model.predict(data_batch)
        for pred_idx in range(len(predictions)):
            prediction = predictions[pred_idx]
            abs_idx = rel_idx + pred_idx
            log_item = _classification_fields_handler(
                prediction, targets[abs_idx], idx + pred_idx,
                old_dict=metadata[abs_idx].copy())
            yield log_item

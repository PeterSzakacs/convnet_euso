import net.constants as net_cons
import net.network_utils as net_utils


class TfModelTrainer:

    DEFAULT_OPTIONAL_SETTINGS = {
        'batch_size': None, 'validation_batch_size': None, 'show_metric': True,
        'snapshot_step': 100,
    }

    def __init__(self, data_dict, num_epochs=11, **optsettings):
        self.train_test_data = data_dict
        self.default_num_epochs = num_epochs
        self._settings = self.DEFAULT_OPTIONAL_SETTINGS.copy()
        self.optional_settings = optsettings

    # properties

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

    # main methods

    def train_model(self, model, data_dict=None, num_epochs=None, run_id=None,
                    **optsettings):
        data = data_dict or self._data_dict
        tr_data, tr_targets = data['train_data'], data['train_targets']
        te_data, te_targets = data['test_data'], data['test_targets']
        epochs = num_epochs or self.default_num_epochs
        settings = self._get_new_settings_dict(**optsettings)
        run_id = run_id or net_utils.get_default_run_id(
            model.network_graph.__module__)

        tf_model = model.network_model
        tf_model.fit(tr_data, tr_targets, n_epoch=epochs, run_id=run_id,
                     validation_set=(te_data, te_targets), **settings)

    # helper and static methods

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

    @staticmethod
    def _validate_setting(key, value):
        if key.endswith('batch_size') or key == 'snapshot_step':
            if value is None:
                return None
            else:
                return int(value)
        elif key == 'show_metric':
            return bool(value)

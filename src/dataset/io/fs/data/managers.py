import dataset.data_utils as data_utils
import dataset.io.fs.facades as data_facades
import dataset.io.fs.managers as managers


class FilesystemDataManager(managers.SingleFilePerItemTypePersistenceManager):

    def __init__(self, io_facades=None, filename_formatters=None):
        custom_handlers = dict(io_facades or {})
        _handlers = data_facades.IO_HANDLERS.copy()
        _handlers.update(custom_handlers)

        super().__init__(_handlers, filename_formatters)

    def _check_and_get_types_subset(self, types_config, types_subset):
        _types_subset = super()._check_and_get_types_subset(
            types_config, types_subset)

        # verify the keys contain only valid item type names
        data_utils.check_item_types(dict.fromkeys(_types_subset, True))
        return {itype: types_config[itype] for itype in _types_subset}

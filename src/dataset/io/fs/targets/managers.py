import dataset.io.fs.facades as targets_facades
import dataset.io.fs.managers as managers


class FilesystemTargetsManager(
    managers.SingleFilePerItemTypePersistenceManager
):

    def __init__(self, io_facades=None, filename_formatters=None):
        custom_handlers = dict(io_facades or {})
        _handlers = targets_facades.IO_HANDLERS.copy()
        _handlers.update(custom_handlers)

        super().__init__(_handlers, filename_formatters)

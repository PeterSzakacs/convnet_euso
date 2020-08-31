import dataset.io.fs.facades as targets_facades
import dataset.io.fs.managers as managers


class FilesystemTargetsManager(
    managers.SingleFilePerItemTypePersistenceManager
):

    def __init__(self, io_facades=None, filename_formatters=None):
        custom_facades = dict(io_facades or {})
        _facades = targets_facades.FACADES.copy()
        _facades.update(custom_facades)

        super().__init__(_facades, filename_formatters)

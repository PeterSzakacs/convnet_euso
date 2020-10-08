import dataset.io.fs.facades as facades
import dataset.io.fs.managers as managers


class FilesystemTargetsManager(
    managers.SingleFilePerItemTypePersistenceManager
):

    def __init__(self, facades_provider=None, filename_formatters=None):
        facades_provider = facades_provider or facades.get_facades_provider()

        super().__init__(facades_provider, filename_formatters)

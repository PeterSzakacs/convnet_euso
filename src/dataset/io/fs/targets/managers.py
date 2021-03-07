import typing as t

import dataset.io.fs.layouts as layouts


class FilesystemTargetsManager:

    def __init__(self, layout_handlers_provider=None):
        self._handlers_provider = (layout_handlers_provider
                                   or layouts.get_layout_handlers_provider())

    def load(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            load_types: t.Iterable[str] = None
    ):
        self._get_handler().load(dataset_name, files_dir, config, load_types)

    def save(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            items: t.Union[t.Mapping[str, t.Sequence], t.Sequence]
    ):
        self._get_handler().save(dataset_name, files_dir, config, items)

    def append(self,
               dataset_name: str,
               files_dir: str,
               config: t.Mapping[str, t.Any],
               items: t.Union[t.Mapping[str, t.Sequence], t.Sequence]
               ):
        self._get_handler().append(dataset_name, files_dir, config, items)

    def delete(
            self,
            dataset_name: str,
            files_dir: str,
            config: t.Mapping[str, t.Any],
            delete_types: t.Iterable[str] = None
    ):
        self._get_handler().delete(
            dataset_name, files_dir, config, delete_types
        )

    def _get_handler(
            self
    ) -> layouts.DatasetSectionFileLayoutHandler:
        return self._handlers_provider.get_instance(
            'single_file_per_item_type'
        )

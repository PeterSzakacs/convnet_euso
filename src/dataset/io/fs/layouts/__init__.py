import common.providers as providers

from .base import DatasetSectionFileLayoutHandler
from .handlers import SingleFilePerItemTypeLayoutHandler


def get_layout_handlers_provider() -> providers.ClassInstanceProvider:
    _handlers_map = {
        'single_file_per_item_type': {
            'class': SingleFilePerItemTypeLayoutHandler,
        },
    }
    return providers.ClassInstanceProvider(
        class_map=_handlers_map, base_class=DatasetSectionFileLayoutHandler
    )

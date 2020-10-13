import common.providers as providers
from .filename_utils import FilenameFormatter
from .filename_utils import NameWithSuffixFormatter
from .filename_utils import NameWithTypeSuffixFormatter
from .filename_utils import TypeOnlyFormatter
from .filename_utils import append_file_extension
from .filename_utils import create_full_path


def get_formatters_provider() -> providers.ClassInstanceProvider:
    _formatters = {
        'type_only': {'class': TypeOnlyFormatter},
        'name_with_type_suffix': {'class': NameWithTypeSuffixFormatter},
        'name_with_suffix': {'class': NameWithSuffixFormatter},
    }
    return providers.ClassInstanceProvider(
        class_map=_formatters, base_class=FilenameFormatter
    )

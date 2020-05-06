from .filename_utils import TypeOnlyFormatter
from .filename_utils import NameWithTypeSuffixFormatter
from .filename_utils import create_full_path
from .filename_utils import append_file_extension

FILENAME_FORMATTERS = {
    'type_only': TypeOnlyFormatter(),
    'name_with_type_suffix': NameWithTypeSuffixFormatter(),
}

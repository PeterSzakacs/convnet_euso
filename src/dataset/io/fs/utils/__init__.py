from .filename_utils import FilenameFormatter
from .filename_utils import NameWithSuffixFormatter
from .filename_utils import NameWithTypeSuffixFormatter
from .filename_utils import TypeOnlyFormatter
from .filename_utils import append_file_extension
from .filename_utils import create_full_path

FILENAME_FORMATTERS = {
    'type_only': TypeOnlyFormatter(),
    'name_with_type_suffix': NameWithTypeSuffixFormatter(),
    'name_with_suffix': NameWithSuffixFormatter(),
}

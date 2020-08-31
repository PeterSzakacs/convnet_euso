import ast

import dataset.constants as cons
import dataset.data_utils as shape_utils
import dataset.io.fs.config.ini.base as ini


class ConfigParser(ini.AbstractIniConfigParser):

    @property
    def version(self):
        return 0

    def parse_config(self, config):
        _conf = config.copy()
        general_sec = _conf['general']
        version = general_sec.get('version', -1)
        if version == -1:
            return self._parse_unversioned(_conf)
        elif version == '0':
            return self._parse_versioned(_conf)
        else:
            raise ValueError(f"Unsupported config version: {version}")

    def create_config(self, dataset_attrs):
        dataset_attrs = dataset_attrs.copy()
        _data = dataset_attrs['data']
        _targets = dataset_attrs['targets']
        _metadata = dataset_attrs['metadata']
        _d_types = _data['types']
        _t_types = _targets['types']
        (n_f, f_h, f_w) = _data['packet_shape']
        return {
            'general': {
                'version': 0,
                'num_items': dataset_attrs['num_items'],
            },
            'data': {
                'types': set(_d_types),
            },
            'data:packet_shape': {
                'num_frames': n_f,
                'frame_height': f_h,
                'frame_width': f_w,
            },
            'data:backend': _data['backend'],
            **{f'data:{itype}': self._format_item_type_config(conf)
               for itype, conf in _d_types.items()},
            'targets': {
                'types': set(_t_types),
            },
            'targets:backend': _targets['backend'],
            **{f'targets:{itype}': self._format_item_type_config(conf)
               for itype, conf in _t_types.items()},
            'metadata': {
                'fields': _metadata['fields'],
            },
            'metadata:backend': _metadata['backend'],
        }

    # helper methods

    @staticmethod
    def _parse_unversioned(config):
        general_sec = config['general']
        packet_shape_sec = config['packet_shape']
        n_f = int(packet_shape_sec['num_frames'])
        f_h = int(packet_shape_sec['frame_height'])
        f_w = int(packet_shape_sec['frame_width'])
        item_types_sec = config['item_types']
        item_types = set(k for k in cons.ALL_ITEM_TYPES
                         if item_types_sec[k] == 'True')

        dtype = general_sec['dtype']
        packet_shape = (n_f, f_h, f_w)
        shapes = shape_utils.get_data_item_shapes(
            packet_shape, item_types
        )

        return {
            'version': 0,
            'num_items': int(general_sec['num_data']),
            'data': {
                'packet_shape': packet_shape,
                'types': {itype: {'dtype': dtype, 'shape': shapes[itype]}
                          for itype in item_types},
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_type_suffix',
                }
            },
            'targets': {
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8', 'shape': (2, ),
                    }
                },
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_suffix',
                    'suffix': 'class_targets',
                },
            },
            'metadata': {
                'fields': ast.literal_eval(general_sec['metafields']),
                'backend': {
                    'name': 'tsv',
                    'filename_extension': 'tsv',
                    'filename_format': 'name_with_suffix',
                    'suffix': 'meta',
                },
            }
        }

    @staticmethod
    def _parse_versioned(config):
        general_sec = config['general']

        data_sec = config['data']
        packet_shape_sec = config['data:packet_shape']
        data_backend_sec = config['data:backend']

        n_f = int(packet_shape_sec['num_frames'])
        f_h = int(packet_shape_sec['frame_height'])
        f_w = int(packet_shape_sec['frame_width'])

        targets_sec = config['targets']
        targets_backend_sec = config['targets:backend']

        metadata_sec = config['metadata']
        metadata_backend_sec = config['metadata:backend'] or {}

        d_types = ast.literal_eval(data_sec['types'])
        t_types = ast.literal_eval(targets_sec['types'])

        backend_defaults = config.get('general:backend', {})
        if backend_defaults:
            data_backend_sec = {**backend_defaults, **data_backend_sec}
            targets_backend_sec = {**backend_defaults, **targets_backend_sec}
            metadata_backend_sec = {**backend_defaults, **metadata_backend_sec}
        return {
            'version': 0,
            'num_items': int(general_sec['num_items']),
            'data': {
                'packet_shape': (n_f, f_h, f_w),
                'types': {
                    itype: ConfigParser._extract_item_type_config(
                        config[f'data:{itype}']
                    ) for itype in d_types
                },
                'backend': data_backend_sec.copy(),
            },
            'targets': {
                'types': {
                    itype: ConfigParser._extract_item_type_config(
                        config[f'targets:{itype}']
                    ) for itype in t_types
                },
                'backend': targets_backend_sec.copy(),
            },
            'metadata': {
                'fields': ast.literal_eval(metadata_sec['fields']),
                'backend': metadata_backend_sec.copy(),
            }
        }

    @staticmethod
    def _format_item_type_config(type_config):
        item_shape = type_config['shape']
        shape_size = len(item_shape)
        return {
            'dtype': type_config['dtype'],
            'shape_size': shape_size,
            **{f'shape[{idx}]': item_shape[idx] for idx in range(shape_size)},
        }

    @staticmethod
    def _extract_item_type_config(type_config):
        shape_size = int(type_config['shape_size'])
        shape_values = tuple(int(type_config[f'shape[{idx}]'])
                             for idx in range(shape_size))
        return {
            'dtype': type_config['dtype'],
            'shape': shape_values,
        }

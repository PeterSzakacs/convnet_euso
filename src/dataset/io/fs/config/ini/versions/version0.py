import ast

import dataset.constants as cons
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
            **{f'data:{itype}': {'dtype': conf['dtype']}
               for itype, conf in _d_types.items()},
            'targets': {
                'types': set(_t_types),
            },
            'targets:backend': _targets['backend'],
            **{f'targets:{itype}': {'dtype': conf['dtype']}
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

        return {
            'version': 0,
            'num_items': int(general_sec['num_data']),
            'data': {
                'packet_shape': (n_f, f_h, f_w),
                'types': {itype: {'dtype': dtype} for itype in item_types},
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_type_suffix',
                }
            },
            'targets': {
                'types': {'softmax_class_value': {'dtype': 'uint8'}},
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_type_suffix',
                },
            },
            'metadata': {
                'fields': ast.literal_eval(general_sec['metafields']),
                'backend': {
                    'name': 'tsv',
                    'filename_extension': 'tsv',
                    'filename_format': 'name_with_type_suffix',
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
                    itype: {'dtype': config[f'data:{itype}']['dtype']}
                    for itype in d_types
                },
                'backend': data_backend_sec.copy(),
            },
            'targets': {
                'types': {
                    itype: {'dtype': config[f'targets:{itype}']['dtype']}
                    for itype in t_types
                },
                'backend': targets_backend_sec.copy(),
            },
            'metadata': {
                'fields': ast.literal_eval(metadata_sec['fields']),
                'backend': metadata_backend_sec.copy(),
            }
        }

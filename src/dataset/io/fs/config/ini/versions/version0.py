import ast

import dataset.constants as cons
import dataset.io.fs.base as fs_io_base


class ConfigParser(fs_io_base.FsPersistencyHandler):

    @property
    def version(self):
        return 0

    def parse_config(self, config):
        general_sec = config['general']
        version = general_sec.get('version', -1)
        if version == -1:
            return self._parse_unversioned(config)
        elif version == '0':
            return self._parse_versioned(config)
        else:
            raise ValueError(f"Unsupported config version: {version}")

    def create_config(self, dataset_attrs):
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
                'backend': _data['backend'],
            },
            'data:packet_shape': {
                'num_frames': n_f,
                'frame_height': f_h,
                'frame_width': f_w,
            },
            **{f'data:{itype}': {'dtype': conf['dtype']}
               for itype, conf in _d_types.items()},
            'targets': {
                'types': set(_t_types),
                'backend': _targets['backend'],
            },
            **{f'targets:{itype}': {'dtype': conf['dtype']}
               for itype, conf in _t_types.items()},
            'metadata': {
                'backend': _metadata['backend'],
                'fields': _metadata['fields'],
            }
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
                'backend': 'npy',
            },
            'targets': {
                'types': {'softmax_class_value': {'dtype': 'uint8'}},
                'backend': 'npy',
            },
            'metadata': {
                'fields': ast.literal_eval(general_sec['metafields']),
                'backend': 'tsv',
            }
        }

    @staticmethod
    def _parse_versioned(config):
        general_sec = config['general']
        data_sec = config['data']
        packet_shape_sec = config['data:packet_shape']
        n_f = int(packet_shape_sec['num_frames'])
        f_h = int(packet_shape_sec['frame_height'])
        f_w = int(packet_shape_sec['frame_width'])
        targets_sec = config['targets']
        metadata_sec = config['metadata']

        d_types = ast.literal_eval(data_sec['types'])
        t_types = ast.literal_eval(targets_sec['types'])

        return {
            'version': 0,
            'num_items': int(general_sec['num_items']),
            'data': {
                'packet_shape': (n_f, f_h, f_w),
                'types': {
                    itype: {'dtype': config[f'data:{itype}']['dtype']}
                    for itype in d_types
                },
                'backend': data_sec['backend'],
            },
            'targets': {
                'types': {
                    itype: {'dtype': config[f'targets:{itype}']['dtype']}
                    for itype in t_types
                },
                'backend': targets_sec['backend'],
            },
            'metadata': {
                'fields': ast.literal_eval(metadata_sec['fields']),
                'backend': metadata_sec['backend'],
            }
        }

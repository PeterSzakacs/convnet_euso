import unittest

import dataset.io.fs.config.ini.versions.version0 as v0


class TestConfigParser(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.maxDiff = None

    def test_parse_unversioned_config(self):
        raw_config = {
            'general': {
                'num_data': '100', 'dtype': 'uint8',
                'metafields': "{'bg_lambda', 'length', 'yx_angle'}",
            },
            'packet_shape': {
                'num_frames': '20', 'frame_height': '24', 'frame_width': '48'
            },
            'item_types': {
                'yx': 'True', 'raw': 'True',
                'gtux': 'False', 'gtuy': 'False'
            },
        }
        exp_config = {
            'version': 0,
            'num_items': 100,
            'data': {
                'packet_shape': (20, 24, 48),
                'backend': 'npy',
                'types': {
                    'raw': {
                        'dtype': 'uint8'
                    },
                    'yx': {
                        'dtype': 'uint8'
                    }
                },
            },
            'targets': {
                'backend': 'npy',
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8'
                    }
                },
            },
            'metadata': {
                'fields': {'bg_lambda', 'length', 'yx_angle'},
                'backend': 'tsv',
            }
        }
        parser = v0.ConfigParser()
        config = parser.parse_config(raw_config)
        self.assertDictEqual(exp_config, config)

    def test_parse_versioned_config(self):
        raw_config = {
            'general': {
                'version': '0', 'num_items': '200',
            },
            'data': {
                'types': "{'yx', 'gtux', 'gtuy'}", 'backend': 'memmap',
            },
            'data:packet_shape': {
                'num_frames': '30', 'frame_height': '48', 'frame_width': '24',
            },
            'data:yx': {
                'dtype': 'uint8'
            },
            'data:gtux': {
                'dtype': 'uint8'
            },
            'data:gtuy': {
                'dtype': 'uint8'
            },
            'targets': {
                'types': "{'softmax_class_value'}", 'backend': 'npy',
            },
            'targets:softmax_class_value': {
                'dtype': 'uint8',
            },
            'metadata': {
                'fields': "{'length', 'yx_angle'}", 'backend': 'tsv',
            }
        }
        exp_config = {
            'version': 0,
            'num_items': 200,
            'data': {
                'packet_shape': (30, 48, 24),
                'backend': 'memmap',
                'types': {
                    'yx': {
                        'dtype': 'uint8'
                    },
                    'gtux': {
                        'dtype': 'uint8'
                    },
                    'gtuy': {
                        'dtype': 'uint8'
                    },
                },
            },
            'targets': {
                'backend': 'npy',
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8'
                    }
                },
            },
            'metadata': {
                'fields': {'length', 'yx_angle'},
                'backend': 'tsv',
            }
        }
        parser = v0.ConfigParser()
        config = parser.parse_config(raw_config)
        self.assertDictEqual(exp_config, config)

    def test_parse_unsupported_version(self):
        raw_config = {
            'general': {
                'version': '2',
            },
        }
        parser = v0.ConfigParser()
        self.assertRaises(ValueError, parser.parse_config, raw_config)

    def test_save_config(self):
        dataset_attrs = {
            'num_items': 200,
            'data': {
                'types': {
                    'yx': {
                        'dtype': 'uint8',
                    },
                    'gtux': {
                        'dtype': 'uint8',
                    },
                    'gtuy': {
                        'dtype': 'uint8',
                    },
                },
                'packet_shape': (30, 48, 24),
                'backend': 'memmap',
            },
            'targets': {
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8',
                    }
                },
                'backend': 'npy',
            },
            'metadata': {
                'fields': {'bg_lambda', 'yx_angle'},
                'backend': 'tsv',
            }
        }
        exp_config = {
            'general': {
                'version': 0, 'num_items': 200,
            },
            'data': {
                'types': {'gtux', 'yx', 'gtuy'}, 'backend': 'memmap',
            },
            'data:packet_shape': {
                'num_frames': 30, 'frame_height': 48, 'frame_width': 24,
            },
            'data:yx': {
                'dtype': 'uint8',
            },
            'data:gtux': {
                'dtype': 'uint8',
            },
            'data:gtuy': {
                'dtype': 'uint8',
            },
            'targets': {
                'types': {'softmax_class_value'}, 'backend': 'npy',
            },
            'targets:softmax_class_value': {
                'dtype': 'uint8',
            },
            'metadata': {
                'fields': {'bg_lambda', 'yx_angle'}, 'backend': 'tsv',
            }
        }
        parser = v0.ConfigParser()
        config = parser.create_config(dataset_attrs)
        self.assertDictEqual(exp_config, config)


if __name__ == '__main__':
    unittest.main()

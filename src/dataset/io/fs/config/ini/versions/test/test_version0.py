import unittest

import dataset.io.fs.config.ini.versions.version0 as v0


class TestLegacyConfigParsing(unittest.TestCase):

    def shortDescription(self):
        return """
        Test parsing of legacy configs without a version attribute 
        (other attributes are implicitly set to defaults)
        """

    @classmethod
    def setUp(cls):
        cls.maxDiff = None
        cls.parser = v0.ConfigParser()
        cls.raw_config = {
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
        cls.exp_config = {
            'version': 0,
            'num_items': 100,
            'data': {
                'packet_shape': (20, 24, 48),
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_type_suffix',
                },
                'types': {
                    'raw': {
                        'dtype': 'uint8',
                        'shape': (20, 24, 48),
                    },
                    'yx': {
                        'dtype': 'uint8',
                        'shape': (24, 48),
                    }
                },
            },
            'targets': {
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_suffix',
                    'suffix': 'class_targets'
                },
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8',
                        'shape': (2, ),
                    }
                },
            },
            'metadata': {
                'backend': {
                    'name': 'tsv',
                    'filename_extension': 'tsv',
                    'filename_format': 'name_with_suffix',
                    'suffix': 'meta',
                },
                'fields': {'bg_lambda', 'length', 'yx_angle'},
            }
        }

    def test_parse_unversioned_config_version(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('version', config)
        self.assertEqual(config['version'], 0)

    def test_parse_version0_config_num_items(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('num_items', config)
        self.assertEqual(config['num_items'], self.exp_config['num_items'])

    def test_parse_unversioned_config_data(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('data', config)
        self.assertDictEqual(config['data'], self.exp_config['data'])

    def test_parse_unversioned_config_targets(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('targets', config)
        self.assertDictEqual(config['targets'], self.exp_config['targets'])

    def test_parse_unversioned_config_metadata(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('metadata', config)
        self.assertDictEqual(config['metadata'], self.exp_config['metadata'])


class TestVersion0ConfigParsing(unittest.TestCase):

    def shortDescription(self):
        return """
        Test parsing of configs with a version attribute set to 0
        """

    @classmethod
    def setUp(cls):
        cls.maxDiff = None
        cls.parser = v0.ConfigParser()
        cls.raw_config = {
            'general': {
                'version': '0', 'num_items': '200',
            },
            'general:backend': {
                'filename_format': 'name_with_type_suffix',
            },
            'data': {
                'types': "{'yx', 'gtux', 'gtuy'}",
            },
            'data:packet_shape': {
                'num_frames': '30', 'frame_height': '48', 'frame_width': '24',
            },
            'data:backend': {
                'name': 'memmap',
                'filename_extension': 'memmap',
                'filename_format': 'type_only',
            },
            'data:yx': {
                'dtype': 'uint8',
                'shape_size': '2',
                'shape[0]': '48',
                'shape[1]': '24',
            },
            'data:gtux': {
                'dtype': 'uint8',
                'shape_size': '2',
                'shape[0]': '30',
                'shape[1]': '24',
            },
            'data:gtuy': {
                'dtype': 'uint8',
                'shape_size': '2',
                'shape[0]': '30',
                'shape[1]': '48',
            },
            'targets': {
                'types': "{'softmax_class_value'}",
            },
            'targets:backend': {
                'name': 'npy',
                'filename_extension': 'npy',
            },
            'targets:softmax_class_value': {
                'dtype': 'uint8',
                'shape_size': '1',
                'shape[0]': '2',
            },
            'metadata': {
                'fields': "{'length', 'yx_angle'}",
            },
            'metadata:backend': {
                'name': 'tsv',
                'filename_extension': 'tsv',
            },
        }
        cls.exp_config = {
            'version': 0,
            'num_items': 200,
            'data': {
                'packet_shape': (30, 48, 24),
                'backend': {
                    'name': 'memmap',
                    'filename_extension': 'memmap',
                    'filename_format': 'type_only',
                },
                'types': {
                    'yx': {
                        'dtype': 'uint8',
                        'shape': (48, 24),
                    },
                    'gtux': {
                        'dtype': 'uint8',
                        'shape': (30, 24),
                    },
                    'gtuy': {
                        'dtype': 'uint8',
                        'shape': (30, 48),
                    },
                },
            },
            'targets': {
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'npy',
                    'filename_format': 'name_with_type_suffix',
                },
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8',
                        'shape': (2, ),
                    }
                },
            },
            'metadata': {
                'backend': {
                    'name': 'tsv',
                    'filename_extension': 'tsv',
                    'filename_format': 'name_with_type_suffix',
                },
                'fields': {'length', 'yx_angle'},
            }
        }

    def test_parse_version0_config_version(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('version', config)
        self.assertEqual(config['version'], 0)

    def test_parse_version0_config_num_items(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('num_items', config)
        self.assertEqual(config['num_items'], self.exp_config['num_items'])

    def test_parse_version0_config_data(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('data', config)
        self.assertDictEqual(config['data'], self.exp_config['data'])

    def test_parse_version0_config_targets(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('targets', config)
        self.assertDictEqual(config['targets'], self.exp_config['targets'])

    def test_parse_version0_config_metadata(self):
        config = self.parser.parse_config(self.raw_config)
        self.assertIn('metadata', config)
        self.assertDictEqual(config['metadata'], self.exp_config['metadata'])

    def test_parse_unsupported_version(self):
        raw_config = {
            'general': {
                'version': '2',
            },
        }
        parser = v0.ConfigParser()
        self.assertRaises(ValueError, parser.parse_config, raw_config)


class TestVersion0ConfigCreation(unittest.TestCase):

    def shortDescription(self):
        return """
        Test creation of config (version 0) dict for saving to FS
        """

    @classmethod
    def setUp(cls):
        cls.maxDiff = None
        cls.parser = v0.ConfigParser()
        cls.dataset_attrs = {
            'num_items': 200,
            'data': {
                'types': {
                    'yx': {
                        'dtype': 'uint8',
                        'shape': (48, 24),
                    },
                    'gtux': {
                        'dtype': 'uint8',
                        'shape': (30, 24),
                    },
                    'gtuy': {
                        'dtype': 'uint8',
                        'shape': (30, 48),
                    },
                },
                'packet_shape': (30, 48, 24),
                'backend': {
                    'name': 'memmap',
                    'filename_extension': 'memmap',
                    'filename_format': 'type_only',
                },
            },
            'targets': {
                'types': {
                    'softmax_class_value': {
                        'dtype': 'uint8',
                        'shape': (2, ),
                    }
                },
                'backend': {
                    'name': 'npy',
                    'filename_extension': 'test',
                    'filename_format': 'type_only',
                },
            },
            'metadata': {
                'fields': {'bg_lambda', 'yx_angle'},
                'backend': {
                    'name': 'tsv',
                    'filename_extension': 'tsv',
                    'filename_format': 'name_with_type_suffix',
                },
            }
        }
        cls.exp_config = {
            'general': {
                'version': 0, 'num_items': 200,
            },
            'data': {
                'types': {'gtux', 'yx', 'gtuy'},
            },
            'data:backend': {
                'name': 'memmap',
                'filename_extension': 'memmap',
                'filename_format': 'type_only',
            },
            'data:packet_shape': {
                'num_frames': 30, 'frame_height': 48, 'frame_width': 24,
            },
            'data:yx': {
                'dtype': 'uint8',
                'shape_size': 2,
                'shape[0]': 48,
                'shape[1]': 24,
            },
            'data:gtux': {
                'dtype': 'uint8',
                'shape_size': 2,
                'shape[0]': 30,
                'shape[1]': 24,
            },
            'data:gtuy': {
                'dtype': 'uint8',
                'shape_size': 2,
                'shape[0]': 30,
                'shape[1]': 48,
            },
            'targets': {
                'types': {'softmax_class_value'},
            },
            'targets:backend': {
                'name': 'npy',
                'filename_extension': 'test',
                'filename_format': 'type_only',
            },
            'targets:softmax_class_value': {
                'dtype': 'uint8',
                'shape_size': 1,
                'shape[0]': 2,
            },
            'metadata': {
                'fields': {'bg_lambda', 'yx_angle'},
            },
            'metadata:backend': {
                'name': 'tsv',
                'filename_extension': 'tsv',
                'filename_format': 'name_with_type_suffix',
            }
        }

    def test_create_configv0_general_sec(self):
        config = self.parser.create_config(self.dataset_attrs)
        self.assertIn('general', config)
        self.assertDictEqual(self.exp_config['general'], config['general'])

    def test_create_configv0_data_sec(self):
        input_attrs = self.dataset_attrs
        conf = self.parser.create_config(input_attrs)
        exp_conf = self.exp_config
        self.assertIn('data', conf)
        self.assertDictEqual(exp_conf['data'], conf['data'])
        self.assertIn('data:backend', conf)
        self.assertDictEqual(exp_conf['data:backend'], conf['data:backend'])
        for item_type in input_attrs['data']['types']:
            section_name = f'data:{item_type}'
            self.assertIn(section_name, conf)
            self.assertDictEqual(exp_conf[section_name], conf[section_name])

    def test_create_configv0_targets_sec(self):
        input_attrs = self.dataset_attrs
        conf = self.parser.create_config(input_attrs)
        exp_conf = self.exp_config
        self.assertIn('targets', conf)
        self.assertDictEqual(exp_conf['targets'], conf['targets'])
        self.assertIn('targets:backend', conf)
        self.assertDictEqual(
            exp_conf['targets:backend'], conf['targets:backend'])
        for item_type in input_attrs['targets']['types']:
            section_name = f'targets:{item_type}'
            self.assertIn(section_name, conf)
            self.assertDictEqual(exp_conf[section_name], conf[section_name])

    def test_create_configv0_metadata_sec(self):
        input_attrs = self.dataset_attrs
        conf = self.parser.create_config(input_attrs)
        exp_conf = self.exp_config
        self.assertIn('metadata', conf)
        self.assertDictEqual(exp_conf['metadata'], conf['metadata'])
        self.assertIn('metadata:backend', conf)
        self.assertDictEqual(
            exp_conf['metadata:backend'], conf['metadata:backend'])


if __name__ == '__main__':
    unittest.main()

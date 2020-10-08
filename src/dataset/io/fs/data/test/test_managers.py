import random
import unittest
import unittest.mock as mock
import uuid

import numpy as np

import common.providers as providers
import dataset.constants as constants
import dataset.data_utils as data_utils
import dataset.io.fs.data.managers as managers
import dataset.io.fs.facades as fs_facades


class TestLoadDataFromFilesystem(unittest.TestCase):

    @mock.patch('os.path.isdir', return_value=True)
    def test_load_all_data(self, m_isdir):
        # prepare input config
        config = _get_config()
        config['backend']['filename_format'] = 'type_only'

        item_types = config['types']
        num_items = config['num_items']
        exp_backend = config['backend']['name']
        exp_extension = config['backend']['filename_extension']

        # setup facades
        mock_facades = _get_mock_facades()
        exp_method = mock_facades[exp_backend].load
        provider = _get_mocks_provider(mock_facades)

        # call load
        manager = managers.FilesystemDataManager(facades_provider=provider)
        dset_data = manager.load('testset', '/res', config)

        # check returned data and the correct call args to IO facades
        self.assertSetEqual(set(dset_data.keys()), set(item_types.keys()))
        self.assertEqual(len(item_types), exp_method.call_count)
        for item_type, values in item_types.items():
            exp_filename = f'/res/{item_type}.{exp_extension}'
            exp_method.assert_any_call(exp_filename, dtype=values['dtype'],
                                       item_shape=values['shape'],
                                       num_items=num_items)

    @mock.patch('os.path.isdir', return_value=True)
    def test_load_data_subset(self, m_isdir):
        # prepare input config
        config = _get_config()
        config['backend']['filename_format'] = 'name_with_type_suffix'

        packet_shape = config['packet_shape']
        item_types = _get_types_config(packet_shape,
                                       item_types=constants.ALL_ITEM_TYPES)
        config['types'] = item_types
        num_items = config['num_items']
        exp_backend = config['backend']['name']
        exp_extension = config['backend']['filename_extension']

        load_types = ['yx']

        # setup facades
        mock_facades = _get_mock_facades()
        exp_method = mock_facades[exp_backend].load
        provider = _get_mocks_provider(mock_facades)

        # call load
        manager = managers.FilesystemDataManager(facades_provider=provider)
        dset_data = manager.load('test', '/data', config,
                                 load_types=load_types)

        # check returned data and the correct call args to IO facades
        self.assertSetEqual(set(dset_data.keys()), set(load_types))
        self.assertEqual(len(load_types), exp_method.call_count)
        for item_type in load_types:
            exp_filename = f'/data/test_{item_type}.{exp_extension}'
            config = item_types[item_type]
            exp_method.assert_any_call(exp_filename, dtype=config['dtype'],
                                       item_shape=config['shape'],
                                       num_items=num_items)


class TestSaveDataToFilesystem(unittest.TestCase):

    @mock.patch('os.path.isdir', return_value=True)
    def test_save_data(self, m_isdir):
        # prepare input config
        config = _get_config()
        config['backend']['filename_format'] = 'type_only'

        packet_shape = config['packet_shape']
        item_types = config['types']
        num_items = config['num_items']
        exp_backend = config['backend']['name']
        exp_extension = config['backend']['filename_extension']

        # prepare input data arrays
        items = data_utils.convert_packet(
            np.empty(shape=packet_shape, dtype='uint32'), item_types.keys()
        )
        for item_type in item_types:
            item = items[item_type]
            item = np.reshape(item, (1, *item.shape))
            items[item_type] = np.repeat(item, num_items, axis=0)

        # setup facades
        mock_facades = _get_mock_facades()
        exp_method = mock_facades[exp_backend].save
        provider = _get_mocks_provider(mock_facades)

        # call save
        manager = managers.FilesystemDataManager(facades_provider=provider)
        filenames = manager.save('dataset', '/set', config, items)

        # verify expected calls
        self.assertSetEqual(set(filenames.keys()), set(item_types.keys()))
        self.assertEqual(len(item_types), exp_method.call_count)
        for item_type, values in item_types.items():
            exp_filename = f'/set/{item_type}.{exp_extension}'
            exp_method.assert_any_call(exp_filename, items[item_type])


class TestAppendDataToFilesystem(unittest.TestCase):

    @mock.patch('os.path.isdir', return_value=True)
    def test_append_data(self, m_isdir):
        # prepare input config
        config = _get_config()
        config['backend']['filename_format'] = 'type_only'

        packet_shape = config['packet_shape']
        item_types = config['types']
        num_items = config['num_items']
        exp_backend = config['backend']['name']
        exp_extension = config['backend']['filename_extension']

        # prepare input data arrays
        items = data_utils.convert_packet(
            np.empty(shape=packet_shape, dtype='uint32'), item_types.keys()
        )
        for item_type in item_types:
            item = items[item_type]
            item = np.reshape(item, (1, *item.shape))
            items[item_type] = np.repeat(item, num_items, axis=0)

        # setup facades
        mock_facades = _get_mock_facades()
        exp_method = mock_facades[exp_backend].append
        provider = _get_mocks_provider(mock_facades)

        # call append
        manager = managers.FilesystemDataManager(facades_provider=provider)
        filenames = manager.append('dataset', '/set', config, items)

        # verify expected calls
        self.assertSetEqual(set(filenames.keys()), set(item_types.keys()))
        self.assertEqual(len(item_types), exp_method.call_count)
        for item_type, values in item_types.items():
            exp_filename = f'/set/{item_type}.{exp_extension}'
            exp_method.assert_any_call(exp_filename, items[item_type],
                                       dtype=values['dtype'],
                                       item_shape=values['shape'],
                                       num_items=num_items)


class TestDeleteDataFromFilesystem(unittest.TestCase):

    @mock.patch('os.path.isdir', return_value=True)
    def test_delete_all_data(self, m_isdir):
        # prepare input config
        config = _get_config()
        config['backend']['filename_format'] = 'name_with_type_suffix'

        item_types = config['types']
        exp_backend = config['backend']['name']
        exp_extension = config['backend']['filename_extension']

        # setup facades
        mock_facades = _get_mock_facades()
        exp_method = mock_facades[exp_backend].delete
        provider = _get_mocks_provider(mock_facades)

        # call delete
        manager = managers.FilesystemDataManager(facades_provider=provider)
        filenames = manager.delete('dataset', '/set', config)

        # verify expected calls
        self.assertSetEqual(set(filenames.keys()), set(item_types.keys()))
        self.assertEqual(len(item_types), exp_method.call_count)
        for item_type, values in item_types.items():
            exp_filename = f'/set/dataset_{item_type}.{exp_extension}'
            exp_method.assert_any_call(exp_filename)

    @mock.patch('os.path.isdir', return_value=True)
    def test_delete_data_subset(self, m_isdir):
        # prepare input config
        config = _get_config()
        config['backend']['filename_format'] = 'name_with_type_suffix'

        item_types = config['types']
        exp_backend = config['backend']['name']
        exp_extension = config['backend']['filename_extension']

        # setup facades
        mock_facades = _get_mock_facades()
        exp_method = mock_facades[exp_backend].delete
        provider = _get_mocks_provider(mock_facades)

        # call delete
        manager = managers.FilesystemDataManager(facades_provider=provider)
        filenames = manager.delete('dataset', '/set', config)

        # verify expected calls
        self.assertSetEqual(set(filenames.keys()), set(item_types.keys()))
        self.assertEqual(len(item_types), exp_method.call_count)
        for item_type, values in item_types.items():
            exp_filename = f'/set/dataset_{item_type}.{exp_extension}'
            exp_method.assert_any_call(exp_filename)


def _get_mock_facades(return_value=np.ones(shape=(1, 10, 2))):
    _resolver = fs_facades.get_facades_provider()
    _available = _resolver.available_keys
    _mocks = {key: mock.create_autospec(_resolver.get_instance(key))
              for key in _available}
    for key, facade in _mocks.items():
        facade.load.return_value = return_value
    return _mocks


def _get_mocks_provider(mock_facades):
    _provider = providers.ClassInstanceProvider()
    for key, facade in mock_facades.items():
        _provider.set_class(key, type(facade), cached_instance=facade)
    return _provider


def _get_config():
    randint = random.randint
    packet_shape = (randint(1, 128), randint(1, 48), randint(1, 48))
    return {
        'num_items': randint(1, 1000),
        'packet_shape': packet_shape,
        'backend': _get_backend_config(),
        'types': _get_types_config(packet_shape),
    }


def _get_backend_config():
    choice = random.choice

    backend_names = ['npy', 'memmap']
    filename_formats = ['types_only', 'name_with_type_suffix']
    filename_extensions = backend_names.copy()
    filename_extensions.extend(str(uuid.uuid4()) for idx in range(3))

    return {
        'name': choice(backend_names),
        'filename_extension': choice(filename_extensions),
        'filename_format': choice(filename_formats),
    }


def _get_types_config(packet_shape, item_types=None):
    randint = random.randint
    choice = random.choice
    samples = random.sample

    sctypes = np.sctypes
    all_dtypes = sctypes['int'] + sctypes['uint'] + sctypes['float']
    all_itypes = constants.ALL_ITEM_TYPES

    if item_types is None:
        itypes = samples(all_itypes, randint(1, len(all_itypes)))
    else:
        itypes = item_types
    shapes = data_utils.get_data_item_shapes(packet_shape, itypes)
    return {itype: {'dtype': choice(all_dtypes),
                    'shape': shapes[itype]}
            for itype in itypes}


if __name__ == '__main__':
    unittest.main()

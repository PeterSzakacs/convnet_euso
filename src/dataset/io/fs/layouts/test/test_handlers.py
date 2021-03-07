import abc
import os
import unittest
import unittest.mock as mock

import dataset.io.fs.config.test.mixins as config_test_mixins
import dataset.io.fs.facades.test.mixins as facade_test_mixins
from .. import base
from .. import handlers


class DatasetSectionFileLayoutHandlerTest(
    abc.ABC,
    facade_test_mixins.IoFacadeMocksTestMixin,
    config_test_mixins.DatasetConfigRandomGeneratorTestMixin,
):

    # misc. (helper methods)

    @abc.abstractmethod
    def _create_handler(
            self,
            io_facades_provider=None,
            formatters_provider=None,
    ) -> base.DatasetSectionFileLayoutHandler:
        pass

    def _create_config(self, types, filename_formats):
        return self._generate_section_config(
            types=self._generate_types_config(item_types=types),
            backend=self._generate_backend_config(
                filename_formats=filename_formats
            )
        )

    def _get_exp_filenames(self, dataset_name, files_dir, config,
                           types_subset=None):
        if not types_subset:
            types_subset = config['types']
        filename_format = config['backend']['filename_format']
        exp_extension = config['backend']['filename_extension']
        if filename_format == 'type_only':
            return {
                item_type: os.path.join(
                    files_dir, f'{item_type}.{exp_extension}')
                for item_type in types_subset
            }
        elif filename_format == 'name_with_type_suffix':
            return {
                item_type: os.path.join(
                    files_dir, f'{dataset_name}_{item_type}.{exp_extension}')
                for item_type in types_subset
            }
        else:
            return {}


@mock.patch('os.path.isdir', return_value=True)
class TestSingleFilePerItemTypeLayoutHandler(
    unittest.TestCase,
    DatasetSectionFileLayoutHandlerTest,
):

    def _create_handler(
            self,
            io_facades_provider=None,
            formatters_provider=None,
    ) -> base.DatasetSectionFileLayoutHandler:
        return handlers.SingleFilePerItemTypeLayoutHandler(
            io_facades_provider=io_facades_provider,
            formatters_provider=formatters_provider,
        )

    def test_load_all_and_check_returned_items_are_indexed_by_type(
            self, m_isdir
    ):
        # prepare input config
        types = ['raw', 'yx']
        config = self._create_config(types, ['type_only'])

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call load
        handler = self._create_handler(io_facades_provider=provider)
        dset_data = handler.load('testset', '/res', config)

        # verify returned data
        self.assertSetEqual(set(dset_data.keys()), set(types))

    def test_load_all_and_check_load_was_called_only_once_per_item_type(
            self, m_isdir
    ):
        # prepare input config
        types = ['gtux', 'gtuy']
        config = self._create_config(types, ['type_only'])
        num_items, item_types = config['num_items'], config['types']

        exp_filenames = self._get_exp_filenames('testing_dset', '/res', config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())
        facade = provider.get_instance(config['backend']['name'])

        # call load
        handler = self._create_handler(io_facades_provider=provider)
        handler.load('testing_dset', '/res', config)

        # verify expected calls
        self.assertEqual(len(types), facade.load.call_count)
        for item_type in types:
            type_config = item_types[item_type]
            facade.load.assert_any_call(exp_filenames[item_type],
                                        item_shape=type_config['shape'],
                                        dtype=type_config['dtype'],
                                        num_items=num_items)

    def test_load_subset_and_check_returned_items_are_indexed_by_type(
            self, m_isdir
    ):
        # prepare input config
        all_types = ['yx', 'gtux', 'test', 'some_type']
        load_types = ['yx', 'test']
        config = self._create_config(all_types, ['name_with_type_suffix'])

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call load
        handler = self._create_handler(io_facades_provider=provider)
        dset_data = handler.load('test', '/data', config,
                                 load_types=load_types)

        # verify returned data
        self.assertSetEqual(set(dset_data.keys()), set(load_types))

    def test_load_subset_and_check_load_was_called_only_once_per_item_type(
            self, m_isdir
    ):
        # prepare input config
        all_types = ['yx', 'gtux', 'test', 'some_type']
        load_types = ['yx', 'test']
        config = self._create_config(all_types, ['name_with_type_suffix'])

        num_items, item_types = config['num_items'], config['types']
        exp_filenames = self._get_exp_filenames('test', '/data', config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())
        facade = provider.get_instance(config['backend']['name'])

        # call load
        handler = self._create_handler(io_facades_provider=provider)
        handler.load('test', '/data', config, load_types=load_types)

        # verify expected calls
        self.assertEqual(len(load_types), facade.load.call_count)
        for item_type in load_types:
            type_config = item_types[item_type]
            facade.load.assert_any_call(exp_filenames[item_type],
                                        item_shape=type_config['shape'],
                                        dtype=type_config['dtype'],
                                        num_items=num_items)

    def test_load_invalid_subset_and_check_error_is_thrown(
            self, m_isdir
    ):
        # prepare input config
        all_types = ['yx', 'gtux', 'test', 'some_type']
        load_types = ['yx', 'test_2']
        config = self._create_config(all_types, ['name_with_type_suffix'])

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call load
        handler = self._create_handler(io_facades_provider=provider)
        self.assertRaises(ValueError, handler.load, 'dataset', '/set', config,
                          load_types=load_types)

    def test_save_and_check_returned_filenames_are_indexed_by_item_type(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['yx', 'test']
        config = self._create_config(item_types, ['type_only'])

        # prepare input data arrays
        items = self._create_items_from_config(config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # prepare expected data
        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # call save
        handler = self._create_handler(io_facades_provider=provider)
        filenames = handler.save('dataset', '/set', config, items)

        # verify result
        self.assertDictEqual(filenames, exp_filenames)

    def test_save_and_check_save_was_called_only_once_per_item_type(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['test1', 'test2']
        config = self._create_config(item_types, ['type_only'])
        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # prepare input data arrays
        items = self._create_items_from_config(config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())
        facade = provider.get_instance(config['backend']['name'])

        # call save
        handler = self._create_handler(io_facades_provider=provider)
        handler.save('dataset', '/set', config, items)

        # verify expected calls
        self.assertEqual(len(item_types), facade.save.call_count)
        for item_type in item_types:
            facade.save.assert_any_call(exp_filenames[item_type],
                                        items[item_type])

    def test_save_items_with_mismatched_lengths_and_check_error_is_thrown(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['test1', 'test2']
        config = self._create_config(item_types, ['type_only'])

        # prepare input data arrays
        items = self._create_items_from_config(config)
        items['test1'] = items['test1'][1:]

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call save
        handler = self._create_handler(io_facades_provider=provider)
        self.assertRaises(ValueError, handler.save, 'dataset', '/set', config,
                          items)

    def test_save_mismatched_item_and_config_types_and_check_error_is_thrown(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['test1', 'test2']
        config = self._create_config(item_types, ['type_only'])

        # prepare input data arrays
        items = self._create_items_from_config(config)
        del items['test2']

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call save
        handler = self._create_handler(io_facades_provider=provider)
        self.assertRaises(ValueError, handler.save, 'dataset', '/set', config,
                          items)

    def test_append_and_check_returned_filenames_are_indexed_by_item_type(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['yx2', 'raw']
        config = self._create_config(item_types, ['type_only'])

        # prepare input data arrays
        items = self._create_items_from_config(config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # prepare expected data
        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # call append
        handler = self._create_handler(io_facades_provider=provider)
        filenames = handler.append('dataset', '/set', config, items)

        # verify result
        self.assertDictEqual(filenames, exp_filenames)

    def test_append_and_check_append_was_called_only_once_per_item_type(
            self, m_isdir
    ):
        # prepare input config
        config = self._create_config(['test_raw', 'gtuy'], ['type_only'])
        num_items, item_types = config['num_items'], config['types']

        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # prepare input data arrays
        items = self._create_items_from_config(config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())
        facade = provider.get_instance(config['backend']['name'])

        # call append
        handler = self._create_handler(io_facades_provider=provider)
        handler.append('dataset', '/set', config, items)

        # verify expected calls
        self.assertEqual(len(item_types), facade.append.call_count)
        for item_type, values in item_types.items():
            facade.append.assert_any_call(exp_filenames[item_type],
                                          items[item_type],
                                          item_shape=values['shape'],
                                          dtype=values['dtype'],
                                          num_items=num_items)

    def test_append_items_with_mismatched_lengths_and_check_error_is_thrown(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['yx', 'test2']
        config = self._create_config(item_types, ['type_only'])

        # prepare input data arrays
        items = self._create_items_from_config(config)
        items['yx'] = items['yx'][1:]

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call append
        handler = self._create_handler(io_facades_provider=provider)
        self.assertRaises(ValueError, handler.append, 'name', '/set', config,
                          items)

    def test_append_mismatched_item_and_config_types_and_check_error_is_thrown(
            self, m_isdir
    ):
        # prepare input config
        item_types = ['test1', 'gtux']
        config = self._create_config(item_types, ['type_only'])

        # prepare input data arrays
        items = self._create_items_from_config(config)
        del items['gtux']

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call save
        handler = self._create_handler(io_facades_provider=provider)
        self.assertRaises(ValueError, handler.append, 'name', '/set', config,
                          items)

    def test_delete_all_and_check_returned_filenames_are_indexed_by_item_type(
            self, m_isdir
    ):
        # prepare input config
        types = ['test_raw', 'gtuy']
        config = self._create_config(types, ['name_with_type_suffix'])

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # prepare expected data
        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # call delete
        handler = self._create_handler(io_facades_provider=provider)
        filenames = handler.delete('dataset', '/set', config)

        # verify result
        self.assertDictEqual(filenames, exp_filenames)

    def test_delete_all_and_check_delete_was_called_only_once_per_item_type(
            self, m_isdir
    ):
        # prepare input config
        types = ['test_raw', 'gtuy']
        config = self._create_config(types, ['name_with_type_suffix'])

        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())
        facade = provider.get_instance(config['backend']['name'])

        # call delete
        handler = self._create_handler(io_facades_provider=provider)
        handler.delete('dataset', '/set', config)

        # verify expected calls
        self.assertEqual(len(types), facade.delete.call_count)
        for item_type in types:
            facade.delete.assert_any_call(exp_filenames[item_type])

    def test_delete_subset_and_check_returned_filenames_are_indexed_by_type(
            self, m_isdir
    ):
        # prepare input config
        all_types = ['yx', 'gtux', 'test', 'some_type']
        delete_types = ['yx', 'test']
        config = self._create_config(all_types, ['name_with_type_suffix'])

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # prepare expected data
        exp_filenames = self._get_exp_filenames('dataset', '/set', config,
                                                types_subset=delete_types)

        # call delete
        handler = self._create_handler(io_facades_provider=provider)
        filenames = handler.delete('dataset', '/set', config,
                                   delete_types=delete_types)

        # verify expected calls
        self.assertDictEqual(filenames, exp_filenames)

    def test_delete_subset_and_check_delete_was_called_only_once_per_type(
            self, m_isdir
    ):
        # prepare input config
        all_types = ['yx', 'gtux', 'test', 'some_type']
        delete_types = ['yx', 'test']
        config = self._create_config(all_types, ['name_with_type_suffix'])

        exp_filenames = self._get_exp_filenames('dataset', '/set', config)

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())
        facade = provider.get_instance(config['backend']['name'])

        # call delete
        handler = self._create_handler(io_facades_provider=provider)
        handler.delete('dataset', '/set', config, delete_types=delete_types)

        # verify expected calls
        self.assertEqual(len(delete_types), facade.delete.call_count)
        for item_type in delete_types:
            facade.delete.assert_any_call(exp_filenames[item_type])

    def test_delete_invalid_subset_and_check_error_is_thrown(
            self, m_isdir
    ):
        # prepare input config
        all_types = ['yx', 'gtux', 'test', 'some_type']
        delete_types = ['yx', 'test_delete']
        config = self._create_config(all_types, ['name_with_type_suffix'])

        # setup facades
        provider = self._get_mock_facades_provider(self._get_mock_facades())

        # call delete
        handler = self._create_handler(io_facades_provider=provider)
        self.assertRaises(ValueError, handler.delete, 'test', '/set', config,
                          delete_types=delete_types)


if __name__ == '__main__':
    unittest.main()

import os
import unittest
import unittest.mock as mock

import dataset.constants as cons
import dataset.data_utils as dat
import dataset.dataset_utils as ds
import dataset.io.fs.data.npy_io as data_io
import dataset.io.fs.meta.tsv_io as meta_io
import dataset.io.fs.targets.npy_io as targets_io
import dataset.io.fs_io as fs_io
import test.test_setups as testset


class TestDatasetFsPersistencyManager(testset.DatasetItemsMixin,
                                      testset.DatasetTargetsMixin,
                                      testset.DatasetMetadataMixin,
                                      unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDatasetFsPersistencyManager, cls).setUpClass()
        cls.meta_order = list(cls.metafields)
        cls.meta_order.sort()

        n_data = cls.n_packets
        item_types = cls.item_types
        item_types['gtux'], item_types['gtuy'] = False, False
        items = cls.items
        items['gtux'], items['gtuy'] = False, False
        # a mock dataset to use (also serves as value container)
        m_dataset = mock.create_autospec(ds.NumpyDataset)
        m_dataset.name = 'test'
        m_dataset.dtype = 'float32'
        m_dataset.num_data = n_data
        m_dataset.accepted_packet_shape = cls.packet_shape
        m_dataset.item_types = item_types
        m_dataset.item_shapes = dat.get_data_item_shapes(
            m_dataset.accepted_packet_shape, item_types)
        m_dataset.metadata_fields = cls.metafields
        m_dataset.get_data_as_dict.return_value = items
        m_dataset.get_data_as_arraylike.return_value = tuple(
            items[k] for k in cons.ALL_ITEM_TYPES if not item_types[k])
        m_dataset.get_targets.return_value = cls.mock_targets
        m_dataset.get_metadata.return_value = cls.mock_meta
        cls.m_dataset = m_dataset

        cls.configfile_contents = (
            '[general]{}'.format(os.linesep) +
            'num_data = {}{}'.format(n_data, os.linesep) +
            'metafields = {}{}'.format(cls.metafields, os.linesep) +
            'dtype = {}{}'.format(m_dataset.dtype, os.linesep) +
            '{}[packet_shape]{}'.format(os.linesep, os.linesep) +
            'num_frames = {}{}'.format(cls.n_f, os.linesep) +
            'frame_height = {}{}'.format(cls.f_h, os.linesep) +
            'frame_width = {}{}'.format(cls.f_w, os.linesep) +
            '{}[item_types]{}'.format(os.linesep, os.linesep) +
            ''.join('{} = {}{}'.format(k, item_types[k], os.linesep)
                    for k in cons.ALL_ITEM_TYPES) +
            os.linesep
        )

        cls.loaddir, cls.savedir = '/dsets', '/test'
        suffixes = {k: '_{}_test'.format(k) for k in cons.ALL_ITEM_TYPES}
        cls.datafiles = {k: '{}{}.npy'.format(m_dataset.name, suffixes[k])
                         for k in cons.ALL_ITEM_TYPES}
        conf_suffix = '_config_test'
        cls.configfile = '{}{}.ini'.format(m_dataset.name, conf_suffix)

        with mock.patch('os.path.isdir', return_value=True),\
             mock.patch('os.path.exists', return_value=True):
            cls.handler = fs_io.DatasetFsPersistencyHandler(
                cls.loaddir,
                cls.savedir,
                configfile_suffix=conf_suffix,
                data_handler = mock.create_autospec(
                    data_io.NumpyDataPersistencyHandler
                ),
                targets_handler=mock.create_autospec(
                    targets_io.NumpyTargetsPersistencyHandler
                ),
                metadata_handler=mock.create_autospec(
                    meta_io.TSVMetadataPersistencyHandler
                )
            )

    # test dataset loading methods

    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('os.path.isfile', return_value=True)
    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_load_dataset_config(self, m_open, m_isfile, m_exists):
        m_open.return_value = testset.MockTextFileStream(self.configfile_contents)
        m_dataset = self.m_dataset
        exp_attrs = { 'num_data': m_dataset.num_data,
            'metafields': m_dataset.metadata_fields,
            'item_types': m_dataset.item_types,
            'packet_shape': m_dataset.accepted_packet_shape,
            'dtype': m_dataset.dtype }
        exp_filename = os.path.join(self.loaddir, self.configfile)

        config = self.handler.load_dataset_config(m_dataset.name)
        self.assertDictEqual(config, exp_attrs)
        m_open.assert_called_once_with(exp_filename, encoding='UTF-8')

    @mock.patch('os.path.exists', return_value=False)
    @mock.patch('os.path.isfile', return_value=False)
    def test_load_dataset_config_configfile_does_not_exist(self, m_i, m_e):
        self.assertRaises(FileNotFoundError, self.handler.load_dataset_config,
                          self.m_dataset.name)

    @mock.patch('os.path.exists', return_value=True)
    @mock.patch('os.path.isfile', return_value=True)
    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_load_empty_dataset(self, m_open, m_isfile, m_exists):
        m_open.return_value = testset.MockTextFileStream(self.configfile_contents)
        name = self.m_dataset.name
        itypes = self.m_dataset.item_types
        dataset = self.handler.load_empty_dataset(name, itypes)
        self.assertEqual(dataset.num_data, 0)
        self.assertEqual(dataset.name, name)
        self.assertEqual(dataset.item_types, itypes)

    # test dataset save methods

    @mock.patch('builtins.open', new_callable=mock.mock_open())
    def test_save_dataset(self, m_open):
        m_open.return_value = buf = testset.MockTextFileStream()
        m_dataset = self.m_dataset
        name = m_dataset.name
        items, itypes = m_dataset.get_data_as_dict(), m_dataset.item_types
        exp_configfile = os.path.join(self.savedir, self.configfile)
        exp_filenames = {k: os.path.join(self.savedir, self.datafiles[k])
                         for k in items.keys()}

        self.handler.save_dataset(m_dataset, metafields_order=self.meta_order)
        # main postcondition asserted: that the config file was saved with
        # the proper filename
        self.assertEqual(buf.temp_buf, self.configfile_contents)
        m_open.assert_called_with(exp_configfile, 'w', encoding='UTF-8')
        # assert relevant data items were saved
        data_handler = self.handler.data_persistency_handler
        data_handler.save_data.assert_called_with(name, items, m_dataset.dtype)
        # targets and metadata save assertions
        m_targets, m_meta = m_dataset.get_targets(), m_dataset.get_metadata()
        targ_handler = self.handler.targets_persistency_handler
        targ_handler.save_targets.assert_called_with(name, m_targets)
        meta_handler = self.handler.metadata_persistency_handler
        meta_handler.save_metadata.assert_called_with(name, m_meta,
            metafields=m_dataset.metadata_fields,
            metafields_order=self.meta_order)


if __name__ == '__main__':
    unittest.main()

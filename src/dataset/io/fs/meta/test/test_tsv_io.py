import os
import unittest
import unittest.mock as mock

import dataset.io.fs.meta.tsv_io as tsv_io
import test.test_setups as testset


class TestTSVMetadataPersistencyManager(testset.DatasetMetadataMixin,
                                        unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestTSVMetadataPersistencyManager, cls).setUpClass()
        cls.name, cls.loaddir, cls.savedir = 'test', '/dsets', '/test'
        file_suffix = '_meta_test'
        cls.metafile = '{}{}.tsv'.format(cls.name, file_suffix)
        with mock.patch('os.path.isdir', return_value=True), \
             mock.patch('os.path.exists', return_value=True):
            cls.handler = tsv_io.TSVMetadataPersistencyHandler(
                cls.loaddir,
                cls.savedir,
                metafile_suffix=file_suffix
            )

    @mock.patch('utils.io_utils.load_TSV')
    def test_load_dataset_metadata(self, m_load):
        m_load.return_value = self.mock_meta
        exp_filename = os.path.join(self.loaddir, self.metafile)

        dset_meta = self.handler.load_metadata(self.name)
        self.assertListEqual(dset_meta, self.mock_meta)
        m_load.assert_called_once_with(exp_filename, selected_columns=None)

    @mock.patch('utils.io_utils.load_TSV')
    def test_load_dataset_metadata_specific_fields(self, m_load):
        m_load.return_value = self.mock_meta
        exp_filename = os.path.join(self.loaddir, self.metafile)
        metafields = [next(iter(self.metafields))]

        dset_meta = self.handler.load_metadata(self.name, metafields)
        self.assertListEqual(dset_meta, self.mock_meta)
        m_load.assert_called_once_with(exp_filename,
                                       selected_columns=set(metafields))

    @mock.patch('utils.io_utils.save_TSV')
    def test_save_dataset_metadata(self, m_save):
        exp_filename = os.path.join(self.savedir, self.metafile)
        ordered_meta = list(self.metafields)
        ordered_meta.sort()

        filename = self.handler.save_metadata(self.name, self.mock_meta)
        self.assertEqual(filename, exp_filename)
        m_save.assert_called_once_with(exp_filename, self.mock_meta,
                                       ordered_meta,
                                       file_exists_overwrite=True)

    @mock.patch('utils.io_utils.save_TSV')
    def test_save_dataset_metadata_unaccounted_metafields(self, m_save):
        metafields = self.metafields.copy()
        metafields.add('testfield')
        ordered_meta = list(self.metafields)
        ordered_meta.sort()

        self.assertRaises(Exception, self.handler.save_metadata, self.name,
                          self.mock_meta, metafields=metafields,
                          metafields_order=ordered_meta)
        m_save.assert_not_called()


if __name__ == '__main__':
    unittest.main()

import unittest

import dataset.metadata_utils as meta
import test.test_setups as testset

class TestModuleFunctions(unittest.TestCase):

    def test_extract_metafields(self):
        metadata = [{'test': 'val'}, {'test2': 'otherval'},
                    {'test': 'valval', 'test3': 'someval2'}]
        metafields = meta.extract_metafields(metadata)
        exp_metafields = set(['test', 'test2', 'test3'])
        self.assertSetEqual(metafields, exp_metafields)

class TestMetadataHolder(testset.DatasetMetadataMixin, unittest.TestCase):

    def test_extend_metadata(self):
        holder = meta.MetadataHolder()
        metadata = self.mock_meta
        metafields = self.metafields
        holder.extend(metadata)
        self.assertListEqual(holder[slice(None)], metadata)
        self.assertSetEqual(holder.metadata_fields, metafields)

    def test_append_metadata(self):
        holder = meta.MetadataHolder()
        metadata = [{'test': 'val'}, {'test2': 'otherval'},
                    {'test': 'valval', 'test3': 'someval2'}]
        holder.append(metadata[0])
        holder.append(metadata[1])
        exp_metadata = metadata[0:2]
        exp_metafields = set(['test', 'test2'])
        self.assertListEqual(holder[slice(None)], exp_metadata)
        self.assertSetEqual(holder.metadata_fields, exp_metafields)

    def test_add_metafield(self):
        holder = meta.MetadataHolder()
        metadata = [{'test': 'val'}, {'test2': 'otherval'},
                    {'test': 'valval', 'test3': 'someval2'}]
        holder.extend(metadata)
        exp_metadata = metadata.copy()
        for meta_dict in exp_metadata:
            meta_dict['somekey'] = 'someval'
        exp_metafields = set(['test', 'test2', 'test3', 'somekey'])
        holder.add_metafield('somekey', default_value='someval')
        self.assertListEqual(holder[slice(None)], exp_metadata)
        self.assertSetEqual(holder.metadata_fields, exp_metafields)

    def test_shuffle(self):
        holder = meta.MetadataHolder()
        metadata = [{'test': 'val'}, {'test2': 'otherval'},
                    {'test': 'valval', 'test3': 'someval2'}]
        holder.extend(metadata)
        exp_metadata = metadata.copy()
        def shuffler(seq):
            temp = seq[0]
            seq[0] = seq[1]
            seq[1] = temp
        shuffler(exp_metadata)
        exp_metafields = set(['test', 'test2', 'test3'])
        holder.shuffle(shuffler)
        self.assertListEqual(holder[slice(None)], exp_metadata)
        self.assertSetEqual(holder.metadata_fields, exp_metafields)

    def test_length_empty(self):
        holder = meta.MetadataHolder()
        self.assertEqual(len(holder), 0)

    def test_length_after_append(self):
        holder = meta.MetadataHolder()
        holder.append(self.mock_meta[0])
        self.assertEqual(len(holder), 1)

    def test_length_after_extend(self):
        holder = meta.MetadataHolder()
        holder.extend(self.mock_meta)
        self.assertEqual(len(holder), len(self.mock_meta))

if __name__ == '__main__':
    unittest.main()

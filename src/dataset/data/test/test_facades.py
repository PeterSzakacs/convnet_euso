import unittest

import numpy as np
import numpy.testing as nptest

import dataset.data.facades as facades
import dataset.data.test.utils as test_data_utils
import dataset.test.utils as test_utils


class TestDataFacade(test_data_utils.DataItemsDictUtilsMixin):

    # custom asserts

    def assertItemsTupleEqual(self, items, exp_items):
        self.assertEqual(len(items), len(exp_items))
        for idx in range(len(exp_items)):
            nptest.assert_array_equal(items[idx], exp_items[idx])

    # test setup

    @classmethod
    def setUpClass(cls):
        num_items, dtype = 8, np.uint8
        packets = np.ones((num_items, 100, 10, 20), dtype=dtype)
        items = {
            "raw": packets,
            "yx": np.ones((num_items, 10, 20), dtype=dtype),
            "gtux": np.ones((num_items, 100, 20), dtype=dtype),
            "gtuy": np.ones((num_items, 100, 10), dtype=dtype),
        }
        cls.num_items = num_items
        cls.dtype = dtype
        cls.packet_shape = packets[0].shape
        cls.items = items
        cls.item_shapes = {k: item[0].shape for k, item in items.items()}

    # test facade properties

    def test_item_types(self):
        included_types = ('yx', 'gtux')
        items = {k: self.items[k] for k in included_types}
        exp_types = {k: True for k in included_types}

        facade = facades.DataFacade(self.packet_shape, items)
        self.assertDictEqual(facade.item_types, exp_types)

    def test_item_shapes(self):
        included_types = ('yx', 'gtux', 'gtuy')
        items = {k: self.items[k] for k in included_types}
        exp_shapes = {k: self.item_shapes[k] for k in included_types}

        facade = facades.DataFacade(self.packet_shape, items)
        self.assertDictEqual(facade.item_shapes, exp_shapes)

    def test_packet_shape(self):
        facade = facades.DataFacade(self.packet_shape, self.items)
        self.assertTupleEqual(facade.packet_shape, self.packet_shape)

    def test_length(self):
        included_types = ('yx', 'gtux')
        items = {k: self.items[k] for k in included_types}

        facade = facades.DataFacade(self.packet_shape, items)
        self.assertEqual(len(facade), self.num_items)

    # test item retrieval methods (no copy on read)

    def test_get_data_as_dict(self):
        included_types = ('raw', 'yx')
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = {k: items[k] for k in included_types}

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsDictEqual(facade.get_data_as_dict(), exp_items)

    def test_get_data_as_dict_with_slice(self):
        included_types = ('yx', 'gtux', 'gtuy')
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = {k: items[k][0:2] for k in included_types}

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsDictEqual(facade.get_data_as_dict(slice(0, 2)),
                                  exp_items)

    def test_get_data_as_dict_with_range(self):
        included_types = ('raw', 'yx')
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = {k: items[k][range(1, 3)] for k in included_types}

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsDictEqual(facade.get_data_as_dict(range(1, 3)),
                                  exp_items)

    def test_get_data_as_dict_with_indexes_list(self):
        included_types = ('yx',)
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = {k: items[k][1, 4, 7] for k in included_types}

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsDictEqual(facade.get_data_as_dict([1, 4, 7]),
                                  exp_items)

    def test_get_data_as_arraylike(self):
        included_types = ('yx',)
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = tuple(items[k] for k in included_types)

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsTupleEqual(facade.get_data_as_tuple(), exp_items)

    def test_get_data_as_arraylike_with_slice(self):
        included_types = ('yx', 'gtux', 'gtuy')
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = tuple(items[k][4:6] for k in included_types)

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsTupleEqual(facade.get_data_as_tuple(slice(4, 6)),
                                   exp_items)

    def test_get_data_as_arraylike_with_range(self):
        included_types = ('gtux',)
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = tuple(items[k][range(3, 7)] for k in included_types)

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsTupleEqual(facade.get_data_as_tuple(range(3, 7)),
                                   exp_items)

    def test_get_data_as_arraylike_with_indexes_list(self):
        included_types = ('gtux', 'gtuy',)
        items = self.items
        in_items = {k: items[k] for k in included_types}
        exp_items = tuple(items[k][1, 6, 3] for k in included_types)

        facade = facades.DataFacade(self.packet_shape, in_items)
        self.assertItemsTupleEqual(facade.get_data_as_tuple((1, 6, 3)),
                                   exp_items)

    # test item shuffling

    def test_shuffle(self):
        included_types = ('raw', 'yx')
        items = self.items
        in_items = {k: items[k] for k in included_types}
        facade = facades.DataFacade(self.packet_shape, in_items)

        mock_shuffler = test_utils.MockShuffler(swap_indexes=[2, 3])
        exp_items = {k: items[k].copy() for k in included_types}
        mock_shuffler.shuffle(exp_items['raw'])
        mock_shuffler.reset_state()
        mock_shuffler.shuffle(exp_items['yx'])
        mock_shuffler.reset_state()

        facade.shuffle(mock_shuffler)
        self.assertItemsDictEqual(facade.get_data_as_dict(), exp_items)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import numpy.testing as nptest

import test.test_setups as testset
import utils.data_utils as dat

class TestDatasetUtilsFunctions(testset.DatasetItemsMixin, unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        super(TestDatasetUtilsFunctions, cls).setUpClass()
        cls.packet = cls.items['raw'][0]
        cls.start, cls.end = 0, 10

    # test create data item holders

    def test_create_packet_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['raw'])
        result = dat.create_packet_holder(self.packet_shape,
                                         num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_y_x_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['yx'])
        result = dat.create_y_x_projection_holder(self.packet_shape,
                                                 num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_gtu_x_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['gtux'])
        result = dat.create_gtu_x_projection_holder(self.packet_shape,
                                                   num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_gtu_y_projection_holder(self):
        exp_arr_shape = (self.n_packets, *self.item_shapes['gtuy'])
        result = dat.create_gtu_y_projection_holder(self.packet_shape,
                                                   num_items=self.n_packets)
        self.assertEqual(result.shape, exp_arr_shape)

    def test_create_data_holders(self):
        exp_shapes = {k: (self.n_packets, *self.item_shapes[k])
                         for k in dat.ALL_ITEM_TYPES}
        item_types = {k: True for k in dat.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in dat.ALL_ITEM_TYPES:
            holders = dat.create_data_holders(self.packet_shape, item_types,
                                             num_items=self.n_packets)
            holder_shapes = {k: (v.shape if v is not None else None)
                             for k, v in holders.items()}
            self.assertDictEqual(holder_shapes, exp_shapes)
            exp_shapes[item_type] = None
            item_types[item_type] = False

    def test_create_packet_holder_unknown_num_packets(self):
        result = dat.create_packet_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_y_x_projection_holder_unknown_num_packets(self):
        result = dat.create_y_x_projection_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_gtu_x_projection_holder_unknown_num_packets(self):
        result = dat.create_gtu_x_projection_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_gtu_y_projection_holder_unknown_num_packets(self):
        result = dat.create_gtu_y_projection_holder(self.packet_shape)
        self.assertListEqual(result, [])

    def test_create_data_holders(self):
        exp_holders = {k: [] for k in dat.ALL_ITEM_TYPES}
        item_types = {k: True for k in dat.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in dat.ALL_ITEM_TYPES:
            holders = dat.create_data_holders(self.packet_shape, item_types)
            self.assertDictEqual(holders, exp_holders)
            exp_holders[item_type] = None
            item_types[item_type] = False

    # test create packet projections

    def test_create_subpacket(self):
        expected_result = self.items['raw'][0][self.start:self.end]
        result = dat.create_subpacket(self.packet, start_idx=self.start,
                                     end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_create_y_x_projection(self):
        expected_result = self.items['yx'][0]
        result = dat.create_y_x_projection(self.packet, start_idx=self.start,
                                          end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_create_gtu_x_projection(self):
        expected_result = self.items['gtux'][0][self.start:self.end]
        result = dat.create_gtu_x_projection(self.packet, start_idx=self.start,
                                            end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_create_gtu_y_projection(self):
        expected_result = self.items['gtuy'][0][self.start:self.end]
        result = dat.create_gtu_y_projection(self.packet, start_idx=self.start,
                                            end_idx=self.end)
        nptest.assert_array_equal(result, expected_result)

    def test_convert_packet(self):
        exp_items = {'raw': self.items['raw'][0][self.start:self.end],
            'gtux': self.items['gtux'][0][self.start:self.end],
            'gtuy': self.items['gtuy'][0][self.start:self.end],
            'yx'  : self.items['yx'][0]}
        item_types = {k: True for k in dat.ALL_ITEM_TYPES}
        exp_types = {k: True for k in dat.ALL_ITEM_TYPES}
        ## gradually turn off all item types except 'gtuy'
        for item_type in dat.ALL_ITEM_TYPES:
            items = dat.convert_packet(self.packet, item_types,
                                      start_idx=self.start, end_idx=self.end)
            all_equal = {k: (not item_types[k] if v is None
                             else np.array_equal(v, exp_items[k]))
                             for k, v in items.items()}
            self.assertDictEqual(all_equal, exp_types)
            exp_items[item_type] = None
            item_types[item_type] = False

    # test get item shapes

    def test_get_y_x_projection_shape(self):
        result = dat.get_y_x_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['yx'])

    def test_get_gtu_x_projection_shape(self):
        result = dat.get_gtu_x_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['gtux'])

    def test_get_gtu_y_projection_shape(self):
        result = dat.get_gtu_y_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['gtuy'])

    def test_get_item_shape_gettters(self):
        # Test function which gets item shapes based on which item types are
        # set to True
        item_shapes = self.item_shapes.copy()
        item_types = {k: True for k in dat.ALL_ITEM_TYPES}
        for item_type in dat.ALL_ITEM_TYPES:
            shapes = dat.get_data_item_shapes(self.packet_shape, item_types)
            self.assertDictEqual(shapes, item_shapes)
            item_shapes[item_type] = None
            item_types[item_type] = False


class TestDataHolder(testset.DatasetItemsMixin, unittest.TestCase):

    # helper methods (custom asserts)

    def _assertItemsDtype(self, items_dict, exp_dtype, item_types):
        for itype, is_present in item_types.items():
            if is_present:
                err_msg = "wrong dtype for items of type '{}'".format(itype)
                ndarr = np.array(items_dict[itype])
                self.assertEqual(ndarr.dtype, exp_dtype, err_msg)

    def _assertItemsDict(self, data, exp_data, exp_item_types):
        # unfortunately, assertDictEqual does not work in this case
        self.assertSetEqual(set(data.keys()), set(exp_data.keys()),
                            "Returned item keys not equal")
        for itype, is_present in exp_item_types.items():
            if is_present:
                err_msg = "items of type '{}' are not equal".format(itype)
                nptest.assert_array_equal(data[itype], exp_data[itype],
                                          err_msg=err_msg)

    def _assertItemsArraylike(self, data, exp_data, exp_item_types):
        # unfortunately, assertTupleEqual does not work in this case
        self.assertEqual(len(data), len(exp_data))
        keys = [itype for itype in dat.ALL_ITEM_TYPES if exp_item_types[itype]]
        for idx in range(len(keys)):
            err_msg = "items of type '{}' are not equal".format(keys[idx])
            nptest.assert_array_equal(data[idx], exp_data[idx],
                                      err_msg=err_msg)

    # helper methods (items and types setup)

    def _create_items(self, contained_types, itm_slice):
        items = dict.fromkeys(dat.ALL_ITEM_TYPES, None)
        for itype in contained_types:
            items[itype] = self.items[itype][itm_slice]
        return items

    def _create_item_types(self, contained_types):
        item_types = dict.fromkeys(dat.ALL_ITEM_TYPES, False)
        for itype in contained_types:
            item_types[itype] = True
        return item_types

    # test methods
    ## test holder properties

    def test_item_shapes(self):
        # might perhaps be better to not even have the dict keys for items
        # that are not contained in the holder
        included_types = ('yx', 'gtux', 'gtuy')
        item_types = self._create_item_types(included_types)
        item_shapes = self.item_shapes.copy()
        item_shapes['raw'] = None
        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        self.assertDictEqual(holder.item_shapes, item_shapes)

    def test_accepted_packet_shape(self):
        packet_shape = self.packet_shape
        holder = dat.DataHolder(packet_shape)
        self.assertTupleEqual(holder._packet_shape, packet_shape)

    def test_item_types(self):
        included_types = ('yx', 'gtux')
        item_types = self._create_item_types(included_types)
        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        self.assertDictEqual(holder.item_types, item_types)

    ## test holder dtype

    def test_dtype_on_creation_empty(self):
        dtype = 'uint16'
        holder = dat.DataHolder(self.packet_shape, dtype=dtype)
        self.assertEqual(holder.dtype, dtype)

    def test_dtype_on_casting_empty(self):
        dtype = 'uint32'
        holder = dat.DataHolder(self.packet_shape, dtype=dtype)
        dtype = 'uint16'
        holder.dtype = dtype
        self.assertEqual(holder.dtype, dtype)

    def test_dtype_after_creation_with_items(self):
        dtype = 'uint16'
        included_types = ('raw', 'yx')
        item_types = self._create_item_types(included_types)
        holder = dat.DataHolder(self.packet_shape, dtype=dtype,
                                item_types=item_types)
        packets = self.items['raw'].copy()
        holder.extend_packets(packets)
        items = holder.get_data_as_dict()
        self._assertItemsDtype(items, dtype, item_types)
        self.assertEqual(holder.dtype, dtype)

    def test_dtype_after_casting_with_items(self):
        included_types = ('raw', 'yx')
        item_types = self._create_item_types(included_types)
        dtype = 'uint32'
        holder = dat.DataHolder(self.packet_shape, dtype=dtype,
                                item_types=item_types)
        packets = self.items['raw'].copy()
        holder.extend_packets(packets)
        dtype = 'uint16'
        holder.dtype = dtype
        items = holder.get_data_as_dict()
        self._assertItemsDtype(items, dtype, item_types)
        self.assertEqual(holder.dtype, dtype)

    ## test item retrieval methods

    def test_get_data_as_dict_empty(self):
        included_types = ('raw', 'yx')
        item_types = self._create_item_types(included_types)
        exp_items = {itype: [] for itype in included_types}

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)

    def test_get_data_as_dict(self):
        included_types = ('yx', 'gtux', 'gtuy')
        item_types = self._create_item_types(included_types)
        items = self._create_items(included_types, slice(0, 2))
        exp_items = {itype: list(items[itype]) for itype in included_types}

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.extend(items)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)

    def test_get_data_as_arraylike_empty(self):
        included_types = ('yx', )
        item_types = self._create_item_types(included_types)
        exp_items = tuple([] for itype in included_types)

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        self._assertItemsArraylike(holder.get_data_as_arraylike(), exp_items,
                                   item_types)

    def test_get_data_as_arraylike(self):
        included_types = ('yx', 'gtux', 'gtuy')
        item_types = self._create_item_types(included_types)
        items = self._create_items(included_types, slice(0, 2))
        exp_items = tuple(list(items[itype]) for itype in included_types)

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.extend(items)
        self._assertItemsArraylike(holder.get_data_as_arraylike(), exp_items,
                                   item_types)

    ## test item adding methods

    def test_extend(self):
        included_types = ('raw', 'gtux')
        items = self._create_items(included_types, slice(0, 2))
        item_types = self._create_item_types(included_types)
        exp_items = {itype: list(items[itype]) for itype in included_types}

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.extend(items)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)

    def test_extend_packets(self):
        included_types = ('raw', 'gtuy')
        items = self._create_items(included_types, slice(0, 2))
        item_types = self._create_item_types(included_types)
        exp_items = {itype: list(items[itype]) for itype in included_types}
        packets = items['raw']

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.extend_packets(packets)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)

    def test_append(self):
        included_types = ('yx', 'gtux')
        items = self._create_items(included_types, 0)
        item_types = self._create_item_types(included_types)
        exp_items = {itype: [items[itype]] for itype in included_types}

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.append(items)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)

    def test_append_packet(self):
        included_types = ('raw', 'gtuy')
        items = self._create_items(included_types, 0)
        item_types = self._create_item_types(included_types)
        exp_items = {itype: [items[itype]] for itype in included_types}
        packet = items['raw']

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.append_packet(packet)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)

    def test_append_packet_raises_error_on_misshaped_packet_passed(self):
        packet_shape = (self.n_f + 1, self.f_h, self.f_w)
        packet = np.ones(packet_shape)
        holder = dat.DataHolder(self.packet_shape)
        self.assertRaises(Exception, holder.append_packet, packet)

    def test_extend_packets_raises_error_on_misshaped_packets_passed(self):
        packet_shape = (self.n_f + 1, self.f_h, self.f_w)
        packets = list(self.items['raw'])
        packets.append(np.zeros(packet_shape))

        holder = dat.DataHolder(self.packet_shape)
        self.assertRaises(Exception, holder.extend_packets, packets)

    def test_append_raises_error_on_missing_items(self):
        item_types = self._create_item_types(('raw', 'yx', 'gtux'))
        items = self._create_items(('raw', 'yx'), 0)

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        self.assertRaises(Exception, holder.append, items)

    def test_extend_raises_error_on_missing_items(self):
        item_types = self._create_item_types(('raw', 'yx', 'gtuy'))
        items = self._create_items(('raw', 'yx'), slice(0, 2))

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        self.assertRaises(Exception, holder.extend, items)

    ## test item shuffling

    def test_shuffle(self):
        included_types = ('raw', 'yx')
        item_types = self._create_item_types(included_types)
        items = self._create_items(included_types, slice(0, 2))
        exp_items = {itype: items[itype] for itype in included_types}
        def shuffler(seq):
            temp = seq[0]
            seq[0] = seq[1]
            seq[1] = temp
        shuffler(exp_items['raw'])
        shuffler(exp_items['yx'])

        holder = dat.DataHolder(self.packet_shape, item_types=item_types)
        holder.extend_packets(items['raw'])
        holder.shuffle(shuffler, lambda: None)
        self._assertItemsDict(holder.get_data_as_dict(), exp_items, item_types)


if __name__ == '__main__':
    unittest.main()

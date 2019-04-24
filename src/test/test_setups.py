import io
import unittest

import numpy as np

import dataset.constants as cons
import dataset.dataset_utils as ds


class MockTextFileStream(io.StringIO):

    def __exit__(self, type, value, traceback):
        self.temp_buf = self.getvalue()
        super(MockTextFileStream, self).__exit__(type, value, traceback)


class DatasetItemsMixin(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls, num_items=2):
        if (num_items < 2):
            raise ValueError('Number of items must be at least 2')
        super(DatasetItemsMixin, cls).setUpClass()
        f_w, f_h, n_f = 48, 64, 20
        packet_shape = (n_f, f_h, f_w)
        item_shapes = {
            'raw' : (n_f, f_h, f_w),
            'yx'  : (f_h, f_w),
            'gtux': (n_f, f_w),
            'gtuy': (n_f, f_h),
        }
        items = {
            'raw' : np.ones((num_items, *item_shapes['raw'])),
            'yx'  : np.ones((num_items, *item_shapes['yx'])),
            'gtux': np.ones((num_items, *item_shapes['gtux'])),
            'gtuy': np.ones((num_items, *item_shapes['gtuy']))}
        item_types = {k: True for k in cons.ALL_ITEM_TYPES}
        # make some randomness in the first packet and derived items
        packet = items['raw'][0]
        packet[0, 0, 0], packet[1] = 3, 4
        yx = items['yx'][0]
        yx.fill(4)
        gtux, gtuy = items['gtux'][0], items['gtuy'][0]
        gtux[0, 0], gtuy[0, 0], gtux[1], gtuy[1] = 3, 3, 4, 4
        cls.n_packets = num_items
        cls.n_f, cls.f_h, cls.f_w = n_f, f_h, f_w
        cls.packet_shape = (n_f, f_h, f_w)
        cls.item_shapes = item_shapes
        cls.items = items
        cls.item_types = item_types


class DatasetTargetsMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls, num_items=2):
        if (num_items < 2):
            raise ValueError('Number of items must be at least 2')
        super(DatasetTargetsMixin, cls).setUpClass()
        cls.mock_targets = [[0, 0] if idx % 2 == 0 else [0, 1]
                            for idx in range(num_items)]


class DatasetMetadataMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls, num_items=2):
        if (num_items < 2):
            raise ValueError('Number of items must be at least 2')
        super(DatasetMetadataMixin, cls).setUpClass()
        meta_dict = {k: None for k in cons.FLIGHT_METADATA}
        cls.mock_meta = [meta_dict.copy() for idx in range(num_items)]
        cls.metafields = set(cons.FLIGHT_METADATA)


class DatasetMixin():

    @classmethod
    def setUpClass(cls, num_items=2, name='test', item_types=None):
        items_mixin = DatasetItemsMixin()
        items_mixin.setUpClass(num_items=num_items)
        targets_mixin = DatasetTargetsMixin()
        targets_mixin.setUpClass(num_items=num_items)
        meta_mixin = DatasetMetadataMixin()
        meta_mixin.setUpClass(num_items=num_items)
        packets = items_mixin.items['raw']
        targets = targets_mixin.mock_targets
        metadata = meta_mixin.mock_meta

        if item_types is None:
            item_types = items_mixin.item_types
        dset = ds.NumpyDataset(name, items_mixin.packet_shape,
                               item_types=item_types)
        for idx in range(num_items):
            dset.add_data_item(packets[idx], targets[idx], metadata[idx])
        cls.dset = dset

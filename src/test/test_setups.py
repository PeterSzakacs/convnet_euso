import io
import unittest

import numpy as np

import utils.dataset_utils as ds
import utils.metadata_utils as meta


class MockTextFileStream(io.StringIO):

    def __exit__(self, type, value, traceback):
        self.temp_buf = self.getvalue()
        super(MockTextFileStream, self).__exit__(type, value, traceback)


_NUM_PACKETS = 2


class DatasetItemsMixin(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        super(DatasetItemsMixin, cls).setUpClass()
        n_packets = _NUM_PACKETS
        f_w, f_h, n_f = 48, 64, 20
        packet_shape = (n_f, f_h, f_w)
        item_shapes = {
            'raw' : (n_f, f_h, f_w),
            'yx'  : (f_h, f_w),
            'gtux': (n_f, f_w),
            'gtuy': (n_f, f_h),
        }
        items = {
            'raw' : np.ones((n_packets, *item_shapes['raw'])),
            'yx'  : np.ones((n_packets, *item_shapes['yx'])),
            'gtux': np.ones((n_packets, *item_shapes['gtux'])),
            'gtuy': np.ones((n_packets, *item_shapes['gtuy']))}
        item_types = {k: True for k in ds.ALL_ITEM_TYPES}
        # make some randomness in the first packet and derived items
        packet = items['raw'][0]
        packet[0, 0, 0], packet[1] = 3, 4
        yx = items['yx'][0]
        yx.fill(4)
        gtux, gtuy = items['gtux'][0], items['gtuy'][0]
        gtux[0, 0], gtuy[0, 0], gtux[1], gtuy[1] = 3, 3, 4, 4
        cls.n_packets = n_packets
        cls.n_f, cls.f_h, cls.f_w = n_f, f_h, f_w
        cls.packet_shape = (n_f, f_h, f_w)
        cls.item_shapes = item_shapes
        cls.items = items
        cls.item_types = item_types


class DatasetTargetsMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(DatasetTargetsMixin, cls).setUpClass()
        n_targets = _NUM_PACKETS
        cls.mock_targets = np.zeros((n_targets, 2))


class DatasetMetadataMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(DatasetMetadataMixin, cls).setUpClass()
        n_meta = _NUM_PACKETS
        meta_dict = {k: None for k in meta.FLIGHT_METADATA}
        cls.mock_meta = [meta_dict.copy() for idx in range(n_meta)]
        cls.metafields = set(meta.FLIGHT_METADATA)


import unittest

import dataset.data.constants as cons
import dataset.data.shapes as shapes


class TestUtilsFunctions(unittest.TestCase):

    # test setup

    @classmethod
    def setUpClass(cls):
        n_f, f_h, f_w = 128, 16, 32
        packet_shape = (n_f, f_h, f_w)
        item_shapes = {
            "raw": packet_shape,
            "yx": (f_h, f_w),
            "gtux": (n_f, f_w),
            "gtuy": (n_f, f_h),
        }
        cls.packet_shape = packet_shape
        cls.item_shapes = item_shapes

    # test get item shapes

    def test_get_y_x_projection_shape(self):
        result = shapes.get_y_x_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['yx'])

    def test_get_gtu_x_projection_shape(self):
        result = shapes.get_gtu_x_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['gtux'])

    def test_get_gtu_y_projection_shape(self):
        result = shapes.get_gtu_y_projection_shape(self.packet_shape)
        self.assertTupleEqual(result, self.item_shapes['gtuy'])

    def test_get_item_shape_gettters(self):
        # Test function which gets item shapes based on which item types are
        # set to True
        item_types = {k: True for k in cons.ALL_ITEM_TYPES}
        exp_item_shapes = self.item_shapes.copy()
        item_types['yx'] = False
        del exp_item_shapes['yx']

        item_shapes = shapes.get_data_item_shapes(self.packet_shape,
                                                  item_types)
        self.assertDictEqual(item_shapes, exp_item_shapes)


if __name__ == '__main__':
    unittest.main()

import unittest

import utils.packets.packet_utils as pack

# Note: EC height is delibarately unrealistic and different from EC_width for better testing and debugging
class TestPacketTemplate(unittest.TestCase):

    def test_visible_properties(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 128
        num_EC = (width * height) / (EC_width * EC_height)
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        self.assertEqual(EC_width, template.EC_width)
        self.assertEqual(EC_height, template.EC_height)
        self.assertEqual(width, template.frame_width)
        self.assertEqual(height, template.frame_height)
        self.assertEqual(int(width/EC_width), template.num_cols)
        self.assertEqual(int(height/EC_height), template.num_rows)
        self.assertEqual(num_EC, template.num_EC)
        self.assertEqual(num_frames, template.num_frames)
        self.assertEqual((num_frames, height, width), template.packet_shape)


    def test_unit_conversions(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 128
        template = pack.packet_template(EC_width, EC_height, width, height, num_frames)
        xs      = [0,  10, 21, 30, 21, 40, 0]
        ys      = [10, 0,  1,  42, 10, 50, 0]
        ec_ids  = [0,  0,  1,  4,  1,  5,  0]
        ec_xys  = [[0, 0], [0, 0], [1, 0], [1, 1], [1, 0], [2, 1], [0, 0]]
        for idx in range(len(xs)):
            x, y, ec_x, ec_y, ec_idx = xs[idx], ys[idx], ec_xys[idx][0], ec_xys[idx][1], ec_ids[idx]
            self.assertEqual(template.x_to_ec_x(x), ec_x)
            self.assertEqual(template.y_to_ec_y(y), ec_y)
            self.assertEqual(template.ec_idx_to_ec_xy(ec_idx), (ec_x, ec_y))
            self.assertEqual(template.xy_to_ec_idx(x, y), ec_idx)
            self.assertEqual(template.ec_xy_to_ec_idx(ec_x, ec_y), ec_idx)

if __name__ == '__main__':
    unittest.main()
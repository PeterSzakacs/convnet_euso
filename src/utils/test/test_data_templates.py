import unittest

import utils.shower_generators as gen
import utils.data_templates as templates

# Note: EC height is delibarately unrealistic and different from EC_width for better testing and debugging
class TestPacketTemplate(unittest.TestCase):

    def test_visible_properties(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 128
        num_EC = (width * height) / (EC_width * EC_height)
        template = templates.packet_template(EC_width, EC_height, width, height, num_frames)
        self.assertEqual(EC_width, template.EC_width)
        self.assertEqual(EC_height, template.EC_height)
        self.assertEqual(width, template.frame_width)
        self.assertEqual(height, template.frame_height)
        self.assertEqual(int(width/EC_width), template.num_cols)
        self.assertEqual(int(height/EC_height), template.num_rows)
        self.assertEqual(num_EC, template.num_EC)
        self.assertEqual(num_frames, template.num_frames)
        self.assertEqual((num_frames, height, width), template.packet_shape)

        # test creating a template from incosistent properties
        ## first: negative values
        self.assertRaises(ValueError, templates.packet_template, -1, EC_height, width, height, num_frames)
        self.assertRaises(ValueError, templates.packet_template, EC_width, -1, width, height, num_frames)
        self.assertRaises(ValueError, templates.packet_template, EC_width, EC_height, -1, height, num_frames)
        self.assertRaises(ValueError, templates.packet_template, EC_width, EC_height, width, -1, num_frames)
        self.assertRaises(ValueError, templates.packet_template, EC_width, EC_height, width, height, -1)

        # second: width and height that are not evenly divisible by EC_width or EC_height respectively
        self.assertRaises(ValueError, templates.packet_template, EC_height+1, EC_width, width, height, num_frames)
        self.assertRaises(ValueError, templates.packet_template, EC_height, EC_width+1, width, height, num_frames)

    def test_unit_conversions(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 128
        template = templates.packet_template(EC_width, EC_height, width, height, num_frames)
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
        self.assertTupleEqual(template.ec_idx_to_xy_slice(1), (slice(16,32), slice(0,32)))

    def test_equality_check(self):
        EC_width, EC_height = 16, 32
        width, height, num_frames = 48, 64, 128
        template = templates.packet_template(EC_width, EC_height, width, height, num_frames)
        template2 = templates.packet_template(EC_width, EC_height, width, height, num_frames)
        self.assertEqual(template, template2)
        template2 = templates.packet_template(EC_width, EC_height, width + EC_width, height, num_frames)
        self.assertNotEqual(template, template2)
        template2 = templates.packet_template(EC_width, EC_height, width, height + EC_height, num_frames)
        self.assertNotEqual(template, template2)
        template2 = templates.packet_template(EC_width, EC_height, width, height, num_frames + 1)
        self.assertNotEqual(template, template2)


class TestSimuShowerTemplate(unittest.TestCase):

    def _assert_set_prop_raises(self, obj, prop_name, value, expected_exception):
        with self.assertRaises(expected_exception, msg=('Failed to raise {}'
                               ' for property {} being set to {}').format(
                                   expected_exception, prop_name, value)):
            setattr(obj, prop_name, value)

    def _assert_set_prop_not_raises(self, obj, prop_name, value):
        try:
            setattr(obj, prop_name, value)
        except Exception:
            self.fail(msg=('Exception raised when setting property {}'
                           ' to value {}').format(prop_name, value))

    def test_property_checking(self):
        gtu, w, h, ec_w, ec_h = 128, 64, 48, 32, 16
        template = templates.packet_template(ec_w, ec_h, w, h, gtu)
        start_x, start_y, start_gtu = (3, 5), (1, 10), (2, 4)
        duration, shower_max = (2, 10), (7, 15)
        shower_template = templates.simulated_shower_template(template, duration, shower_max)

        # start_x lower bound is less than 0, upper bound is less than lower bound or upper bound is larger than frame width
        self._assert_set_prop_raises(shower_template, 'start_x', (-start_x[0], start_x[1]), ValueError)
        self._assert_set_prop_raises(shower_template, 'start_x', (start_x[1], start_x[0]), ValueError)
        self._assert_set_prop_raises(shower_template, 'start_x', (start_x[0], w+1), ValueError)
        self._assert_set_prop_not_raises(shower_template, 'start_x', start_x)
        # start_y lower bound is less than 0, upper bound is less than lower bound or upper bound is larger than frame height
        self._assert_set_prop_raises(shower_template, 'start_y', (-start_y[0], start_y[1]), ValueError)
        self._assert_set_prop_raises(shower_template, 'start_y', (start_y[1], start_y[0]), ValueError)
        self._assert_set_prop_raises(shower_template, 'start_y', (start_y[0], h+1), ValueError)
        self._assert_set_prop_not_raises(shower_template, 'start_y', start_y)
        # start_gtu lower bound is less than 0, upper bound is less than lower bound or upper bound is larger than number of frames
        self._assert_set_prop_raises(shower_template, 'start_gtu', (-start_gtu[0], start_gtu[1]), ValueError)
        self._assert_set_prop_raises(shower_template, 'start_gtu', (start_gtu[1], start_gtu[0]), ValueError)
        self._assert_set_prop_raises(shower_template, 'start_gtu', (start_gtu[0], gtu+1), ValueError)
        self._assert_set_prop_not_raises(shower_template, 'start_gtu', start_gtu)
        # duration lower bound is less than 1, upper bound is less than lower bound or upper bound is larger than the number of frames
        self._assert_set_prop_raises(shower_template, 'shower_duration', (0, duration[1]), ValueError)
        self._assert_set_prop_raises(shower_template, 'shower_duration', (duration[1], duration[0]), ValueError)
        self._assert_set_prop_raises(shower_template, 'shower_duration', (duration[0], gtu+1), ValueError)
        self._assert_set_prop_not_raises(shower_template, 'duration', duration)
        # bg_diff lower bound is less than 1 or upper bound is less than lower bound
        self._assert_set_prop_raises(shower_template, 'shower_max', (0, shower_max[1]), ValueError)
        self._assert_set_prop_raises(shower_template, 'shower_max', (shower_max[1], shower_max[0]), ValueError)
        self._assert_set_prop_not_raises(shower_template, 'shower_max', shower_max)

    def test_property_generators(self):
        gtu, w, h, ec_w, ec_h = 128, 64, 48, 32, 16
        template = templates.packet_template(ec_w, ec_h, w, h, gtu)
        sx, sy, sg = 3, 10, 2
        d, m = 7, 15
        shower_template = templates.simulated_shower_template(template,
                                                             (d, d), (m, m),
                                                             start_x=(sx, sx),
                                                             start_y=(sy, sy),
                                                             start_gtu=(sg, sg))
        # if MIN == MAX, returned value should never change
        for idx in range(10):
            self.assertTupleEqual(shower_template.get_new_start_coordinate(), (sg, sy, sx))
            self.assertEqual(shower_template.get_new_shower_max(), m)
            self.assertEqual(shower_template.get_new_shower_duration(), d)

        sx, sy, sg = (3, 5), (1, 10), (2, 4)
        d, m = (2, 10), (7, 15)
        shower_template.start_x, shower_template.start_y = sx, sy
        shower_template.start_gtu = sg
        shower_template.shower_duration, shower_template.shower_max = d, m
        # if MIN != MAX, returned values should be within a certain range
        for idx in range(10):
            start_gtu, start_y, start_x = shower_template.get_new_start_coordinate()[0:3]
            self.assertGreaterEqual(start_gtu, sg[0])
            self.assertGreaterEqual(start_y, sy[0])
            self.assertGreaterEqual(start_x, sx[0])
            self.assertLessEqual(start_gtu, sg[1])
            self.assertLessEqual(start_y, sy[1])
            self.assertLessEqual(start_x, sx[1])
            shower_max = shower_template.get_new_shower_max()
            self.assertGreaterEqual(shower_max, m[0])
            self.assertLessEqual(shower_max, m[1])
            duration = shower_template.get_new_shower_duration()
            self.assertGreaterEqual(duration, d[0])
            self.assertLessEqual(duration, d[1])

    def test_equality_check(self):
        gtu, w, h, ec_w, ec_h = 128, 64, 48, 32, 16
        template = templates.packet_template(ec_w, ec_h, w, h, gtu)
        sx, sy, sg = (3, 5), (1, 10), (2, 4)
        d, m = (2, 10), (7, 15)
        shower_template = templates.simulated_shower_template(template, d, m,
                                                              start_x=sx,
                                                              start_y=sy,
                                                              start_gtu=sg)
        shower_template2 = templates.simulated_shower_template(template, d, m,
                                                               start_x=sx,
                                                               start_y=sy,
                                                               start_gtu=sg)
        self.assertEqual(shower_template, shower_template2)

        shower_template2.start_x = (sx[0], sx[1] + 1)
        self.assertNotEqual(shower_template, shower_template2)
        shower_template2.start_x = sx
        shower_template2.start_y = (sy[0], sy[1] + 1)
        self.assertNotEqual(shower_template, shower_template2)
        shower_template2.start_y = sy
        shower_template2.start_gtu = (sg[0], sg[1] + 1)
        self.assertNotEqual(shower_template, shower_template2)
        shower_template2.start_gtu = sg
        shower_template2.shower_duration = (d[0], d[1] + 1)
        self.assertNotEqual(shower_template, shower_template2)
        shower_template2.shower_duration = d
        shower_template2.shower_max = (m[0], m[1] + 1)
        self.assertNotEqual(shower_template, shower_template2)
        shower_template2.shower_max = m
        shower_template2.values_generator = gen.flat_vals_generator(10, 10)
        self.assertNotEqual(shower_template, shower_template2)


class TestSimuShowerTemplate(unittest.TestCase):

    def test_equality_check(self):
        gtu, w, h, ec_w, ec_h = 128, 64, 48, 32, 16
        t = templates.packet_template(ec_w, ec_h, w, h, gtu)

        lam, bec = (0.1, 0.6), (1, 2)
        bg_temp = templates.synthetic_background_template(t, bg_lambda=lam,
                                                          bad_ECs_range=bec)
        bg_temp2 = templates.synthetic_background_template(t, bg_lambda=lam,
                                                           bad_ECs_range=bec)
        self.assertEqual(bg_temp, bg_temp2)
        bg_temp2.bg_lambda_range = (0.2, 0.6)
        self.assertNotEqual(bg_temp, bg_temp2)
        bg_temp2.bg_lambda_range = lam
        bg_temp2.bad_ECs_range = (3, 4)
        self.assertNotEqual(bg_temp, bg_temp2)

if __name__ == '__main__':
    unittest.main()
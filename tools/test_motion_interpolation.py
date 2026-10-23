import unittest
from unittest import mock

import cv2
import numpy as np

from tools.motion_interpolation import MotionPair, grid_between


class MotionInterpolationTests(unittest.TestCase):
    def pair_with_flows(self, a, b, forward, backward, scale=1):
        dis = mock.Mock()
        dis.calc.side_effect = [forward, backward]
        with mock.patch.object(cv2, 'DISOpticalFlow_create', return_value=dis):
            return MotionPair(a, b, cv2.resize(a, None, fx=scale, fy=scale),
                              cv2.resize(b, None, fx=scale, fy=scale))

    def test_inconsistent_flow_is_explicit_nearest_not_fresh_motion(self):
        a = np.full((96, 160, 3), 90, np.uint8)
        forward = np.full((96, 160, 2), 8, np.float32)
        pair = self.pair_with_flows(a, a, forward, forward)
        image, kind = pair.render(.4)
        self.assertEqual(kind, 'UNRELIABLE_NEAREST')
        self.assertTrue(np.array_equal(image, a))
        self.assertLess(pair.diagnostics['reliable_fraction'], .2)
        self.assertEqual(pair.render(1)[1], 'REAL_B')

    def test_local_new_object_uses_endpoint_without_gray_ghost(self):
        a = np.full((96, 160, 3), 50, np.uint8)
        b = a.copy()
        b[32:64, 64:96] = 200
        flow = np.zeros((96, 160, 2), np.float32)
        pair = self.pair_with_flows(a, b, flow, flow, scale=2)
        result, kind = pair.render(.6)
        self.assertEqual(kind, 'UNRELIABLE_NEAREST')
        self.assertTrue(np.array_equal(result, pair.b))
        self.assertTrue(np.array_equal(result[68:120, 132:184],
                                       pair.b[68:120, 132:184]))
        self.assertGreater(pair.diagnostics['unreliable_pixel_fraction'], .03)
        self.assertFalse(pair.diagnostics['global_fallback'])

    def test_reliable_pair_keeps_blend_without_endpoint_pixel_splicing(self):
        a = np.full((96, 160, 3), 90, np.uint8)
        b = np.full_like(a, 100)
        flow = np.zeros((96, 160, 2), np.float32)
        pair = self.pair_with_flows(a, b, flow, flow)
        image, kind = pair.render(.4)
        self.assertEqual(kind, 'FLOW')
        self.assertTrue(np.array_equal(image, cv2.addWeighted(a, .6, b, .4, 0)))

    def test_real_dis_rejects_ghost_in_newly_visible_region(self):
        rng = np.random.default_rng(17)
        a = cv2.GaussianBlur(rng.integers(40, 100, (180, 320, 3), dtype=np.uint8), (5, 5), 0)
        b = a.copy()
        cv2.rectangle(b, (130, 70), (190, 120), (230, 230, 230), -1)
        # Known overlay event at t=.5. At t=2/3 the newly visible region is B.
        result, kind = MotionPair(a, b, a, b).render(2/3)
        region = np.s_[75:115, 140:180]
        error = np.abs(result[region].astype(float)-b[region]).mean()
        blend = cv2.addWeighted(a, 1/3, b, 2/3, 0)
        blend_error = np.abs(blend[region].astype(float)-b[region]).mean()
        self.assertEqual(kind, 'UNRELIABLE_NEAREST')
        self.assertLess(error, blend_error*.1)

    def test_same_histogram_scene_change_is_rejected_by_photo_gate(self):
        a = np.random.default_rng(25).integers(0, 256, (96, 160, 3), dtype=np.uint8)
        a = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        b = np.flip(a, axis=1).copy()
        flow = np.zeros((96, 160, 2), np.float32)
        pair = self.pair_with_flows(a, b, flow, flow)
        # A spatially shuffled image need not trip a histogram cut detector;
        # photometric rejection must still suppress unreliable local blending.
        image, kind = pair.render(.4)
        self.assertIn(kind, ('FLOW', 'UNRELIABLE_NEAREST'))
        self.assertLess(pair.diagnostics['reliable_fraction'], .5)
        if kind == 'FLOW':
            self.assertGreater(pair.diagnostics['unreliable_pixel_fraction'], .5)

    def test_large_but_consistent_motion_uses_global_protection(self):
        a = np.full((96, 160, 3), 90, np.uint8)
        flow = np.zeros((96, 160, 2), np.float32)
        flow[:, :, 0] = 40
        pair = self.pair_with_flows(a, a, flow, -flow)
        self.assertGreater(pair.diagnostics['reliable_fraction'], .2)
        self.assertEqual(pair.render(.6)[1], 'UNRELIABLE_NEAREST')

    def test_uniform_grid_10_to_15_and_boundary_not_duplicated(self):
        rows = []
        index = 0
        for n in range(10):
            batch = list(grid_between(n/10, (n+1)/10, index, 15))
            rows.extend(batch)
            index = batch[-1][0]+1
        self.assertEqual([r[0] for r in rows], list(range(16)))
        self.assertTrue(all(0 <= r[2] <= 1 for r in rows))
        self.assertEqual(sum(1 for _, _, a in rows[:-1] if 1e-5 < a < 1-1e-5), 10)

    def test_motion_midpoint_beats_unwarped_blend(self):
        rng = np.random.default_rng(10)
        a = rng.integers(30, 220, (96, 160, 3), dtype=np.uint8)
        a = cv2.GaussianBlur(a, (5, 5), 0)
        def shift(x):
            return cv2.warpAffine(a, np.float32([[1, 0, x], [0, 1, 0]]),
                                  (160, 96), borderMode=cv2.BORDER_REFLECT)
        b, truth = shift(8), shift(4)
        result, kind = MotionPair(a, b, a, b).render(.5)
        roi = np.s_[12:-12, 20:-20]
        error = np.abs(result[roi].astype(float)-truth[roi]).mean()
        blend_error = np.abs(cv2.addWeighted(a, .5, b, .5, 0)[roi].astype(float)-truth[roi]).mean()
        self.assertEqual(kind, "FLOW")
        self.assertLess(error, blend_error * .6)

    def test_cut_not_counted_as_motion_frame(self):
        a = np.zeros((96, 160, 3), np.uint8)
        b = np.full_like(a, 255)
        frame, kind = MotionPair(a, b, a, b).render(.4)
        self.assertEqual(kind, "CUT_NEAREST")
        self.assertTrue(np.array_equal(frame, a))

    def test_dimensions_and_no_extrapolation(self):
        a = np.zeros((96, 160, 3), np.uint8)
        pair = MotionPair(a, a, cv2.resize(a, (320, 192)), cv2.resize(a, (320, 192)))
        frame, _ = pair.render(.33)
        self.assertEqual(frame.shape, (192, 320, 3))
        with self.assertRaises(ValueError):
            pair.render(1.1)


if __name__ == "__main__":
    unittest.main()

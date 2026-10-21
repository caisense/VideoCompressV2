import unittest

import cv2
import numpy as np

from tools.motion_interpolation import MotionPair, grid_between


class MotionInterpolationTests(unittest.TestCase):
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

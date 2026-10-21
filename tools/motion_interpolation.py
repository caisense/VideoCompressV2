"""Experimental bidirectional DIS motion compensation, with explicit cut fallback.

Flow is estimated on decoded frames; the x2 native GAN images are warped at
their native size. This is CPU optical flow, not RIFE or neural interpolation.
"""
from __future__ import annotations

import cv2
import numpy as np


def grid_between(left: float, right: float, next_index: int, fps: float):
    """Yield a strictly increasing uniform media grid, without extrapolation."""
    if fps <= 0 or right <= left:
        raise ValueError("fps must be positive and timestamps must increase")
    next_index = max(next_index, int(np.ceil((left - 1e-7) * fps)))
    while next_index / fps <= right + 1e-7:
        timestamp = next_index / fps
        alpha = float(np.clip((timestamp - left) / (right - left), 0, 1))
        yield next_index, timestamp, alpha
        next_index += 1


class MotionPair:
    def __init__(self, source_a, source_b, enhanced_a, enhanced_b):
        if source_a.shape != source_b.shape or enhanced_a.shape != enhanced_b.shape:
            raise ValueError("paired frame dimensions must match")
        self.a, self.b = enhanced_a, enhanced_b
        gray_a = cv2.cvtColor(source_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(source_b, cv2.COLOR_BGR2GRAY)
        # A global cut has no reliable intermediate motion. Do not blend it.
        hist_a = cv2.calcHist([gray_a], [0], None, [32], [0, 256])
        hist_b = cv2.calcHist([gray_b], [0], None, [32], [0, 256])
        cv2.normalize(hist_a, hist_a, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_b, hist_b, norm_type=cv2.NORM_L1)
        self.cut = (cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA) > .65
                    and cv2.absdiff(gray_a, gray_b).mean() > 45)
        self.flow = None
        if self.cut:
            return
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        forward = dis.calc(gray_a, gray_b, None)
        backward = dis.calc(gray_b, gray_a, None)
        h, w = enhanced_a.shape[:2]
        scale = np.array([w / gray_a.shape[1], h / gray_a.shape[0]], np.float32)
        self.forward = cv2.resize(forward, (w, h)) * scale
        self.backward = cv2.resize(backward, (w, h)) * scale
        self.x, self.y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        self.flow = forward

    def render(self, alpha):
        if not 0 <= alpha <= 1:
            raise ValueError("interpolation must not extrapolate")
        if alpha < 1e-5:
            return self.a, "REAL_A"
        if alpha > 1 - 1e-5:
            return self.b, "REAL_B"
        if self.cut:
            return (self.a if alpha < .5 else self.b), "CUT_NEAREST"
        # Inverse-warp approximation. Weight only in-bounds samples. This
        # handles image boundaries, but is not a learned disocclusion model.
        ax = self.x - alpha * self.forward[:, :, 0]
        ay = self.y - alpha * self.forward[:, :, 1]
        bx = self.x - (1-alpha) * self.backward[:, :, 0]
        by = self.y - (1-alpha) * self.backward[:, :, 1]
        h, w = self.a.shape[:2]
        valid_a = ((ax >= 0) & (ax <= w-1) & (ay >= 0) & (ay <= h-1))
        valid_b = ((bx >= 0) & (bx <= w-1) & (by >= 0) & (by <= h-1))
        a = cv2.remap(self.a, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.remap(self.b, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result = cv2.addWeighted(a, 1-alpha, b, alpha, 0)
        cv2.copyTo(a, (valid_a & ~valid_b).astype(np.uint8), result)
        cv2.copyTo(b, (valid_b & ~valid_a).astype(np.uint8), result)
        return result, "FLOW"

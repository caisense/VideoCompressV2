"""Bidirectional DIS with consistency/photometric gates and explicit fallbacks.

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
        self.diagnostics = dict(cut=bool(self.cut), reliable_fraction=0.,
                                motion_p95_px=0., global_fallback=False)
        if self.cut:
            return
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        forward = dis.calc(gray_a, gray_b, None)
        backward = dis.calc(gray_b, gray_a, None)
        # Estimate reliability in source pixels, independent of GAN scale.
        sh, sw = gray_a.shape
        sx, sy = np.meshgrid(np.arange(sw, dtype=np.float32),
                             np.arange(sh, dtype=np.float32))

        def confidence(flow, reverse, first, second):
            tx, ty = sx + flow[:, :, 0], sy + flow[:, :, 1]
            inside = (tx >= 0) & (tx <= sw-1) & (ty >= 0) & (ty <= sh-1)
            inverse = cv2.remap(reverse, tx, ty, cv2.INTER_LINEAR)
            error = flow + inverse
            residual = error[:, :, 0]**2 + error[:, :, 1]**2
            tolerance = .5 + .01 * (flow[:, :, 0]**2 + inverse[:, :, 0]**2 +
                                    flow[:, :, 1]**2 + inverse[:, :, 1]**2)
            warped = cv2.remap(second, tx, ty, cv2.INTER_LINEAR)
            photo = cv2.absdiff(first, warped)
            # Compressed pixels are noisy: allow moderate photometric error,
            # but reject inconsistent vectors instead of blending ghost edges.
            reliable = inside & (residual <= tolerance) & (photo <= 30)
            return reliable.astype(np.float32)

        ca = confidence(forward, backward, gray_a, gray_b)
        cb = confidence(backward, forward, gray_b, gray_a)
        reliable_fraction = float((ca.mean() + cb.mean()) / 2)
        motion = float(np.percentile(cv2.magnitude(forward[:, :, 0], forward[:, :, 1]), 95))
        self.global_fallback = (reliable_fraction < .2 or
                                motion > .18 * np.hypot(sw, sh))
        self.diagnostics.update(reliable_fraction=reliable_fraction,
                                motion_p95_px=motion,
                                global_fallback=bool(self.global_fallback))
        h, w = enhanced_a.shape[:2]
        scale = np.array([w / gray_a.shape[1], h / gray_a.shape[0]], np.float32)
        self.forward = cv2.resize(forward, (w, h)) * scale
        self.backward = cv2.resize(backward, (w, h)) * scale
        self.confidence_a = cv2.resize(ca, (w, h), interpolation=cv2.INTER_NEAREST)
        self.confidence_b = cv2.resize(cb, (w, h), interpolation=cv2.INTER_NEAREST)
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
        if self.global_fallback:
            return (self.a if alpha < .5 else self.b), "UNRELIABLE_NEAREST"
        # Inverse-warp approximation, gated by source-space forward/backward
        # agreement. This is a conservative heuristic, not learned occlusion.
        ax = self.x - alpha * self.forward[:, :, 0]
        ay = self.y - alpha * self.forward[:, :, 1]
        bx = self.x - (1-alpha) * self.backward[:, :, 0]
        by = self.y - (1-alpha) * self.backward[:, :, 1]
        h, w = self.a.shape[:2]
        valid_a = ((ax >= 0) & (ax <= w-1) & (ay >= 0) & (ay <= h-1))
        valid_b = ((bx >= 0) & (bx <= w-1) & (by >= 0) & (by <= h-1))
        a = cv2.remap(self.a, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.remap(self.b, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ca = cv2.remap(self.confidence_a, ax, ay, cv2.INTER_NEAREST) * valid_a
        cb = cv2.remap(self.confidence_b, bx, by, cv2.INTER_NEAREST) * valid_b
        # Do not splice unwarped endpoint pixels into a warped body: that
        # creates torn contours. Reject the whole synthetic frame when a
        # substantial area has conflicting predictions or no reliable side.
        differences = cv2.split(cv2.absdiff(a, b))
        disagreement = cv2.max(cv2.max(differences[0], differences[1]), differences[2]) > 30
        valid_ca, valid_cb = ca > .5, cb > .5
        missing = ~valid_ca & ~valid_cb
        result = cv2.addWeighted(a, 1-alpha, b, alpha, 0)
        # Image boundaries already have one-sided sampling protection. Exclude
        # the motion-width border from global rejection so a clean pan does
        # not fail solely because the camera reveals a new border strip.
        margin = min(min(h, w)//4, max(2, int(np.ceil(
            self.diagnostics['motion_p95_px'] * max(w/self.flow.shape[1], h/self.flow.shape[0])))+2))
        interior = np.s_[margin:h-margin, margin:w-margin]
        unknown = float(missing[interior].mean())
        conflict = float(disagreement[interior].mean())
        self.diagnostics.update(unreliable_pixel_fraction=float(missing.mean()),
                                disagreement_fraction=float(disagreement.mean()),
                                interior_unreliable_fraction=unknown,
                                interior_disagreement_fraction=conflict)
        rejected = (unknown > .05 and conflict > .05) or conflict > .20
        self.diagnostics['frame_rejected'] = rejected
        if rejected:
            return (self.a if alpha < .5 else self.b), "UNRELIABLE_NEAREST"
        cv2.copyTo(a, (valid_a & ~valid_b).astype(np.uint8), result)
        cv2.copyTo(b, (valid_b & ~valid_a).astype(np.uint8), result)
        return result, "FLOW"

#!/usr/bin/env python3
"""Unit tests for the independent GAN full-frame path."""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from tools.full_frame_enhancer import FullFrameEnhancer, LatestOnlyEnhancerWorker


class SlowFakeEnhancer:
    backend = "fake"
    provider = "CPUExecutionProvider"

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.started = threading.Event()

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        self.started.set()
        time.sleep(self.delay)
        return np.zeros((360, 640, 3), dtype=np.uint8)


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FullFrameEnhancerTests(unittest.TestCase):
    def test_none_backend_has_exact_native_and_display_sizes(self) -> None:
        enhancer = FullFrameEnhancer("none", warmup=0)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        native = enhancer.enhance_native(frame)
        output = enhancer.enhance(frame)
        self.assertEqual(native.shape, (288, 512, 3))
        self.assertEqual(output.shape, (360, 640, 3))
        self.assertEqual(enhancer.provider, "Lanczos4")

    def test_model_postprocess_converts_rgb_to_bgr(self) -> None:
        enhancer = FullFrameEnhancer("none", warmup=0)
        rgb = np.zeros((1, 3, 288, 512), dtype=np.float32)
        rgb[0, 0, :, :] = 1.0
        rgb[0, 1, :, :] = 0.25
        converted = enhancer._postprocess_native(rgb)
        self.assertEqual(tuple(converted[0, 0]), (0, 63, 255))

    def test_latest_only_keeps_one_running_plus_latest_pending(self) -> None:
        fake = SlowFakeEnhancer(0.04)
        worker = LatestOnlyEnhancerWorker(fake, max_latency_ms=500)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 1, rtp_timestamp=1, input_fps=10))
            self.assertTrue(fake.started.wait(1.0))
            self.assertTrue(worker.submit(frame, 2, rtp_timestamp=2, input_fps=10))
            self.assertTrue(worker.submit(frame, 3, rtp_timestamp=3, input_fps=10))
            self.assertTrue(wait_for(lambda: worker.snapshot()["completed"] >= 2))
            outputs = []
            while True:
                output = worker.poll_output()
                if output is None:
                    break
                outputs.append(output.source_sequence)
            # The middle pending frame is replaced before it can run.
            self.assertEqual(outputs, [3])
            values = worker.snapshot()
            self.assertEqual(values["completed"], 2)
            self.assertGreaterEqual(values["replaced_pending"], 1)
            self.assertLessEqual(int(values["running"]) + int(values["pending"]), 2)
            self.assertIn("rolling_5s_fps", values)
            self.assertIn("rolling_10s_fps", values)
        finally:
            worker.stop()

    def test_stale_result_is_dropped_after_inference(self) -> None:
        worker = LatestOnlyEnhancerWorker(SlowFakeEnhancer(0.03), max_latency_ms=5)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 7, input_fps=10))
            self.assertTrue(wait_for(lambda: worker.snapshot()["dropped"] >= 1))
            values = worker.snapshot()
            self.assertEqual(values["completed"], 0)
            self.assertGreaterEqual(values["stale_after_infer"], 1)
            self.assertIsNone(worker.poll_output())
        finally:
            worker.stop()

    def test_snapshot_reports_age_of_an_inflight_enhancement(self) -> None:
        fake = SlowFakeEnhancer(0.08)
        worker = LatestOnlyEnhancerWorker(fake, max_latency_ms=500)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 11, input_fps=10))
            self.assertTrue(fake.started.wait(1.0))
            self.assertTrue(wait_for(lambda: worker.snapshot()["running_age_ms"] >= 5.0))
            values = worker.snapshot()
            self.assertTrue(values["running"])
            self.assertEqual(values["running_sequence"], 11)
            self.assertGreater(values["running_age_ms"], 0.0)
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()

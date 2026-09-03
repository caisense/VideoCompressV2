#!/usr/bin/env python3
"""Unit tests for the independent GAN full-frame path."""

from __future__ import annotations

import threading
import time
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.full_frame_enhancer import (
    EnhancementDiagnostics,
    FullFrameEnhancer,
    LatestOnlyEnhancerWorker,
)


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


class DiagnosticFakeEnhancer(SlowFakeEnhancer):
    def enhance_with_diagnostics(self, frame: np.ndarray):
        self.started.set()
        time.sleep(self.delay)
        return np.zeros((360, 640, 3), dtype=np.uint8), EnhancementDiagnostics(
            raw_min=-0.01,
            raw_max=1.05,
            postprocess_mode="ZERO_TO_ONE",
            clip_low_count=2,
            clip_high_count=3,
            output_luma_mean=42.0,
        )


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FullFrameEnhancerTests(unittest.TestCase):
    @staticmethod
    def _esrgan_without_loading_model() -> FullFrameEnhancer:
        enhancer = object.__new__(FullFrameEnhancer)
        enhancer.backend = "esrgan"
        enhancer.native_size = (512, 288)
        return enhancer

    def test_none_backend_has_exact_native_and_display_sizes(self) -> None:
        enhancer = FullFrameEnhancer("none", warmup=0)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        native = enhancer.enhance_native(frame)
        output = enhancer.enhance(frame)
        self.assertEqual(native.shape, (288, 512, 3))
        self.assertEqual(output.shape, (360, 640, 3))
        self.assertEqual(enhancer.provider, "Lanczos4")

    def test_model_postprocess_converts_rgb_to_bgr(self) -> None:
        enhancer = self._esrgan_without_loading_model()
        rgb = np.zeros((1, 3, 288, 512), dtype=np.float32)
        rgb[0, 0, :, :] = 1.0
        rgb[0, 1, :, :] = 0.25
        converted = enhancer._postprocess_native(rgb)
        self.assertEqual(tuple(converted[0, 0]), (0, 63, 255))

    def test_esrgan_negative_overshoot_is_clipped_without_gray_lift(self) -> None:
        enhancer = self._esrgan_without_loading_model()
        rgb = np.full((1, 3, 288, 512), 0.2, dtype=np.float32)
        rgb[0, 0, 0, 0] = -0.01
        rgb[0, 1, 0, 1] = 1.05
        converted, diagnostics = enhancer._postprocess_native_with_diagnostics(
            rgb, collect_diagnostics=True
        )
        self.assertEqual(diagnostics.postprocess_mode, "ZERO_TO_ONE")
        self.assertAlmostEqual(diagnostics.raw_min, -0.01, places=5)
        self.assertAlmostEqual(diagnostics.raw_max, 1.05, places=5)
        self.assertEqual(diagnostics.clip_low_count, 1)
        self.assertEqual(diagnostics.clip_high_count, 1)
        self.assertEqual(int(converted[0, 0, 2]), 0)
        self.assertEqual(int(converted[10, 10, 0]), 51)
        self.assertLess(float(converted.mean()), 52.0)

    def test_esrgan_1p1_boundary_has_no_scale_switch(self) -> None:
        enhancer = self._esrgan_without_loading_model()
        first = np.full((1, 3, 288, 512), 0.2, dtype=np.float32)
        second = first.copy()
        first[0, 0, 0, 0] = 1.0999
        second[0, 0, 0, 0] = 1.1001
        output_a, stats_a = enhancer._postprocess_native_with_diagnostics(
            first, collect_diagnostics=True
        )
        output_b, stats_b = enhancer._postprocess_native_with_diagnostics(
            second, collect_diagnostics=True
        )
        self.assertEqual(stats_a.postprocess_mode, "ZERO_TO_ONE")
        self.assertEqual(stats_b.postprocess_mode, "ZERO_TO_ONE")
        self.assertEqual(int(output_a[10, 10, 0]), int(output_b[10, 10, 0]))
        self.assertLess(int(output_b[0, 0, 0]), 256)

    def test_esrgan_positive_overshoot_is_clipped(self) -> None:
        enhancer = self._esrgan_without_loading_model()
        rgb = np.full((1, 3, 288, 512), 0.2, dtype=np.float32)
        rgb[0, 0, 0, 0] = 1.2
        converted, diagnostics = enhancer._postprocess_native_with_diagnostics(
            rgb, collect_diagnostics=True
        )
        self.assertEqual(diagnostics.postprocess_mode, "ZERO_TO_ONE")
        self.assertEqual(diagnostics.clip_high_count, 1)
        self.assertEqual(int(converted[0, 0, 2]), 255)
        self.assertEqual(int(converted[10, 10, 0]), 51)

    def test_esrgan_nan_and_inf_are_rejected(self) -> None:
        enhancer = self._esrgan_without_loading_model()
        for bad_value in (np.nan, np.inf, -np.inf):
            rgb = np.zeros((1, 3, 288, 512), dtype=np.float32)
            rgb[0, 0, 0, 0] = bad_value
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                enhancer._postprocess_native(rgb)

    def test_debug_record_keeps_pixel_stats_with_same_sequence(self) -> None:
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "gan.jsonl"
            worker = LatestOnlyEnhancerWorker(
                DiagnosticFakeEnhancer(0.001), max_latency_ms=500, debug_log=log
            )
            try:
                self.assertTrue(worker.submit(frame, 123, input_fps=8.0))
                self.assertTrue(wait_for(lambda: worker.snapshot()["completed"] >= 1))
            finally:
                worker.stop()
            records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["SEQ"], 123)
        self.assertEqual(records[0]["POSTPROCESS_MODE"], "ZERO_TO_ONE")
        self.assertEqual(records[0]["RAW_MIN"], -0.01)
        self.assertEqual(records[0]["CLIP_HIGH_COUNT"], 3)

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

    def test_auto_budget_accepts_eight_fps_one_hundred_ten_ms_total(self) -> None:
        worker = LatestOnlyEnhancerWorker(SlowFakeEnhancer(0.11), max_latency_ms=143.75)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 21, input_fps=8.0))
            self.assertTrue(wait_for(lambda: worker.snapshot()["inference_completed"] >= 1))
            values = worker.snapshot()
            self.assertEqual(values["completed"], 1)
            self.assertEqual(values["stale_after_infer"], 0)
            self.assertGreaterEqual(values["infer_all_p50_ms"], 100.0)
            self.assertGreaterEqual(values["good_total_p50_ms"], 100.0)
            self.assertAlmostEqual(values["frame_period_ms"], 125.0, delta=0.1)
        finally:
            worker.stop()

    def test_auto_budget_accepts_ten_fps_ninety_ms_total(self) -> None:
        worker = LatestOnlyEnhancerWorker(SlowFakeEnhancer(0.09), max_latency_ms=115.0)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 31, input_fps=10.0))
            self.assertTrue(wait_for(lambda: worker.snapshot()["inference_completed"] >= 1))
            values = worker.snapshot()
            self.assertEqual(values["completed"], 1)
            self.assertEqual(values["stale_after_infer"], 0)
            self.assertAlmostEqual(values["frame_period_ms"], 100.0, delta=0.1)
        finally:
            worker.stop()

    def test_ten_fps_slow_input_replaces_pending_without_queue_growth(self) -> None:
        fake = SlowFakeEnhancer(0.13)
        worker = LatestOnlyEnhancerWorker(fake, max_latency_ms=115.0)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 41, input_fps=10.0))
            self.assertTrue(fake.started.wait(1.0))
            self.assertTrue(worker.submit(frame, 42, input_fps=10.0))
            self.assertTrue(worker.submit(frame, 43, input_fps=10.0))
            self.assertTrue(wait_for(lambda: worker.snapshot()["inference_completed"] >= 1))
            values = worker.snapshot()
            self.assertGreaterEqual(values["replaced_pending"], 1)
            self.assertLessEqual(int(values["running"]) + int(values["pending"]), 2)
            self.assertGreaterEqual(values["stale_after_infer"], 1)
        finally:
            worker.stop()

    def test_twelve_fps_throughput_limit_stays_bounded_and_not_hard_stall(self) -> None:
        fake = SlowFakeEnhancer(0.095)
        worker = LatestOnlyEnhancerWorker(
            fake, max_latency_ms=(1.15 * 1000.0 / 12.0)
        )
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 51, input_fps=12.0))
            self.assertTrue(fake.started.wait(1.0))
            self.assertTrue(worker.submit(frame, 52, input_fps=12.0))
            self.assertTrue(worker.submit(frame, 53, input_fps=12.0))
            self.assertTrue(wait_for(lambda: worker.snapshot()["inference_completed"] >= 1))
            values = worker.snapshot()
            self.assertGreaterEqual(values["replaced_pending"], 1)
            self.assertLessEqual(int(values["running"]) + int(values["pending"]), 2)
            self.assertLess(values["running_age_ms"], 500.0)
        finally:
            worker.stop()

    def test_all_returned_inference_stats_include_stale_after_infer(self) -> None:
        worker = LatestOnlyEnhancerWorker(SlowFakeEnhancer(0.03), max_latency_ms=5)
        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(frame, 22, input_fps=8.0))
            self.assertTrue(wait_for(lambda: worker.snapshot()["inference_completed"] >= 1))
            values = worker.snapshot()
            self.assertEqual(values["completed"], 0)
            self.assertEqual(values["inference_completed"], 1)
            self.assertGreater(values["infer_all_max_ms"], 0.0)
            self.assertGreater(values["all_total_max_ms"], 0.0)
            self.assertGreater(values["last_finished_total_ms"], 0.0)
            self.assertFalse(values["last_finished_accepted"])
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()

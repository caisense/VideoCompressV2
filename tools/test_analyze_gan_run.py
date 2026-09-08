"""Tests for the GAN JSONL summary, including transition/event records."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools import analyze_gan_run


class AnalyzeGanRunTests(unittest.TestCase):
    def test_summary_counts_all_inference_and_event_drop_reasons(self) -> None:
        records = [
            {
                "TYPE": "GAN_WORKER", "SEQ": 1, "ARRIVAL": 100.0,
                "INPUT_FPS": 8.0, "EFFECTIVE_BUDGET_MS": 143.75,
                "INFER_MS": 90.0, "TOTAL_PC_MS": 100.0,
                "INFER_RETURNED": True, "OUTPUT_READY": True, "DROPPED": False,
                "RAW_MIN": -0.01, "RAW_MAX": 1.05,
                "POSTPROCESS_MODE": "ZERO_TO_ONE",
                "CLIP_LOW_COUNT": 3, "CLIP_HIGH_COUNT": 4,
                "OUTPUT_LUMA_MEAN": 101.0,
            },
            {
                "TYPE": "GAN_WORKER", "SEQ": 2, "ARRIVAL": 101.0,
                "INPUT_FPS": 8.0, "EFFECTIVE_BUDGET_MS": 143.75,
                "INFER_MS": 120.0, "TOTAL_PC_MS": 160.0,
                "INFER_RETURNED": True, "OUTPUT_READY": False, "DROPPED": True,
                "DROP_REASON": "stale_after_infer",
                "RAW_MIN": -0.02, "RAW_MAX": 1.20,
                "POSTPROCESS_MODE": "ZERO_TO_ONE",
                "CLIP_LOW_COUNT": 5, "CLIP_HIGH_COUNT": 6,
                "OUTPUT_LUMA_MEAN": 102.0,
            },
            {
                "TYPE": "GAN_WORKER", "SEQ": 3, "ARRIVAL": 101.1,
                "INPUT_FPS": 8.0, "EFFECTIVE_BUDGET_MS": 143.75,
                "INFER_MS": 0.0, "TOTAL_PC_MS": 0.0,
                "PREDICTED_INFER_MS": 90.0, "PREDICTED_TOTAL_MS": 160.0,
                "INFER_RETURNED": False, "OUTPUT_READY": False, "DROPPED": True,
                "DROP_REASON": "stale_before_predicted",
            },
            {"TYPE": "GAN_WORKER_DROP", "SEQ": 1, "DROP_REASON": "output_replaced"},
            {"TYPE": "GAN_STATE", "STATE": "SOFT_FALLBACK", "AT": 101.0},
            {"TYPE": "GAN_STATE", "STATE": "RECOVERED", "AT": 101.5,
             "FALLBACK_DURATION_MS": 500.0, "RECOVERY_STREAK": 3},
            {"TYPE": "GAN_PRESENTATION_DROP", "REASON": "OUTPUT_TOO_OLD", "SEQ": 2},
            {"TYPE": "GAN_PRESENTATION_DROP", "REASON": "RECOVERY_NOT_FRESH", "SEQ": 2},
            {"TYPE": "GAN_PRESENTATION_DROP", "REASON": "PRESENTATION_SEQUENCE_OLD", "SEQ": 2},
            {"TYPE": "GAN_PRESENTATION_DROP", "REASON": "RECOVERY_TOO_OLD", "SEQ": 2},
            {"TYPE": "GAN_RECOVERY_RESET", "REASON": "WORKER_STALE_AFTER_INFER"},
            {"TYPE": "GAN_HARD_STALL", "SEQ": 3, "RUN_AGE_MS": 700.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gan.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            old_argv = sys.argv
            output = io.StringIO()
            try:
                sys.argv = [
                    "analyze_gan_run.py", str(path),
                    "--latency-budget-ms", "143.75",
                ]
                with contextlib.redirect_stdout(output):
                    self.assertEqual(analyze_gan_run.main(), 0)
            finally:
                sys.argv = old_argv
        result = json.loads(output.getvalue())
        self.assertEqual(result["source_fps"], 8.0)
        self.assertEqual(result["inference_completed"], 2)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["worker_dropped"], 3)
        self.assertEqual(result["presentation_dropped"], 4)
        self.assertEqual(result["drop_reason_counts"]["stale_after"], 1)
        self.assertEqual(result["drop_reason_counts"]["stale_before_predicted"], 1)
        self.assertEqual(result["drop_reason_counts"]["output_replaced"], 1)
        self.assertEqual(result["drop_reason_counts"]["presentation_old"], 1)
        self.assertEqual(result["drop_reason_counts"]["recovery_not_fresh"], 1)
        self.assertEqual(result["drop_reason_counts"]["presentation_sequence_old"], 1)
        self.assertEqual(result["drop_reason_counts"]["recovery_too_old"], 1)
        self.assertEqual(result["infer_all_ms"]["max"], 120.0)
        self.assertEqual(result["all_total_ms"]["max"], 160.0)
        self.assertEqual(result["fallback_longest_duration_ms"], 500.0)
        self.assertEqual(result["hard_stall"]["count"], 1)
        self.assertEqual(result["recovery_stale_rejects"], 1)
        self.assertEqual(result["recovery_rejection_reasons"], {
            "RECOVERY_NOT_FRESH": 1,
            "RECOVERY_TOO_OLD": 1,
        })
        self.assertEqual(result["recovery_worker_resets"], 1)
        self.assertEqual(result["prediction"]["stale_before_predicted"], 1)
        self.assertEqual(result["prediction"]["predicted_infer_ms"]["max"], 90.0)
        self.assertEqual(result["prediction"]["predicted_total_ms"]["max"], 160.0)
        self.assertEqual(result["postprocess_mode_counts"], {"ZERO_TO_ONE": 2})
        self.assertEqual(result["raw_output_range"], {"min": -0.02, "max": 1.2})
        self.assertEqual(result["clipped_low_total"], 8)
        self.assertEqual(result["clipped_high_total"], 10)
        self.assertEqual(result["output_luma"]["max"], 102.0)


if __name__ == "__main__":
    unittest.main()

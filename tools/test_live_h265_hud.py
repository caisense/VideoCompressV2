#!/usr/bin/env python3

import importlib.util
import io
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("live_h265_hud.py")
SPEC = importlib.util.spec_from_file_location("live_h265_hud", MODULE_PATH)
HUD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HUD)


def rtp_packet(sequence: int, marker: bool = False, nal_type: int = 1) -> bytes:
    packet = bytearray(14)
    packet[0] = 0x80
    packet[1] = 96 | (0x80 if marker else 0)
    packet[2:4] = sequence.to_bytes(2, "big")
    packet[12] = nal_type << 1
    packet[13] = 1
    return bytes(packet)


def rtp_profile_packet(sequence: int, profile: int, width: int, height: int,
                       fps: int, generation: int, nal_type: int = 19) -> bytes:
    packet = bytearray(rtp_packet(sequence, marker=True, nal_type=nal_type))
    packet[0] |= 0x10
    extension = bytearray(b"RO\x00\x02")
    extension.extend((1, profile))
    extension.extend(width.to_bytes(2, "big"))
    extension.extend(height.to_bytes(2, "big"))
    extension.extend((fps, generation))
    return bytes(packet[:12] + extension + packet[12:])


class FixedDelayEnhancer:
    backend = "fake"
    provider = "CUDAExecutionProvider"

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def enhance(self, frame):
        time.sleep(self.delay)
        return HUD.np.zeros((360, 640, 3), dtype=HUD.np.uint8)


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def gan_candidate(sequence: int, completed_at: float, total_pc_ms: float = 90.0):
    return SimpleNamespace(
        source_sequence=sequence,
        completed_at=completed_at,
        total_pc_ms=total_pc_ms,
    )


def idle_worker_snapshot(**overrides):
    values = {
        "running": False,
        "running_age_ms": 0.0,
        "running_sequence": None,
        "last_finished_sequence": None,
        "last_finished_infer_ms": 0.0,
        "drop_reason_counts": {},
    }
    values.update(overrides)
    return values


class HudTests(unittest.TestCase):
    def test_rtp_metrics_detect_loss_marker_and_idr(self):
        stats = HUD.RtpStats()
        stats.on_packet(rtp_packet(10, nal_type=19))
        stats.on_packet(rtp_packet(12, marker=True))
        values = stats.snapshot()
        self.assertEqual(values["lost"], 1)
        self.assertEqual(values["packets"], 2)
        self.assertGreater(values["rtp_kbps"], 0)
        self.assertEqual(values["pps"], 2.0)
        self.assertEqual(values["packet_last_bytes"], 14)
        self.assertEqual(values["packet_avg_bytes"], 14.0)
        self.assertEqual(values["packet_max_bytes"], 14)

    def test_rtp_metrics_classify_i_and_p_access_units(self):
        stats = HUD.RtpStats()
        stats.on_packet(rtp_packet(20, marker=True, nal_type=19))
        stats.on_packet(rtp_packet(21, marker=True, nal_type=1))
        values = stats.snapshot()
        self.assertEqual(values["i_frames"], 1)
        self.assertEqual(values["p_frames"], 1)
        self.assertEqual(values["i_fps"], 1.0)
        self.assertEqual(values["p_fps"], 1.0)

    def test_runtime_profile_extension_and_shifted_payload(self):
        stats = HUD.RtpStats()
        self.assertFalse(stats.on_packet(rtp_profile_packet(30, 1, 480, 270, 15, 3)))
        values = stats.snapshot()
        self.assertEqual(values["profile"], {
            "name": "medium", "width": 480, "height": 270,
            "fps": 15, "generation": 3,
        })
        self.assertEqual(values["i_frames"], 1)
        self.assertTrue(stats.on_packet(rtp_profile_packet(31, 0, 320, 180, 10, 4)))
        self.assertEqual(stats.snapshot()["profile"]["name"], "low")

    def test_rebuild_profile_and_presentation_provenance(self):
        stats = HUD.RtpStats()
        stats.on_packet(rtp_profile_packet(32, 3, 256, 144, 6, 5))
        self.assertEqual(stats.snapshot()["profile"]["name"], "rebuild")
        presentation = HUD.PresentationStats()
        self.assertEqual(presentation.present(10, "ROI-LANCZOS", 5, now=1.0),
                         "DECODED+ROI-LANCZOS")
        self.assertEqual(presentation.present(10, "ROI-LANCZOS", 5, now=1.08),
                         "HOLD+ROI-LANCZOS")
        self.assertEqual(presentation.snapshot()["held_percent"], 50.0)
        self.assertEqual(HUD.display_dimensions(640, 360, "ccw90", 1), (360, 640))

    def test_gan_profile_extension_is_distinct_from_rebuild(self):
        stats = HUD.RtpStats()
        stats.on_packet(rtp_profile_packet(33, 4, 256, 144, 8, 6))
        profile = stats.snapshot()["profile"]
        self.assertEqual(profile["name"], "gan")
        self.assertEqual(profile["width"], 256)
        self.assertEqual(profile["height"], 144)
        self.assertEqual(profile["fps"], 8)

    def test_decoded_pts_handoff_preserves_fifo_order_with_bounded_backlog(self):
        queue = HUD.DecodedPtsQueue()
        queue.push_rtp_timestamp(90000)
        queue.push_rtp_timestamp(99000)
        self.assertEqual(queue.pop(), 90000)
        self.assertEqual(queue.pop(), 99000)
        self.assertIsNone(queue.pop())
        bounded = HUD.DecodedPtsQueue(max_pending=2)
        bounded.push_rtp_timestamp(1)
        bounded.push_rtp_timestamp(2)
        bounded.push_rtp_timestamp(3)
        self.assertEqual(bounded.pop(), 2)
        self.assertEqual(bounded.snapshot()["dropped"], 1)

    def test_profile_switch_gate_replays_complete_idr(self):
        gate = HUD.ProfileSwitchGate()
        high = {"generation": 2}
        low = {"generation": 3}
        high_idr = rtp_profile_packet(40, 2, 640, 360, 20, 2)
        generation, packets = gate.feed(high_idr, high)
        self.assertEqual(generation, 2)
        self.assertEqual(packets, [high_idr])
        low_p = rtp_profile_packet(41, 0, 320, 180, 10, 3, nal_type=1)
        generation, packets = gate.feed(low_p, low)
        self.assertIsNone(generation)
        self.assertEqual(packets, [])
        low_idr = rtp_profile_packet(42, 0, 320, 180, 10, 3)
        generation, packets = gate.feed(low_idr, low)
        self.assertEqual(generation, 3)
        self.assertEqual(packets, [low_idr])

    def test_profile_switch_gate_supports_legacy_stream(self):
        gate = HUD.ProfileSwitchGate()
        self.assertEqual(gate.feed(rtp_packet(50, marker=True, nal_type=1), None),
                         (None, []))
        idr = rtp_packet(51, marker=True, nal_type=19)
        self.assertEqual(gate.feed(idr, None), (-1, [idr]))

    def test_depacketizer_single_nal_with_profile_extension(self):
        packet = rtp_profile_packet(100, 2, 640, 360, 20, 7, nal_type=32)
        payload_offset = HUD.RtpStats._payload_offset(packet)
        self.assertIsNotNone(payload_offset)
        self.assertEqual(HUD.HevcRtpDepacketizer().feed(packet),
                         b"\x00\x00\x00\x01" + packet[payload_offset:])

    def test_depacketizer_fragmented_nal(self):
        def fragment(sequence, start, end, body):
            packet = bytearray(rtp_packet(sequence))
            packet[12:] = bytes((49 << 1, 1,
                                 (0x80 if start else 0) | (0x40 if end else 0) | 19)) + body
            return bytes(packet)

        depacketizer = HUD.HevcRtpDepacketizer()
        self.assertEqual(depacketizer.feed(fragment(200, True, False, b"abc")),
                         b"\x00\x00\x00\x01" + bytes((19 << 1, 1)) + b"abc")
        self.assertEqual(depacketizer.feed(fragment(201, False, True, b"def")), b"def")

    def test_bmp_pipe_frame_is_self_delimiting(self):
        image = HUD.np.zeros((7, 11, 3), dtype=HUD.np.uint8)
        image[3, 5] = (10, 20, 30)
        encoded, payload = HUD.cv2.imencode(".bmp", image)
        self.assertTrue(encoded)
        decoded = HUD.read_bmp_frame(io.BytesIO(payload.tobytes()))
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, image.shape)
        self.assertTrue((decoded == image).all())

    def test_display_rotation_and_denoise(self):
        image = HUD.np.zeros((4, 7, 3), dtype=HUD.np.uint8)
        image[0, 0] = (255, 0, 0)
        rotated = HUD.postprocess_frame(image, "ccw90", False)
        self.assertEqual(rotated.shape, (7, 4, 3))
        self.assertEqual(tuple(rotated[6, 0]), (255, 0, 0))

    def test_gan_fallback_bounds_stalled_enhancer_and_preserves_output_size(self):
        frame = HUD.np.zeros((144, 256, 3), dtype=HUD.np.uint8)
        self.assertEqual(HUD.gan_fallback_stale_after_ms(10, 100), 200.0)
        self.assertTrue(HUD.should_use_gan_fallback(
            frame, 1, None, 0.0, False, 10.0, 10.0, 100.0))
        self.assertFalse(HUD.should_use_gan_fallback(
            frame, 1, 1, 10.0, False, 10.19, 10.0, 100.0))
        self.assertTrue(HUD.should_use_gan_fallback(
            frame, 1, 1, 10.0, False, 10.21, 10.0, 100.0))
        self.assertTrue(HUD.should_use_gan_fallback(
            frame, 2, 1, 10.01, True, 10.02, 10.0, 100.0))
        self.assertEqual(HUD.gan_lanczos_fallback(frame, "none").shape, (360, 640, 3))

    def test_gan_auto_budget_is_fps_aware(self):
        self.assertAlmostEqual(HUD.gan_effective_output_budget_ms(8), 143.75)
        self.assertAlmostEqual(HUD.gan_effective_output_budget_ms(10), 115.0)
        self.assertAlmostEqual(
            HUD.gan_effective_output_budget_ms(12), 1.15 * 1000.0 / 12.0
        )
        self.assertEqual(
            HUD.gan_effective_output_budget_ms(8, legacy_budget_ms=100), 100.0
        )

    def test_eight_fps_ninety_ms_stays_healthy_for_twenty_frames(self):
        budget = HUD.gan_effective_output_budget_ms(8.0)
        worker = HUD.LatestOnlyEnhancerWorker(
            FixedDelayEnhancer(0.09), max_latency_ms=budget
        )
        controller = HUD.GanFallbackController(budget)
        frame = HUD.np.zeros((144, 256, 3), dtype=HUD.np.uint8)
        try:
            for sequence in range(1, 21):
                arrival = time.monotonic()
                self.assertTrue(worker.submit(
                    frame, sequence, arrived_at=arrival, input_fps=8.0
                ))
                deadline = arrival + 0.125
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    candidate = worker.poll_output()
                    decision = controller.update(
                        now, 8.0, sequence, worker.snapshot(), candidate
                    )
                    self.assertEqual(decision.state, HUD.GanFallbackController.HEALTHY)
                    time.sleep(0.003)
            self.assertTrue(wait_for(lambda: worker.snapshot()["completed"] >= 19))
            self.assertEqual(controller.soft_fallback_entries, 0)
            self.assertEqual(controller.hard_stall_count, 0)
            self.assertEqual(controller.state, HUD.GanFallbackController.HEALTHY)
        finally:
            worker.stop()

    def test_soft_overdue_reports_good_age_not_run_age(self):
        controller = HUD.GanFallbackController(143.75)
        controller.update(
            0.0, 8.0, 1, idle_worker_snapshot(), gan_candidate(1, 0.0)
        )
        decision = controller.update(
            0.251, 8.0, 2,
            idle_worker_snapshot(running=True, running_age_ms=80.0, running_sequence=2),
        )
        self.assertEqual(decision.state, HUD.GanFallbackController.SOFT_FALLBACK)
        self.assertEqual(decision.action, HUD.GanAction.USE_LANCZOS)
        event = next(item for item in decision.events if item["TYPE"] == "GAN_STATE")
        self.assertGreaterEqual(event["GOOD_AGE_MS"], 250.0)
        self.assertEqual(event["RUN_AGE_MS"], 80.0)
        self.assertEqual(controller.hard_stall_count, 0)

    def test_hard_stall_is_reported_once_and_return_is_not_immediate_recovery(self):
        controller = HUD.GanFallbackController(143.75, hard_stall_ms=500)
        controller.update(
            0.0, 8.0, 1, idle_worker_snapshot(), gan_candidate(1, 0.0)
        )
        first = controller.update(
            0.5, 8.0, 2,
            idle_worker_snapshot(running=True, running_age_ms=520.0, running_sequence=2),
        )
        self.assertEqual(first.state, HUD.GanFallbackController.HARD_STALLED)
        self.assertEqual(controller.hard_stall_count, 1)
        second = controller.update(
            0.7, 8.0, 2,
            idle_worker_snapshot(running=True, running_age_ms=700.0, running_sequence=2),
        )
        self.assertEqual(controller.hard_stall_count, 1)
        returned = controller.update(
            0.8, 8.0, 2,
            idle_worker_snapshot(
                last_finished_sequence=2, last_finished_infer_ms=700.0
            ),
            gan_candidate(2, 0.7, total_pc_ms=700.0),
        )
        self.assertEqual(returned.action, HUD.GanAction.DISCARD_STALE_GAN)
        self.assertEqual(returned.state, HUD.GanFallbackController.HARD_STALLED)
        self.assertTrue(any(item["TYPE"] == "GAN_HARD_STALL_RETURNED"
                            for item in returned.events))

    def test_recovery_requires_three_fresh_outputs_and_rejects_stale_sequence(self):
        controller = HUD.GanFallbackController(143.75, recovery_good_frames=3)
        controller.update(
            0.0, 8.0, 1, idle_worker_snapshot(), gan_candidate(1, 0.0)
        )
        controller.update(0.251, 8.0, 2, idle_worker_snapshot())
        controller.note_lanczos_display(2)
        stale = controller.update(
            0.3, 8.0, 100, idle_worker_snapshot(),
            gan_candidate(95, 0.29),
        )
        self.assertEqual(stale.action, HUD.GanAction.DISCARD_STALE_GAN)
        self.assertEqual(controller.recovery_streak, 0)
        one = controller.update(
            0.4, 8.0, 3, idle_worker_snapshot(), gan_candidate(3, 0.39)
        )
        two = controller.update(
            0.5, 8.0, 4, idle_worker_snapshot(), gan_candidate(4, 0.49)
        )
        three = controller.update(
            0.6, 8.0, 5, idle_worker_snapshot(), gan_candidate(5, 0.59)
        )
        self.assertEqual(one.action, HUD.GanAction.KEEP_GAN)
        self.assertEqual(two.action, HUD.GanAction.KEEP_GAN)
        self.assertEqual(three.action, HUD.GanAction.ACCEPT_GAN)
        self.assertEqual(three.state, HUD.GanFallbackController.HEALTHY)
        self.assertEqual(controller.recoveries, 1)

    def test_same_source_frame_shown_by_lanczos_cannot_be_replaced_by_gan(self):
        controller = HUD.GanFallbackController(143.75)
        controller.note_lanczos_display(100)
        old = controller.update(
            1.0, 8.0, 100, idle_worker_snapshot(), gan_candidate(100, 0.9)
        )
        self.assertEqual(old.action, HUD.GanAction.DISCARD_STALE_GAN)
        new = controller.update(
            1.1, 8.0, 101, idle_worker_snapshot(), gan_candidate(101, 1.0)
        )
        self.assertEqual(new.action, HUD.GanAction.ACCEPT_GAN)

    def test_window_dimensions_follow_rotation(self):
        self.assertEqual(HUD.display_dimensions(320, 180, "ccw90", 3), (540, 960))
        self.assertEqual(HUD.display_dimensions(320, 180, "none", 3), (960, 540))


if __name__ == "__main__":
    unittest.main()

"""Synthetic audit-policy regression tests; these are not hardware evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_video_quality_sweep import candidates
from analyze_video_quality_sweep import summarize
from verify_video_quality_run import audit, capture_conversion_stats


class CaptureConversionTelemetryTests(unittest.TestCase):
    def test_old_capture_does_not_claim_zero_recovery(self):
        result = capture_conversion_stats([dict(PTS=100)], 0, 200, 10)
        self.assertFalse(result['capture_conversion_reported'])
        self.assertIsNone(result['capture_cpu_recovery_frames'])

    def test_recovery_counts_window_and_keeps_backend_semantics_separate(self):
        rows = [dict(PTS=i*100000, CONVERSION_BACKEND=(
            'OPENCV_RGB_NV12' if i in (0, 2, 3) else 'RGA3_CORE0_BOUNDED'),
            CONVERSION_RECOVERY_REASON=2 if i == 2 else 1, CONVERSION_WAIT_US=20000)
            for i in range(6)]
        result = capture_conversion_stats(rows, 100000, 500000, 10)
        self.assertEqual(result['capture_cpu_recovery_frames'], 2)
        self.assertEqual(result['capture_cpu_recovery_max_run_ms'], 200)
        self.assertEqual(result['capture_cpu_recovery_reason_counts'], {'2': 1, '1': 1})


class FakeCapture:
    def __init__(self, count=700):
        self.remaining = count

    def read(self):
        self.remaining -= 1
        return self.remaining >= 0, None

    def release(self):
        pass


class HardwareAuditPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        config = dict(candidates('B')[0], cap=120, fps=10, duration_s=60,
                      warmup_s=10, status='RECORDED', strict_same_source=True)
        self.write('config.json', config)
        self.packets, self.presentations, workers, encoded, tx = [], [], [], [], []
        for index in range(700):
            pts = 10_000_000 + index * 100_000
            rtp = pts * 90 // 1000
            self.packets.append(dict(at=200 + index * .1, sequence=index, marker=True,
                                     rtp_timestamp=rtp, wire_bytes=100, udp_bytes=34,
                                     rtp_payload_bytes=6))
            self.presentations.append(dict(TYPE='GAN_PRESENT', PTS=rtp, SEQ=index,
                AT=200.05 + index * .1, PROVENANCE='DECODED+GAN-ESRGAN'))
            workers.append(dict(TYPE='GAN_WORKER', PTS=rtp, SEQ=index,
                ARRIVAL=200 + index * .1, OUTPUT_READY=True, INFER_RETURNED=True, INFER_MS=10))
            encoded.append(dict(TYPE='ENC_FRAME', PTS=pts, SEQ=index, ENCODED_BITS=48))
            tx.append(f'QUALITY_TX pts_us={pts} wire_bytes={index * 100} '
                      f'rtp_payload_bytes={index * 6} queue=0 oldest_queue_age_us=0 '
                      'drop_p=0 drop_idr=0')
        self.write_lines('packets.jsonl', self.packets)
        self.write_lines('enhancer.jsonl', self.presentations + workers)
        self.write_lines('encoder.jsonl', encoded)
        self.write_lines('memory.jsonl', [dict(at=200 + index * 5, process_tree_rss_mib=1000)
                                         for index in range(15)])
        self.write('clock_sync.json', dict(
            before=dict(pc_at=199, pc_minus_board_s=190, uncertainty_s=.001),
            after=dict(pc_at=271, pc_minus_board_s=190, uncertainty_s=.001)))
        (self.directory / 'sender.log').write_text('\n'.join(tx))
        (self.directory / 'receiver.log').write_text(
            'decode_fps=10 lost=0 reorder=0 decode_errors=0')
        (self.directory / 'received.h265').write_bytes(b'synthetic policy test only')

    def write(self, name, record):
        (self.directory / name).write_text(json.dumps(record))

    def write_lines(self, name, records):
        (self.directory / name).write_text('\n'.join(json.dumps(item) for item in records))

    def run_audit(self, decoded=700):
        with mock.patch('verify_video_quality_run.cv2.VideoCapture',
                        return_value=FakeCapture(decoded)), mock.patch(
                        'verify_video_quality_run.subprocess.run',
                        return_value=mock.Mock(returncode=0, stderr='')):
            return audit(self.directory, {'synthetic_policy_test': True})

    def test_complete_ordered_screening_evidence(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['acceptance']['duration_scope'], 'screening only')

    def test_modified_capture_invalidates_prior_acceptance(self):
        self.assertEqual(self.run_audit()['status'], 'PASS')
        (self.directory / 'received.h265').write_bytes(b'changed after audit')
        result = summarize(self.directory)
        self.assertFalse(result['evidence_hashes_verified'])
        self.assertEqual(result['status'], 'FAIL')

    def test_unchanged_decode_evidence_can_be_reused_for_new_policy(self):
        self.assertEqual(self.run_audit()['status'], 'PASS')
        with mock.patch('verify_video_quality_run.cv2.VideoCapture', side_effect=AssertionError), \
                mock.patch('verify_video_quality_run.subprocess.run', side_effect=AssertionError):
            result = audit(self.directory, {'synthetic_policy_test': True})
        self.assertTrue(result['acceptance']['diagnostics']['decoder_evidence_reused_by_sha256'])
        self.assertEqual(result['status'], 'PASS')

    def test_changed_capture_requires_fresh_decode(self):
        self.run_audit()
        (self.directory / 'received.h265').write_bytes(b'different capture')
        with mock.patch('verify_video_quality_run.cv2.VideoCapture', return_value=FakeCapture()), \
                mock.patch('verify_video_quality_run.subprocess.run', return_value=mock.Mock(
                    returncode=1, stderr='decode error')) as decode:
            result = audit(self.directory, {'synthetic_policy_test': True})
        decode.assert_called_once()
        self.assertFalse(result['acceptance']['diagnostics']['decoder_evidence_reused_by_sha256'])
        self.assertEqual(result['status'], 'FAIL')

    def test_gui_latency_uses_window_submission_not_schedule_time(self):
        config = json.loads((self.directory / 'config.json').read_text())
        config['display'] = True
        self.write('config.json', config)
        records = [json.loads(line) for line in
                   (self.directory / 'enhancer.jsonl').read_text().splitlines()]
        submissions = []
        for item in records:
            if item['TYPE'] == 'GAN_PRESENT':
                item['HEADLESS'] = False
                submissions.append(dict(item, TYPE='GAN_DISPLAY_SUBMIT', AT=item['AT'] + .07))
        self.write_lines('enhancer.jsonl', records + submissions)
        result = self.run_audit()
        self.assertTrue(result['acceptance']['actual_windowed_presentation'])
        self.assertAlmostEqual(result['acceptance']['end_to_end_ms']['p50'], 120)
        self.assertEqual(result['status'], 'PASS')

    def test_gui_without_submission_timestamps_is_not_accepted(self):
        for item in self.presentations:
            item['HEADLESS'] = False
        self.write_lines('enhancer.jsonl', self.presentations)
        with self.assertRaisesRegex(ValueError, 'post-imshow'):
            self.run_audit()

    def test_live_camera_requires_driver_timestamp_evidence(self):
        config = json.loads((self.directory / 'config.json').read_text())
        config['input_video'] = None
        self.write('config.json', config)
        result = self.run_audit()
        self.assertFalse(result['acceptance']['source_age_bounded'])
        self.assertEqual(result['status'], 'FAIL')
        frames = [json.loads(line) for line in
                  (self.directory / 'encoder.jsonl').read_text().splitlines()]
        for frame in frames:
            frame.update(CAPTURE_SOURCE='V4L2_MONOTONIC', DRIVER_CAPTURE_US=frame['PTS'],
                         CAPTURE_READ_US=frame['PTS'] + 30000)
        self.write_lines('encoder.jsonl', frames)
        result = self.run_audit()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['acceptance']['diagnostics']['live_camera_driver_timestamp_verified'])
        self.assertAlmostEqual(result['acceptance']['diagnostics']['driver_queue_age_ms']['p50'], 30)

    def test_packet_gap_and_decode_mismatch_reject_ordering(self):
        self.packets[200]['sequence'] += 1
        self.write_lines('packets.jsonl', self.packets)
        result = self.run_audit(decoded=699)
        self.assertFalse(result['acceptance']['ordering_verified'])
        self.assertEqual(result['status'], 'FAIL')

    def test_hold_and_fallback_never_become_successful_gan(self):
        for item in self.presentations:
            item['PROVENANCE'] = 'HOLD+GAN-FALLBACK-LANCZOS'
        self.write_lines('enhancer.jsonl', self.presentations)
        result = self.run_audit()
        self.assertEqual(result['acceptance']['unique_presented_gan_fps'], 0)
        self.assertFalse(result['acceptance']['no_fallback_storm'])
        self.assertEqual(result['status'], 'FAIL')

    def test_baseline_counts_fresh_scaled_frames_without_gan_credit(self):
        path = self.directory / 'config.json'
        config = json.loads(path.read_text())
        config['enhancer'] = 'none'
        self.write('config.json', config)
        for item in self.presentations:
            item['PROVENANCE'] = 'DECODED+GAN-FALLBACK-LANCZOS'
        self.write_lines('enhancer.jsonl', self.presentations)
        result = self.run_audit()
        self.assertEqual(result['status'], 'PASS')
        self.assertGreaterEqual(result['acceptance']['unique_presented_fps'], 9)
        self.assertEqual(result['acceptance']['unique_presented_gan_fps'], 0)
        self.assertEqual(result['acceptance']['gan_success_fraction'], 0)
        for item in self.presentations:
            item['PROVENANCE'] = 'HOLD+GAN-FALLBACK-LANCZOS'
        self.write_lines('enhancer.jsonl', self.presentations)
        result = self.run_audit()
        self.assertEqual(result['acceptance']['unique_presented_fps'], 0)
        self.assertEqual(result['status'], 'FAIL')

    def test_eof_hold_tail_does_not_inflate_measured_fallback(self):
        records = [json.loads(line) for line in
                   (self.directory / 'enhancer.jsonl').read_text().splitlines()]
        last = self.presentations[-1]
        tail = [dict(last, AT=270.1 + index * .1,
                     PROVENANCE='HOLD+GAN-FALLBACK-LANCZOS') for index in range(150)]
        self.write_lines('enhancer.jsonl', records + tail)
        result = self.run_audit()
        self.assertTrue(result['acceptance']['no_fallback_storm'])
        self.assertEqual(result['acceptance']['diagnostics']['fallback_tick_fraction'], 0)
        self.assertEqual(result['status'], 'PASS')

    def test_long_hold_is_rejected_even_when_average_fps_is_high(self):
        records = [json.loads(line) for line in
                   (self.directory / 'enhancer.jsonl').read_text().splitlines()]
        for item in records:
            if item['TYPE'] == 'GAN_PRESENT' and 401 <= item['SEQ'] <= 420:
                item.update(SEQ=400, PTS=self.presentations[400]['PTS'], PROVENANCE='HOLD+GAN-ESRGAN')
        self.write_lines('enhancer.jsonl', records)
        result = self.run_audit()
        self.assertGreaterEqual(result['acceptance']['unique_presented_gan_fps'], 9)
        self.assertFalse(result['acceptance']['source_age_bounded'])

    def test_window_submission_pause_is_rejected(self):
        records = [json.loads(line) for line in
                   (self.directory / 'enhancer.jsonl').read_text().splitlines()]
        records = [item for item in records if item['TYPE'] != 'GAN_PRESENT'
                   or not 401 <= item['SEQ'] <= 420]
        self.write_lines('enhancer.jsonl', records)
        result = self.run_audit()
        self.assertGreater(result['acceptance']['diagnostics']['output_gap_ms']['max'], 2000)
        self.assertFalse(result['acceptance']['source_age_bounded'])

    def test_fallback_burst_is_not_hidden_by_low_average_fraction(self):
        records = [json.loads(line) for line in
                   (self.directory / 'enhancer.jsonl').read_text().splitlines()]
        for item in records:
            if item['TYPE'] == 'GAN_PRESENT' and 401 <= item['SEQ'] <= 420:
                item['PROVENANCE'] = 'DECODED+GAN-FALLBACK-LANCZOS'
        self.write_lines('enhancer.jsonl', records)
        result = self.run_audit()
        self.assertLess(result['acceptance']['diagnostics']['fallback_tick_fraction'], .1)
        self.assertFalse(result['acceptance']['no_fallback_storm'])

    def test_sustained_rss_growth_is_rejected(self):
        self.write_lines('memory.jsonl', [dict(at=200 + index * 5,
            process_tree_rss_mib=1000 + index * 10) for index in range(15)])
        result = self.run_audit()
        self.assertFalse(result['acceptance']['memory_stable'])
        self.assertEqual(result['status'], 'FAIL')

    def test_temporary_allocation_is_not_a_sustained_memory_trend(self):
        self.write_lines('memory.jsonl', [dict(at=200 + index * 5,
            process_tree_rss_mib=1100 if index == 8 else 1000) for index in range(15)])
        result = self.run_audit()
        self.assertTrue(result['acceptance']['memory_stable'])
        self.assertEqual(result['acceptance']['diagnostics']['steady_rss_slope_mib_per_min'], 0)

    def test_decoder_concealment_warning_rejects_apparent_frame_count_match(self):
        with mock.patch('verify_video_quality_run.cv2.VideoCapture',
                        return_value=FakeCapture()), mock.patch(
                        'verify_video_quality_run.subprocess.run',
                        return_value=mock.Mock(returncode=0, stderr='Could not find ref with POC 3')):
            result = audit(self.directory, {'synthetic_policy_test': True})
        self.assertFalse(result['acceptance']['ordering_verified'])
        self.assertEqual(result['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()

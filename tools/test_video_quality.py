import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from analyze_video_quality_sweep import packet_rates, summarize
from select_video_quality_profiles import pareto, shared_scene_pareto
from run_video_quality_sweep import candidates, commands
from live_h265_hud import (RtpStats, gan_profile_sizes, gan_profile_signature,
                           gan_native_size_text, receiver_gan_profile, WindowFrameTracker)


class VideoQualityTests(unittest.TestCase):
    def test_window_counts_same_source_gan_upgrade_once_and_ignores_hold(self):
        tracker = WindowFrameTracker()
        tracker.submit(1, 90, 1.0, 'GAN-FALLBACK-LANCZOS', 0, 1.01)
        upgraded = tracker.submit(1, 90, 1.04, 'GAN-ESRGAN', 0, 1.05)
        self.assertEqual(upgraded['PROVENANCE'], 'DECODED+GAN-ESRGAN')
        held = tracker.submit(1, 90, 1.04, 'GAN-ESRGAN', 0, 1.06)
        self.assertEqual(held['PROVENANCE'], 'HOLD+GAN-ESRGAN')

    def test_decode_snapshot_and_ui_snapshot_select_same_x2_worker(self):
        wire = dict(name='ganq120', generation=0, width=320, height=180, fps=6)
        ui = receiver_gan_profile(dict(wire), 2, False)
        decode = receiver_gan_profile(dict(wire), 2, False)
        self.assertEqual(gan_profile_signature(ui), gan_profile_signature(decode))
        self.assertEqual(gan_profile_signature(decode)[-1], 2)
        self.assertNotIn('gan_scale', wire)
        legacy = dict(wire, name='gan4')
        self.assertEqual(receiver_gan_profile(legacy, 2, False), legacy)

    def test_new_profile_metadata_and_scaling(self):
        for number, cap in ((11, 60), (12, 120), (13, 180)):
            packet = bytes([0x90, 0xe0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                            0x52, 0x4f, 0, 3, 1, number, 1, 64, 0, 180, 8, 0,
                            0, 45, 0, cap, 0x26, 1, 0])
            profile = RtpStats._profile_extension(packet)
            self.assertEqual(profile['name'], f'ganq{cap}')
            self.assertEqual(profile['link_cap_kbps'], cap)
            self.assertEqual(gan_profile_sizes(profile), ((320, 180), (1280, 720), (1280, 720)))
            smaller = dict(profile, width=256, height=144)
            self.assertEqual(gan_profile_sizes(smaller), ((256, 144), (1024, 576), (1280, 720)))
            x2 = dict(profile, gan_scale=2)
            self.assertEqual(gan_profile_sizes(x2), ((320, 180), (640, 360), (1280, 720)))
            self.assertNotEqual(gan_profile_signature(x2), gan_profile_signature(profile))
            self.assertEqual(gan_native_size_text(dict(profile, gan_baseline=True), (1280, 720)), 'none(Lanczos)')

    def test_no_evidence_never_passes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'config.json').write_text(json.dumps(dict(cap=120, fps=8, duration_s=60, warmup_s=10)))
            result = summarize(root)
            self.assertEqual(result['status'], 'NOT RUN')
            self.assertIsNone(result['utilization'])
            self.assertEqual(pareto([result]), [])

    def test_wire_windows_and_short_capture(self):
        packets = [dict(at=i * .1, wire_bytes=100, udp_bytes=34, rtp_payload_bytes=6) for i in range(801)]
        result = packet_rates(packets, 10, 60)
        self.assertAlmostEqual(result['wire_kbps'], 8)
        self.assertAlmostEqual(result['rtp_payload_kbps'], .48)
        self.assertAlmostEqual(result['wire_60s_kbps']['max'], 8)
        self.assertEqual(packet_rates(packets[:30], 10, 60)['status'], 'INCOMPLETE')

    def test_prior_remote_log_frames_do_not_contaminate_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = dict(candidates('B')[0], duration_s=60, warmup_s=0)
            (directory / 'config.json').write_text(json.dumps(config))
            (directory / 'sender.log').write_text('QUALITY_TX pts_us=100000000\n')
            frames = [dict(TYPE='ENC_FRAME', PTS=1, ENCODED_BITS=1000000),
                      dict(TYPE='ENC_FRAME', PTS=200000000, ENCODED_BITS=1000000),
                      dict(TYPE='ENC_FRAME', PTS=100000000, ENCODED_BITS=60000)]
            (directory / 'encoder.jsonl').write_text('\n'.join(json.dumps(x) for x in frames))
            worker = [dict(TYPE='GAN_WORKER', SEQ=1, PTS=8999999, ARRIVAL=0,
                           OUTPUT_READY=True, INFER_RETURNED=True, INFER_MS=1),
                      dict(TYPE='GAN_WORKER', SEQ=2, PTS=9000000, ARRIVAL=100,
                           OUTPUT_READY=True, INFER_RETURNED=True, INFER_MS=1)]
            (directory / 'enhancer.jsonl').write_text('\n'.join(json.dumps(x) for x in worker))
            self.assertEqual(summarize(directory)['es_kbps'], 1)
            self.assertEqual(summarize(directory)['worker_unique_ready_fps'], 1 / 60)

    def test_pareto_preserves_quality_fps_tradeoffs(self):
        base = dict(status='PASS', strict_same_source=True, scene='text', cap=120)
        def item(ssim, fps, reference='same'):
            return dict(base, quality=dict(ssim=ssim, temporal_variation=1, reference_id=reference),
                        acceptance=dict(unique_presented_gan_fps=fps))
        good, smooth, bad = item(.95, 6), item(.9, 8), item(.8, 5)
        self.assertEqual(pareto([good, smooth, bad]), [good, smooth])
        distinct = item(.7, 4, 'different')
        self.assertIn(distinct, pareto([good, distinct]))

    def test_bounded_candidate_search(self):
        configs = candidates('B')
        self.assertEqual({item['cap'] for item in configs}, {60, 120, 180})
        self.assertTrue(all(item['width'] * 9 == item['height'] * 16 for item in configs))
        self.assertTrue(all(item['adaptive'] is False for item in configs))
        with self.assertRaises(ValueError):
            candidates('C')

    def test_pareto_preserves_roi_quality_and_counts_baseline_presentation(self):
        base = dict(status='PASS', strict_same_source=True, scene='text', cap=120)
        def item(ssim, roi, actual_fps, gan_fps):
            return dict(base, quality=dict(ssim=ssim, roi_ssim=roi, temporal_variation=1,
                        reference_id='same'), acceptance=dict(unique_presented_fps=actual_fps,
                        unique_presented_gan_fps=gan_fps))
        global_best = item(.95, .7, 8, 8)
        text_best = item(.9, .9, 8, 8)
        self.assertEqual(pareto([global_best, text_best]), [global_best, text_best])
        baseline = item(.96, .91, 8, 0)
        self.assertEqual(pareto([global_best, text_best, baseline]), [baseline])

    def test_baseline_command_uses_auto_provider(self):
        args = SimpleNamespace(pc_host='127.0.0.1', port=5004, duration=60, warmup=10,
                               input_video=None, board_dir='/opt/app', board_host='root@board',
                               provider='tensorrt-fp16')
        config = dict(candidates('B')[0], enhancer='none')
        _, receiver, _ = commands(config, args, Path('run'))
        self.assertIn('--gan-execution-provider=auto', receiver)
        _, receiver, _ = commands(dict(config, enhancer='x2'), args, Path('run'))
        self.assertIn('--ganq-scale=2', receiver)
        self.assertIn('--headless', receiver)
        args.display = True
        _, receiver, _ = commands(config, args, Path('run'))
        self.assertNotIn('--headless', receiver)

    def test_selected_encoder_settings_survive_later_stages(self):
        selected = dict(candidates('B')[0], qp_min_i=30, qp_max_i=32,
                        super_frame_policy='relaxed', max_i_prop=20, qp_ip=2)
        args = SimpleNamespace(pc_host='127.0.0.1', port=5004, duration=60, warmup=10,
                               input_video=None, board_dir='/opt/app', board_host='root@board',
                               provider='tensorrt-fp16')
        for stage in ('C', 'D', 'E', 'F'):
            for config in candidates(stage, [selected]):
                sender, _, _ = commands(config, args, Path('run'))
                self.assertIn('--gan-qp-min-i=30', sender[-1])
                self.assertIn('--qp-max-i=32', sender[-1])
                self.assertIn('--gan-super-frame=relaxed', sender[-1])
                self.assertIn('--gan-max-i-prop=20', sender[-1])
                self.assertIn('--gan-qp-ip=2', sender[-1])

    def test_targeted_qp_followup_uses_existing_selected_configuration(self):
        base = dict(candidates('B')[0], gop_candidates=[2], qp_candidates=[[16, 40]])
        result = candidates('C', [base])
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]['gop_seconds'], result[0]['qp_min'], result[0]['qp_max']),
                         (2, 16, 40))
        with self.assertRaises(ValueError):
            candidates('C', [dict(base, qp_candidates=[[40, 16]])])

    def test_joint_selection_requires_both_scenes_and_preserves_tradeoffs(self):
        def record(target, scene, score, status='PASS', reference=None):
            return dict(candidates('B')[0], target=target, scene=scene, status=status,
                strict_same_source=True,
                quality=dict(reference_id=reference or scene, input_checksum_verified=True,
                             ssim=score, roi_ssim=score, temporal_variation=1),
                acceptance=dict(unique_presented_gan_fps=6,
                                end_to_end_ms=dict(p95=200), hardware='same device'))
        records = [record(40, 'person', .95), record(40, 'text', .8),
                   record(45, 'person', .9), record(45, 'text', .9),
                   record(50, 'person', .99), record(50, 'text', .99, status='FAIL')]
        joint = shared_scene_pareto(records, ['person', 'text'])
        self.assertEqual({item['target'] for item in joint}, {40, 45})
        self.assertTrue(all(set(item['scene_results']) == {'person', 'text'} for item in joint))
        records += [record(55, 'person', .7), record(55, 'text', .7)]
        self.assertEqual({item['target'] for item in shared_scene_pareto(records, ['person', 'text'])},
                         {40, 45})

    def test_joint_selection_does_not_compare_distinct_references(self):
        def record(target, scene, reference):
            return dict(candidates('B')[0], target=target, scene=scene, status='PASS',
                strict_same_source=True,
                quality=dict(reference_id=reference, input_checksum_verified=True,
                             ssim=target / 100, roi_ssim=target / 100, temporal_variation=1),
                acceptance=dict(unique_presented_gan_fps=6, end_to_end_ms=dict(p95=200)))
        records = [record(40, 'person', 'person1'), record(40, 'text', 'text1'),
                   record(50, 'person', 'person2'), record(50, 'text', 'text1')]
        self.assertEqual(len(shared_scene_pareto(records, ['person', 'text'])), 2)


if __name__ == '__main__':
    unittest.main()

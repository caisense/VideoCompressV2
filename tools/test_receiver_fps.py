"""Production CLI dispatch and metadata-driven motion interpolation contracts."""
import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import tools.live_h265_hud as hud
from tools.interpolated_receiver import interpolation_settings


def options(**overrides):
    values = dict(gan_enhancer="esrgan", gan_esrnet_model="net.onnx",
                  gan_esrgan_model="x2.onnx", gan_esrgan_x4_model="x4.onnx",
                  gan_output_latency_budget_ms=0., gan_output_latency_factor=1.15,
                  gan_max_inference_latency_ms=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class ReceiverFpsTests(unittest.TestCase):
    def test_fps_dispatches_from_real_entrypoint_with_existing_flags(self):
        run = mock.Mock(return_value=7)
        with mock.patch.dict(sys.modules, {"interpolated_receiver": SimpleNamespace(run_interpolated_receiver=run)}), mock.patch.dict(__import__("os").environ, {"VC_RUNTIME_RECEIVER_WORKER": "1"}):
            with mock.patch.object(sys, "argv", ["live_h265_hud.py", "--fps=15", "--udp-port=5014", "--gan-enhancer=esrgan"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(hud.main(), 7)
        args, port = run.call_args.args
        self.assertEqual(port, 5014)
        self.assertEqual(args.fps, 15)
        self.assertEqual(args.gan_execution_provider, "tensorrt-fp16")
        self.assertEqual(args.duration, 0)

    def test_invalid_or_ambiguous_fps_is_rejected_before_opening_socket(self):
        cases = [["--fps="+value, "--gan-enhancer=esrgan"] for value in ["0", "-1", "nan", "inf", "121"]]
        cases += [["--fps=15"], ["--fps=15", "--gan-enhancer=esrgan", "--gan-display-fps=15"]]
        for argv in cases:
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", ["live_h265_hud.py"]+argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                    hud.main()
                self.assertEqual(result.exception.code, 2)

    def test_metadata_selects_native_output_and_x4_model_without_receiver_profile_flag(self):
        for scale in [2, 4]:
            profile = dict(name="rate60", width=320, height=180, fps=10, gan_scale=scale)
            result = interpolation_settings(options(), profile)
            self.assertEqual(result['source'], (320, 180))
            self.assertEqual(result['native'], (320*scale, 180*scale))
            self.assertEqual(result['output'], result['native'])
            self.assertEqual(result['model'], f'x{scale}.onnx')

    def test_explicit_sr_budget_is_kept_separate_from_interpolation_playout(self):
        profile = dict(name="rate120", width=480, height=270, fps=10, gan_scale=2)
        result = interpolation_settings(options(gan_output_latency_budget_ms=90), profile)
        self.assertEqual(result['budget_ms'], 90)
        self.assertEqual(result['output'], (960, 540))

    def test_plain_stream_and_incompatible_esrnet_do_not_silently_fallback(self):
        with self.assertRaisesRegex(ValueError, 'sender GAN hint'):
            interpolation_settings(options(), dict(name='rate60', gan_scale=0, width=320, height=180, fps=10))
        with self.assertRaisesRegex(ValueError, 'native x2'):
            interpolation_settings(options(gan_enhancer='esrnet'),
                                   dict(name='rate60', gan_scale=4, width=320, height=180, fps=10))


if __name__ == '__main__':
    unittest.main()

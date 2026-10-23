"""Real window regression: native move modal loop must keep painting."""
import ctypes
import os
import threading
import time
import unittest
import numpy as np


@unittest.skipUnless(os.name == "nt", "Win32 presenter")
class NativeWindowTests(unittest.TestCase):
    def test_move_modal_loop_keeps_presenting(self):
        from tools.win32_video_window import Win32VideoWindow
        window = Win32VideoWindow("VideoCompress presenter regression")
        shown = []
        cancel_thread = None
        try:
            sample = np.full((180, 320, 3), (10, 20, 30), dtype=np.uint8)
            self.assertIsNotNone(window.show(sample, time.perf_counter()+.1))
            dc = ctypes.windll.user32.GetDC(window._hwnd)
            ctypes.windll.gdi32.GetPixel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            try:
                self.assertEqual(ctypes.windll.gdi32.GetPixel(dc, 100, 100), 30+(20<<8)+(10<<16))
            finally:
                ctypes.windll.user32.ReleaseDC(window._hwnd, dc)
            for index in range(75):
                if index == 15:
                    ctypes.windll.user32.PostMessageW(window._hwnd, 0x0112, 0xF010, 0)
                    def cancel_move():
                        time.sleep(1.5)
                        for message, key in [(0x001F, 0), (0x0100, 27), (0x0202, 0)]:
                            ctypes.windll.user32.PostMessageW(window._hwnd, message, key, 0)
                    cancel_thread = threading.Thread(target=cancel_move, daemon=True)
                    cancel_thread.start()
                start = time.perf_counter()
                image = np.full((180, 320, 3), index*3, dtype=np.uint8)
                shown.append(window.show(image, start+.05))
                time.sleep(max(0, .066-(time.perf_counter()-start)))
            self.assertGreaterEqual(sum(x is not None for x in shown), 70)
            self.assertGreaterEqual(window.modal_entries, 1)
            self.assertGreaterEqual(window.modal_paints, 10)
        finally:
            if cancel_thread:
                cancel_thread.join(2)
            window.close()

    def test_overdue_native_commit_is_reported_after_producer_timeout(self):
        from unittest import mock
        from tools.win32_video_window import Win32VideoWindow
        window = Win32VideoWindow("VideoCompress late-commit regression")
        original = ctypes.windll.gdi32.StretchDIBits
        def delayed_paint(*args):
            time.sleep(.08)
            return original(*args)
        try:
            with mock.patch.object(ctypes.windll.gdi32, 'StretchDIBits', side_effect=delayed_paint):
                result = window.show(np.zeros((180,320,3), dtype=np.uint8),
                                     time.perf_counter()+.05, tag=7)
                self.assertIsNone(result)
                time.sleep(.2)
            commits = window.take_late_commits()
            self.assertTrue(any(x['index'] == 7 and x['overdue_ms'] > 20 for x in commits))
        finally:
            window.close()

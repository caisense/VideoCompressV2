import os
import json
import socket
import threading
import unittest
from unittest import mock
import tools.live_h265_hud as hud
from tools.runtime_profile_receiver import selected_fps, worker_arguments, notify_runtime_ready


def packet(scale=2, output=15, source_fps=10, version=3):
    payload = bytes([version,0,1,64,0,180,source_fps,1,0,42,0,60,scale,output,0,0])
    return bytes([0x90,0xe0,0,1,0,0,0,1,0,0,0,1])+b'RO'+bytes([0,4])+payload+b'\x26\x01'


class RuntimeReceiverTests(unittest.TestCase):
    def test_v3_controls_interpolation_independently_of_source_fps(self):
        p = hud.RtpStats._profile_extension(packet())
        self.assertEqual(p['fps'], 10)
        self.assertEqual(p['output_fps'], 15)
        self.assertEqual(p['gan_scale'], 2)
        plain = hud.RtpStats._profile_extension(packet(0,0))
        self.assertEqual(plain['output_fps'], 0)
        self.assertIsNone(selected_fps(plain, 15))
        self.assertIsNone(selected_fps(dict(p, output_fps=0), 15))
        self.assertEqual(selected_fps(p, 30), 15)
        self.assertNotEqual(hud.profile_switch_key(p), hud.profile_switch_key(dict(p, output_fps=0)))

    def test_invalid_v3_is_rejected_and_v2_reserved_bytes_are_ignored(self):
        for scale,output in [(0,15),(2,10),(2,121),(3,15)]:
            self.assertIsNone(hud.RtpStats._profile_extension(packet(scale,output)))
        old = hud.RtpStats._profile_extension(packet(2,0,version=2))
        self.assertNotIn('output_fps', old)
        self.assertEqual(selected_fps(old, 15), 15)

    def test_worker_receives_only_selected_port_fps_and_no_duration_limit(self):
        a=worker_arguments(['--fps','15','--udp-port=5004','--duration','60','--rotate=ccw90'],5017,None)
        self.assertEqual(a,['--rotate=ccw90','--udp-port=5017','--duration=0'])
        self.assertEqual(worker_arguments(a,5019,30)[-1], '--fps=30')

    def test_ready_handshake_and_graceful_stop(self):
        parent=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        parent.bind(('127.0.0.1',0));parent.settimeout(1)
        stop=threading.Event()
        try:
            with mock.patch.dict(os.environ,VC_RUNTIME_READY_PORT=str(parent.getsockname()[1]),VC_RUNTIME_TOKEN='test'):
                notify_runtime_ready(stop)
            message,_=parent.recvfrom(1024);notice=json.loads(message)
            self.assertEqual(notice['token'],'test')
            parent.sendto(b'STOP:test',('127.0.0.1',notice['port']))
            self.assertTrue(stop.wait(1))
        finally:
            stop.set();parent.close()

    def test_loopback_worker_uses_ingress_wire_but_its_own_decode_and_profile(self):
        parent=hud.RtpStats();parent.on_packet(packet())
        network=parent.snapshot();network['lost']=7;network['wire_kbps']=119
        worker=hud.RtpStats();worker.on_packet(packet())
        worker.on_decode_error();worker.on_decoded_frame()
        with mock.patch.dict(os.environ, VC_RUNTIME_RECEIVER_WORKER='1'):
            with mock.patch('runtime_profile_receiver.runtime_network_stats',return_value=network):
                result=worker.snapshot()
        self.assertEqual(result['wire_kbps'],119)
        self.assertEqual(result['lost'],7)
        self.assertEqual(result['decode_errors'],1)
        self.assertEqual(result['decode_fps'],1)
        self.assertEqual(result['profile']['generation'],1)
        self.assertIn('ingress',result['wire_scope'])

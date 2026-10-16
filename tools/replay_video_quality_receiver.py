#!/usr/bin/env python3
"""Functional localhost RTP replay of a real H.265 capture with ganq metadata.

This is not MPP rate-control or physical CAP acceptance. The receiver uses its
normal RTP/decode/profile/GAN path; the source must be a real captured bitstream.
"""
import argparse
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2


def decoded_hashes(ffmpeg, source, frames):
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-xerror', '-err_detect', 'explode', '-i', str(source),
               '-frames:v', str(frames), '-fps_mode', 'passthrough',
               '-f', 'framemd5', '-']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('strict replay frame verification failed: ' + result.stderr)
    return [line.split(',')[-1].strip() for line in result.stdout.splitlines()
            if line and not line.startswith('#')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--cap', type=int, choices=(60, 120, 180), default=180)
    parser.add_argument('--enhancer', choices=('none', 'x2', 'x4'), default='x4')
    parser.add_argument('--duration', type=int, default=20)
    parser.add_argument('--fps', type=int, default=8,
                        help='actual source FPS; used for RTP pacing and profile metadata')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--port', type=int, default=5094)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60 or args.duration <= 0:
        parser.error('FPS must be 1..60 and duration must be positive')
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(args.source))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        parser.error('source must be a decodable real H.265 capture')
    height, width = frame.shape[:2]
    target = {60: 45, 120: 86, 180: 145}[args.cap]
    payload = bytes([1, {60: 11, 120: 12, 180: 13}[args.cap]]) + width.to_bytes(2, 'big') + \
        height.to_bytes(2, 'big') + bytes([args.fps, 0]) + target.to_bytes(2, 'big') + args.cap.to_bytes(2, 'big')
    extension = b'RO\x00\x03' + payload
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(('127.0.0.1', args.port + 1))
    udp.settimeout(.2)
    done = threading.Event()
    def proxy():
        while not done.is_set():
            try:
                packet, _ = udp.recvfrom(65535)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            if len(packet) < 12 or packet[0] >> 6 != 2 or packet[1] & 127 != 96:
                continue
            offset = 12 + 4 * (packet[0] & 15)
            if packet[0] & 16:
                continue
            udp.sendto(bytes([packet[0] | 16]) + packet[1:offset] + extension + packet[offset:],
                       ('127.0.0.1', args.port))
    thread = threading.Thread(target=proxy)
    thread.start()
    root = Path(__file__).resolve().parents[1]
    receiver = [sys.executable, '-u', str(root / 'tools/live_h265_hud.py'), '--rotate=none',
        f'--duration={args.duration + 5}', f'--udp-port={args.port}',
        '--gan-enhancer=' + ('none' if args.enhancer == 'none' else 'esrgan'),
        '--ganq-scale=' + ('2' if args.enhancer == 'x2' else '4'),
        '--gan-execution-provider=' + ('auto' if args.enhancer == 'none' else 'tensorrt-fp16'),
        f'--gan-debug-log={args.output / "enhancer.jsonl"}',
        f'--quality-packet-log={args.output / "packets.jsonl"}',
        f'--quality-video-log={args.output / "received.h265"}']
    if not args.display:
        receiver.append('--headless')
    sender = [args.ffmpeg, '-hide_banner', '-loglevel', 'warning', '-re', '-r', str(args.fps), '-i', str(args.source),
              '-frames:v', str(args.duration * args.fps), '-c:v', 'copy', '-an', '-payload_type', '96', '-f', 'rtp',
              f'rtp://127.0.0.1:{args.port + 1}?pkt_size=1156']
    result = dict(scope='localhost real captured H.265 RTP replay, no CAP enforcement or board acceptance',
                  cap_metadata=args.cap, source_fps=args.fps, display=args.display, source=str(args.source),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  receiver=receiver, sender=sender)
    try:
        with (args.output / 'receiver.log').open('w', encoding='utf-8') as log:
            rx = subprocess.Popen(receiver, stdout=log, stderr=subprocess.STDOUT, cwd=root)
            ready_deadline = time.monotonic() + 10
            while 'H.265 RTP input:' not in (args.output / 'receiver.log').read_text(errors='replace'):
                if rx.poll() is not None or time.monotonic() > ready_deadline:
                    rx.terminate()
                    rx.wait(timeout=5)
                    raise RuntimeError('replay receiver did not bind before bounded readiness deadline')
                time.sleep(.05)
            try:
                tx = subprocess.run(sender, capture_output=True, text=True, timeout=args.duration + 10)
                (args.output / 'replay.log').write_text(tx.stdout + tx.stderr, encoding='utf-8')
                rx.wait(timeout=args.duration + 15)
                result['returncodes'] = [tx.returncode, rx.returncode]
            finally:
                if rx.poll() is None:
                    rx.terminate()
                    rx.wait(timeout=5)
    finally:
        done.set()
        thread.join(timeout=2)
        udp.close()
    if result.get('returncodes') == [0, 0]:
        expected_frames = args.duration * args.fps
        expected = decoded_hashes(args.ffmpeg, args.source, expected_frames)
        actual = decoded_hashes(args.ffmpeg, args.output / 'received.h265', expected_frames + 1)
        result['decoded_frame_identity'] = dict(
            expected_count=expected_frames, source_count=len(expected),
            received_count=len(actual), exact_hash_sequence_match=(
                len(expected) == expected_frames and expected == actual),
            method='strict FFmpeg framemd5 decoded pixel hash sequence, independent of RTP timestamps')
        (args.output / 'decoded_frame_hashes.json').write_text(json.dumps(dict(
            source=expected, received=actual), indent=2))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result.get('returncodes'))
    return int(result.get('returncodes') != [0, 0] or not result.get(
        'decoded_frame_identity', {}).get('exact_hash_sequence_match'))


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay real decoded frames through the production bounded GAN worker.

Measures PC processing only. No sender CAP, true reference quality or
board-to-PC latency claim can be made from this offline test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import psutil

from analyze_video_quality_sweep import percentiles, read_jsonl
from full_frame_enhancer import LatestOnlyEnhancerWorker, ProcessFullFrameEnhancer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--enhancer', choices=('none', 'x2', 'x4'), required=True)
    parser.add_argument('--provider', choices=('cuda', 'tensorrt-fp16'), default='tensorrt-fp16')
    parser.add_argument('--fps', type=int, choices=(6, 8, 10, 12), default=8)
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--warmup-seconds', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.duration < 60 or args.warmup_seconds < 0:
        parser.error('duration must be >=60s; warmup must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(args.source))
    frames = []
    for _ in range(256):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        parser.error('real source video cannot be decoded')
    height, width = frames[0].shape[:2]
    scale = 4 if args.enhancer == 'x4' else 2
    model = Path('model') / ('RealESRGAN_x4plus_dynamic.onnx' if scale == 4 else 'RealESRGAN_x2_dynamic.onnx')
    config = dict(source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  input=[width, height], native=[width * scale, height * scale] if args.enhancer != 'none' else None,
                  output=[1280, 720], source_fps=args.fps, duration_s=args.duration,
                  warmup_s=args.warmup_seconds, enhancer=args.enhancer,
                  scope='offline PC real decoded frame replay, first 256 frames looped; no CAP validation',
                  reference_quality='NOT RUN', board_to_pc_latency='NOT RUN')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    runtime = ProcessFullFrameEnhancer(backend='none' if args.enhancer == 'none' else 'esrgan',
        model_path=None if args.enhancer == 'none' else model, input_size=(width, height),
        native_size=(1280, 720) if args.enhancer == 'none' else (width * scale, height * scale), output_size=(1280, 720),
        execution_provider='auto' if args.enhancer == 'none' else args.provider, warmup=2)
    worker = LatestOnlyEnhancerWorker(runtime, max_latency_ms=1000 / args.fps * 2,
                                     debug_log=args.output / 'enhancer.jsonl', logger=lambda _: None)
    start = time.monotonic()
    next_frame = start
    sequence = 0
    outputs, ages, memory, gpu_memory = [], [], [], []
    process = psutil.Process()
    next_memory = start
    saved = set()
    clip_indices = set()
    clip = cv2.VideoWriter(str(args.output / 'enhanced_clip.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                           args.fps, (1280, 720))
    if not clip.isOpened():
        worker.stop()
        raise RuntimeError('cannot create A/B clip')
    try:
        while time.monotonic() < start + args.warmup_seconds + args.duration:
            now = time.monotonic()
            if now >= next_frame:
                worker.submit(frames[sequence % len(frames)], sequence, arrived_at=now,
                              input_fps=args.fps, generation=0)
                sequence += 1
                next_frame = start + sequence / args.fps
            result = worker.poll_output()
            if result is not None and result.arrived_at >= start + args.warmup_seconds:
                outputs.append(result.source_sequence)
                ages.append((now - result.arrived_at) * 1000)
                # Exact same-source indices enable visual comparison across runs.
                index = result.source_sequence % len(frames)
                if 48 <= index < 88 and index not in clip_indices:
                    clip_indices.add(index)
                    clip.write(result.frame)
                if index in (48, 96, 144) and index not in saved:
                    saved.add(index)
                    cv2.imwrite(str(args.output / f'source_{index}.png'), frames[index])
                    cv2.imwrite(str(args.output / f'output_{index}.png'), result.frame)
            if now >= next_memory:
                children = process.children(recursive=True)
                memory.append(dict(at_s=now - start, rss_mb=sum(p.memory_info().rss for p in [process] + children) / 1048576))
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=5)
                if gpu.returncode == 0:
                    gpu_memory.append(dict(at_s=now - start, device_total_used_mb=gpu.stdout.strip()))
                next_memory = now + 5
            time.sleep(.002)
    finally:
        clip.release()
        worker.stop()
    records = [item for item in read_jsonl(args.output / 'enhancer.jsonl')
               if item.get('TYPE') == 'GAN_WORKER' and item['ARRIVAL'] >= start + args.warmup_seconds]
    summary = dict(config, provider=runtime.provider, execution_precision=runtime.execution_precision,
        unique_output_fps=len(set(outputs)) / args.duration, unique_outputs=len(set(outputs)),
        success_fraction=len(set(outputs)) / (args.duration * args.fps),
        infer_ms=percentiles([item['INFER_MS'] for item in records if item.get('INFER_RETURNED')]),
        pc_age_ms=percentiles(ages), memory=memory, gpu_memory=gpu_memory,
        status='RECORDED', dropped=worker.snapshot()['dropped'],
        clip_source_indices=sorted(clip_indices), clip_fps=args.fps)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('enhancer', 'unique_output_fps', 'success_fraction', 'infer_ms', 'pc_age_ms')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

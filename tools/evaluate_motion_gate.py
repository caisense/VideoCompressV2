"""Compare interpolation revisions on identical compressed input and SR endpoints.

30fps reference -> every third frame encoded at 10fps -> native x2 SR -> 15fps.
Only synthesized grid times are scored against held-out reference frames.
This offline software-encoder experiment is not board/wire/latency acceptance.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from analyze_video_quality_sweep import percentiles
from evaluate_gan_quality import psnr, ssim
from full_frame_enhancer import ProcessFullFrameEnhancer
from motion_interpolation import MotionPair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', choices=('320x180', '480x270'), required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--model', type=Path, default=Path('model/RealESRGAN_x2_dynamic.onnx'))
    args = parser.parse_args()
    if args.frames < 7:
        parser.error('--frames must be at least 7')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    spec = importlib.util.spec_from_file_location('baseline_motion', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    width, height = map(int, args.size.split('x'))
    native = (width*2, height*2)
    capture = cv2.VideoCapture(str(args.reference))
    if abs(capture.get(cv2.CAP_PROP_FPS)-30) > .01:
        raise ValueError('reference must have a true 30fps time grid')
    frames = []
    for _ in range(args.frames):
        ok, image = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(image, native, interpolation=cv2.INTER_LANCZOS4))
    capture.release()
    count = (len(frames)-1)//3 + 1
    rate = 42 if width == 320 else 88
    filters = f'select=not(mod(n\\,3)),setpts=N/(10*TB),scale={width}:{height}:flags=lanczos'
    if width == 320:
        filters += ',format=gray,format=yuv420p'
        frames = [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                  for f in frames]
    command = [args.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-i', str(args.reference), '-vf', filters, '-r', '10',
               '-frames:v', str(count), '-c:v', 'libx265', '-b:v', f'{rate}k',
               '-x265-params', 'bframes=0:keyint=40:min-keyint=40:scenecut=0:log-level=error',
               '-pix_fmt', 'yuv420p', '-an', str(args.output/'input.h265')]
    subprocess.run(command, check=True, timeout=90, capture_output=True)
    capture = cv2.VideoCapture(str(args.output/'input.h265'))
    decoded = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded.append(image)
    capture.release()
    if len(decoded) != count:
        raise ValueError('decoded/source frame-count mismatch')
    enhancer = ProcessFullFrameEnhancer(
        backend='esrgan', model_path=args.model, input_size=(width, height),
        native_size=native, output_size=native, execution_provider='tensorrt-fp16',
        tensorrt_cache_dir='runs/tensorrt_cache', threads=2, warmup=4)
    video = cv2.VideoWriter(str(args.output/'AB.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                            15, (native[0]*3, native[1]))
    if not video.isOpened():
        enhancer.close()
        raise RuntimeError('A/B writer failed')
    rows, times, sr_ms = [], collections.defaultdict(list), []
    snapshots = []
    try:
        start = time.perf_counter()
        previous_sr = enhancer.enhance(decoded[0])
        sr_ms.append((time.perf_counter()-start)*1000)
        for i in range(len(decoded)-1):
            start = time.perf_counter()
            current_sr = enhancer.enhance(decoded[i+1])
            sr_ms.append((time.perf_counter()-start)*1000)
            pairs = {}
            # Alternate revision order to reduce systematic cache/order bias.
            methods = [('baseline', baseline.MotionPair), ('gated', MotionPair)]
            for name, cls in (methods if i % 2 == 0 else methods[::-1]):
                start = time.perf_counter()
                pairs[name] = cls(decoded[i], decoded[i+1], previous_sr, current_sr)
                times[name+'_pair_ms'].append((time.perf_counter()-start)*1000)
            for n in range(3*i, 3*i+3):
                if n % 2:
                    continue
                alpha = (n-3*i)/3
                pictures = {}
                for name, pair in pairs.items():
                    start = time.perf_counter()
                    image, kind = pair.render(alpha)
                    if alpha:
                        times[name+'_render_ms'].append((time.perf_counter()-start)*1000)
                    pictures[name] = image
                    if alpha:
                        rows.append(dict(reference_index=n, revision=name, kind=kind,
                                         psnr_db=psnr(frames[n], image),
                                         ssim=ssim(frames[n], image),
                                         diagnostics=dict(getattr(pair, 'diagnostics', {}))))
                panels = [frames[n].copy(), pictures['baseline'].copy(), pictures['gated'].copy()]
                for image, title in zip(panels, ('HELD-OUT REFERENCE', 'PREVIOUS DIS', 'GATED DIS')):
                    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1)
                canvas = np.hstack(panels)
                video.write(canvas)
                if alpha:
                    delta = rows[-1]['psnr_db']-rows[-2]['psnr_db']
                    snapshots.append((delta, n, canvas))
                    snapshots = sorted(snapshots, key=lambda x: x[0])
                    if len(snapshots) > 2:
                        snapshots = [snapshots[0], snapshots[-1]]
            previous_sr = current_sr
        results = {}
        for name in ('baseline', 'gated'):
            selected = [r for r in rows if r['revision'] == name]
            results[name] = dict(synthetic_grid_frames=len(selected),
                                 psnr_db=float(np.mean([r['psnr_db'] for r in selected])),
                                 ssim=float(np.mean([r['ssim'] for r in selected])),
                                 kinds=dict(collections.Counter(r['kind'] for r in selected)))
        for label, snapshot in zip(('worst_delta', 'best_delta'), snapshots):
            cv2.imwrite(str(args.output/f'{label}_{snapshot[1]}.png'), snapshot[2])
        summary = dict(
            scope='offline software H265; same decoded frames and identical TensorRT x2 endpoints; not board acceptance',
            reference=str(args.reference.resolve()), reference_sha256=digest(args.reference),
            reference_geometry='original sensor resized with Lanczos to output geometry; not native HD ground truth',
            source=[width, height, 10], output=[*native, 15], frames_30fps=len(frames),
            h265_target_kbps=rate, encode_command=command,
            provider=enhancer.provider, precision=enhancer.execution_precision,
            model_sha256=digest(args.model), baseline_sha256=digest(args.baseline),
            current_sha256=digest(Path(__file__).with_name('motion_interpolation.py')),
            results=results, times_ms={k: percentiles(v) for k, v in times.items()},
            sr_ms=percentiles(sr_ms))
        (args.output/'frames.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        enhancer.close()
        video.release()


if __name__ == '__main__':
    main()

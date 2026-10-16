#!/usr/bin/env python3
"""Compare captured H.265 frames with their exact lossless sender input frames.

This is offline quality evaluation, not a throughput or wire-cap benchmark.
Reference inputs must be verified against the board input before acceptance.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from analyze_video_quality_sweep import read_jsonl
from evaluate_gan_quality import psnr, ssim
from full_frame_enhancer import FullFrameEnhancer


def profile_backend(runtime, frame: np.ndarray, output: Path) -> dict:
    """Observe provider assignment, separately from timed live experiments."""
    import onnxruntime as ort
    output.parent.mkdir(parents=True, exist_ok=True)
    options = runtime.session.get_session_options()
    options.enable_profiling = True
    options.profile_file_prefix = str(output.with_suffix(''))
    providers = runtime.session.get_providers()
    provider_options = runtime.session.get_provider_options()
    # ORT's reported defaults are not all round-trippable (for example, newer
    # TensorRT boolean options may be returned as '0' but require 'False').
    # Reuse only the explicit options used to construct the production session.
    trt_keys = ('device_id', 'trt_engine_cache_path', 'trt_engine_cache_prefix',
                'trt_profile_min_shapes', 'trt_profile_opt_shapes', 'trt_profile_max_shapes')
    trt_reported = provider_options.get('TensorrtExecutionProvider', {})
    trt_options = {key: trt_reported[key] for key in trt_keys if key in trt_reported}
    trt_options.update(trt_fp16_enable=True, trt_engine_cache_enable=True)
    requested = {'TensorrtExecutionProvider': trt_options,
                 'CUDAExecutionProvider': {'use_tf32': 0}, 'CPUExecutionProvider': {}}
    session = ort.InferenceSession(str(runtime.model_path), sess_options=options,
        providers=providers, provider_options=[requested.get(name, {}) for name in providers])
    for _ in range(3):
        session.run([runtime.output_name], {runtime.input_name: runtime._preprocess(frame)})
    trace = Path(session.end_profiling())
    events = json.loads(trace.read_text(encoding='utf-8'))
    counts = collections.Counter(event.get('args', {}).get('provider') for event in events
                                 if event.get('args', {}).get('provider'))
    result = dict(node_events_by_provider=dict(counts), trace=str(trace.resolve()),
        provider_options=provider_options, profiling_provider_options=requested,
        model_input_dtype=runtime.session.get_inputs()[0].type,
        precision=runtime.execution_precision,
        precision_scope='TensorRT FP16 enabled; provider events do not prove every layer executes FP16')
    if not counts.get('TensorrtExecutionProvider'):
        raise RuntimeError('profiling did not observe any TensorRT execution events')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def frame_alignment(directory: Path) -> list[tuple[int, int]]:
    encoded = read_jsonl(directory / 'encoder.jsonl')
    sender = (directory / 'sender.log').read_text(encoding='utf-8', errors='replace')
    trial_pts = [int(value) for value in re.findall(r'^QUALITY_TX pts_us=(\d+)', sender, re.MULTILINE)]
    lower, upper = (min(trial_pts), max(trial_pts)) if trial_pts else (0, float('inf'))
    encoded = [item for item in encoded if item.get('TYPE') == 'ENC_FRAME'
               and lower <= item['PTS'] <= upper]
    mapping = {(item['PTS'] * 90 // 1000) & 0xffffffff: item for item in encoded}
    timestamps = []
    seen = set()
    for packet in read_jsonl(directory / 'packets.jsonl'):
        pts = packet.get('rtp_timestamp')
        if packet.get('marker') and pts in mapping and pts not in seen:
            timestamps.append(pts)
            seen.add(pts)
    if not timestamps:
        raise ValueError('no RTP marker / encoder PTS alignment evidence')
    return [(mapping[pts]['SEQ'], mapping[pts]['GOP_POSITION']) for pts in timestamps]


def evaluate(directory: Path, reference: Path, reference_id: str,
             variants: tuple[str, ...] = ('none', 'x2', 'x4'),
             backend_profile_root: Path | None = None,
             save_clips: bool = False,
             runtime_cache: dict | None = None) -> dict:
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    alignment = frame_alignment(directory)
    capture = cv2.VideoCapture(str(directory / 'received.h265'))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != len(alignment):
        raise ValueError(f'decoded frame / complete AU mismatch: {len(frames)} / {len(alignment)}')
    fps = config['fps']
    first_seq = config['warmup_s'] * fps
    window_starts = [round(index * (config['duration_s'] - 2) / 4) for index in range(5)]
    wanted = {first_seq + second * fps + position
              for second in window_starts for position in range(2 * fps)}
    selected = [(frame, seq, phase) for frame, (seq, phase) in zip(frames, alignment)
                if seq in wanted]
    actual = [seq for _, seq, _ in selected]
    missing = wanted.difference(actual)
    if missing or len(actual) != len(wanted):
        raise ValueError(f'incomplete quality windows: missing {len(missing)} of '
                         f'{len(wanted)} source frames; observed {len(actual)}')
    sensor = cv2.VideoCapture(str(reference))
    if not sensor.isOpened() or abs(sensor.get(cv2.CAP_PROP_FPS) - fps) > .01:
        raise ValueError('lossless reference fps does not match sender input fps')
    reference_count = int(sensor.get(cv2.CAP_PROP_FRAME_COUNT))
    if reference_count <= 0:
        raise ValueError('lossless reference has no known frame count')
    reference_frames = {}
    for _, seq, _ in selected:
        source_seq = seq % reference_count if config.get('loop_input_video') else seq
        sensor.set(cv2.CAP_PROP_POS_FRAMES, source_seq)
        ok, frame = sensor.read()
        if not ok:
            raise ValueError(f'reference missing exact source frame {seq}')
        reference_frames[seq] = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
    source_width = int(sensor.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(sensor.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sensor.release()
    # Fixed semantic crops in the unrotated sensor image, inspected in previews.
    roi = (250, 250, 150, 225) if config['scene'] == 'text' else (350, 370, 210, 110)
    x, y, width, height = roi
    bounds = (round(x * 1280 / source_width), round(y * 720 / source_height),
              round((x + width) * 1280 / source_width),
              round((y + height) * 720 / source_height))
    results = {}
    snapshots = {}
    for variant in variants:
        scale = 2 if variant == 'x2' else 4
        cache_key = (variant, config['width'], config['height'])
        runtime = runtime_cache.get(cache_key) if runtime_cache is not None else None
        if runtime is None:
            runtime = FullFrameEnhancer(
                backend='none' if variant == 'none' else 'esrgan',
                model_path=None if variant == 'none' else Path('model') / (
                    'RealESRGAN_x2_dynamic.onnx' if scale == 2 else 'RealESRGAN_x4plus_dynamic.onnx'),
                input_size=(config['width'], config['height']),
                native_size=(1280, 720) if variant == 'none' else (
                    config['width'] * scale, config['height'] * scale),
                output_size=(1280, 720),
                execution_provider='auto' if variant == 'none' else 'tensorrt-fp16',
                threads=2, warmup=1)
            if runtime_cache is not None:
                runtime_cache[cache_key] = runtime
        if variant != 'none' and backend_profile_root is not None:
            profile = backend_profile_root / f'{variant}_{config["width"]}x{config["height"]}.json'
            if not profile.exists():
                profile_backend(runtime, selected[0][0], profile)
        records = []
        variation = []
        previous = None
        writer = None
        if save_clips:
            writer = cv2.VideoWriter(str(directory / f'quality_{variant}.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
            if not writer.isOpened():
                raise RuntimeError('could not open illustrative quality clip writer')
        for frame, seq, phase in selected:
            restored = runtime.enhance(frame)
            target = reference_frames[seq]
            gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
            truth = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            left, top, right, bottom = bounds
            error = gray.astype(np.float32) - truth.astype(np.float32)
            if previous is not None and seq == previous[0] + 1:
                variation.append(float(np.mean(np.abs(error - previous[1]))))
            previous = (seq, error)
            records.append(dict(seq=seq, gop_position=phase, psnr=psnr(gray, truth),
                                ssim=ssim(gray, truth), roi_psnr=psnr(gray[top:bottom, left:right],
                                truth[top:bottom, left:right]), roi_ssim=ssim(gray[top:bottom, left:right],
                                truth[top:bottom, left:right]),
                                mean_luma=float(np.mean(gray)),
                                reference_mean_luma=float(np.mean(truth)),
                                luma_error=float(np.mean(error)),
                                sharpness=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                                reference_sharpness=float(cv2.Laplacian(truth, cv2.CV_32F).var())))
            if variant not in snapshots:
                snapshots[variant] = restored
                cv2.imwrite(str(directory / f'quality_{variant}.png'), restored)
            if writer is not None:
                preview = restored.copy()
                cv2.putText(preview, f'{variant} | source SEQ {seq} | {seq / fps:.3f}s',
                            (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                writer.write(preview)
        if writer is not None:
            writer.release()
        if not records or not variation:
            raise ValueError('not enough aligned consecutive reference frames')
        phases = {}
        for phase in sorted({item['gop_position'] for item in records}):
            group = [item for item in records if item['gop_position'] == phase]
            phases[str(phase)] = dict(count=len(group), **{
                metric: float(np.mean([item[metric] for item in group]))
                for metric in ('ssim', 'roi_ssim', 'luma_error', 'sharpness', 'reference_sharpness')})
        consecutive = [(a, b) for a, b in zip(records, records[1:]) if b['seq'] == a['seq'] + 1]
        temporal = {
            'luma_error_change': float(np.mean([abs(b['luma_error'] - a['luma_error'])
                                               for a, b in consecutive])),
            'sharpness_change': float(np.mean([abs(b['sharpness'] - a['sharpness'])
                                              for a, b in consecutive])),
            'reference_sharpness_change': float(np.mean([
                abs(b['reference_sharpness'] - a['reference_sharpness']) for a, b in consecutive])),
            'gop_phase_ssim_range': max(item['ssim'] for item in phases.values())
                                    - min(item['ssim'] for item in phases.values()),
            'phase_coverage': sorted(int(phase) for phase in phases),
            'scope': 'five sampled 2s windows; phase means may include source motion effects'}
        results[variant] = dict(provider=runtime.provider, precision=runtime.execution_precision,
            samples=records, psnr=float(np.mean([item['psnr'] for item in records])),
            ssim=float(np.mean([item['ssim'] for item in records])),
            roi_psnr=float(np.mean([item['roi_psnr'] for item in records])),
            roi_ssim=float(np.mean([item['roi_ssim'] for item in records])),
            temporal_variation=float(np.mean(variation)), temporal=temporal, gop_phases=phases)
        del runtime
    primary = results[config['enhancer']]
    result = dict(reference_id=reference_id, reference_file=str(reference.resolve()),
        reference_file_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        reference_sensor_size=[source_width, source_height], reference_presentation=[1280, 720],
        reference_scaling='Lanczos resize of lossless original sensor input; not native 720p ground truth',
        alignment='RTP marker timestamp -> encoder capture PTS -> exact input file frame SEQ; count equality required',
        input_checksum_verified=False, roi_sensor=list(roi),
        sample_window_start_seconds=window_starts,
        loop_reference_frames=reference_count if config.get('loop_input_video') else None,
        illustrative_clips=('MPEG-4 previews concatenate sampled windows; metrics use uncompressed '
                            'arrays before labels/preview encoding') if save_clips else None,
        temporal_metric='mean absolute change in restoration residual, consecutive frames within five 2s windows',
        scope='offline quality of real captured H.265; no runtime performance claim',
        variants=results, ssim=primary['ssim'], psnr=primary['psnr'], roi_ssim=primary['roi_ssim'],
        roi_psnr=primary['roi_psnr'], temporal_variation=primary['temporal_variation'])
    (directory / 'quality.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-id', required=True)
    parser.add_argument('--save-clips', action='store_true')
    args = parser.parse_args()
    result = evaluate(args.directory, args.reference, args.reference_id, save_clips=args.save_clips)
    print(json.dumps({key: result[key] for key in ('ssim', 'roi_ssim', 'temporal_variation')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit captured hardware runs against explicit screening policies.

This checks the observed duration only. A 60s screening PASS does not satisfy
the separate 600s candidate and 1800s final hardware stability requirements.
Wire rates use IPv4/Ethernet accounting, not a physical network tap. Latency is
software capture PTS to scheduled presentation or window submission, not optical.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import cv2

from analyze_video_quality_sweep import percentiles, read_jsonl, summarize


def capture_conversion_stats(encoded, begin, end, fps):
    rows = [r for r in encoded if begin <= r['PTS'] < end]
    reported = bool(rows) and all('CONVERSION_BACKEND' in r for r in rows)
    backends = collections.Counter(r.get('CONVERSION_BACKEND', 'UNREPORTED') for r in rows)
    recovery = [r for r in rows if r.get('CONVERSION_BACKEND') == 'OPENCV_RGB_NV12']
    start = last = None
    longest = 0
    for row in rows:
        if row.get('CONVERSION_BACKEND') == 'OPENCV_RGB_NV12':
            start = row['PTS'] if start is None else start
            last = row['PTS']
            longest = max(longest, (last - start) / 1000 + 1000 / fps)
        else:
            start = last = None
    return dict(capture_conversion_reported=reported,
                capture_conversion_backends=dict(backends),
                capture_cpu_recovery_frames=len(recovery) if reported else None,
                capture_cpu_recovery_reason_counts=dict(collections.Counter(
                    str(r.get('CONVERSION_RECOVERY_REASON')) for r in recovery)),
                capture_cpu_recovery_max_run_ms=longest if reported else None,
                capture_conversion_wait_ms=percentiles([
                    r['CONVERSION_WAIT_US'] / 1000 for r in rows if 'CONVERSION_WAIT_US' in r]),
                capture_conversion_scope='encoded frames in measured source-PTS window; CPU preprocessing recovery is separate from receiver GAN fallback')


def audit(directory: Path, hardware: dict, ffmpeg: str = 'ffmpeg') -> dict:
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    summary = summarize(directory)
    sender = (directory / 'sender.log').read_text(encoding='utf-8', errors='replace')
    tx = [{key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)}
          for line in sender.splitlines() if line.startswith('QUALITY_TX ')]
    if not tx:
        raise ValueError('no sender wire/queue counters; legacy baseline remains observational')
    lower, upper = min(item['pts_us'] for item in tx), max(item['pts_us'] for item in tx)
    encoded = [item for item in read_jsonl(directory / 'encoder.jsonl')
               if item.get('TYPE') == 'ENC_FRAME' and lower <= item['PTS'] <= upper]
    if not encoded:
        raise ValueError('no encoded frames within this sender run')
    live_camera = 'input_video' in config and not config['input_video'] and not config.get('legacy')
    driver_timestamp_verified = (bool(encoded) and all(
        item.get('CAPTURE_SOURCE') == 'V4L2_MONOTONIC'
        and item.get('DRIVER_CAPTURE_US') == item['PTS']
        and item.get('CAPTURE_READ_US', -1) >= item['PTS'] for item in encoded))
    driver_ages = [(item['CAPTURE_READ_US'] - item['DRIVER_CAPTURE_US']) / 1000
                   for item in encoded if item.get('DRIVER_CAPTURE_US', 0) > 0
                   and item.get('CAPTURE_READ_US', -1) >= item['DRIVER_CAPTURE_US']]
    begin = encoded[0]['PTS'] + config['warmup_s'] * 1e6
    end = begin + config['duration_s'] * 1e6
    mapping = {(item['PTS'] * 90 // 1000) & 0xffffffff: item for item in encoded}
    measured = {pts for pts, item in mapping.items() if begin <= item['PTS'] < end}
    packets = read_jsonl(directory / 'packets.jsonl')
    packet_order = bool(packets) and all(
        (right['sequence'] - left['sequence']) & 0xffff == 1
        for left, right in zip(packets, packets[1:]))
    markers = [item['rtp_timestamp'] for item in packets if item.get('marker')]
    mapped_markers = [pts for pts in markers if pts in mapping]
    marker_order = len(mapped_markers) == len(markers) and all(
        mapping[right]['SEQ'] > mapping[left]['SEQ']
        for left, right in zip(mapped_markers, mapped_markers[1:]))
    previous_path = directory / 'acceptance.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_hashes = previous.get('evidence_sha256', {})
    decode_files = ('received.h265', 'decoder_validation.json')
    reuse_decode = (isinstance(previous.get('diagnostics', {}).get('total_decoded_frames'), int)
                    and all((directory / name).exists() and previous_hashes.get(name) ==
                            hashlib.sha256((directory / name).read_bytes()).hexdigest()
                            for name in decode_files))
    if reuse_decode:
        decoded = previous['diagnostics']['total_decoded_frames']
        decoder_clean = json.loads((directory / 'decoder_validation.json').read_text())['clean']
    else:
        capture = cv2.VideoCapture(str(directory / 'received.h265'))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        validation_command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                              '-xerror', '-err_detect', 'explode', '-i',
                              str(directory / 'received.h265'), '-f', 'null', '-']
        validation = subprocess.run(validation_command, capture_output=True, text=True,
                                    errors='replace', timeout=max(30, config['duration_s'] / 5))
        decoder_clean = validation.returncode == 0 and not validation.stderr.strip()
        (directory / 'decoder_validation.json').write_text(json.dumps(dict(
            command=validation_command, returncode=validation.returncode,
            stderr=validation.stderr, clean=decoder_clean), indent=2), encoding='utf-8')
    decode_count_matches = decoded == len(markers) > 0
    received = measured.intersection(markers)
    records = read_jsonl(directory / 'enhancer.jsonl')
    pts_drops = max((item.get('DECODED_PTS_DROPPED', 0) for item in records
                     if item.get('TYPE') == 'QUALITY_RX_STATUS'), default=0)
    scheduled = [item for item in records if item.get('TYPE') == 'GAN_PRESENT']
    windowed = any(item.get('HEADLESS') is False for item in scheduled)
    if config.get('display') and not windowed:
        raise ValueError('windowed run has no matching GUI presentation evidence')
    event_type = 'GAN_DISPLAY_SUBMIT' if windowed else 'GAN_PRESENT'
    if windowed and not any(item.get('TYPE') == event_type for item in records):
        raise ValueError('GUI run lacks post-imshow/waitKey submission timestamps')
    presented = [item for item in records if item.get('TYPE') == event_type
                 and item.get('PTS') in measured]
    # HOLD retains the last source PTS, so source membership alone incorrectly
    # includes the receiver's quiet EOF tail. Use a fixed render-time window.
    render_begin = min((item['AT'] for item in presented), default=None)
    render_end = render_begin + config['duration_s'] if render_begin is not None else None
    if render_begin is not None:
        presented = [item for item in presented if render_begin <= item['AT'] < render_end]
    baseline = config['enhancer'] == 'none'
    # Every fresh conventional-scale image is a valid baseline output, including
    # the shared renderer's fallback branch. HOLD is never a fresh output.
    successful = [item for item in presented
                  if item.get('PROVENANCE', '').startswith('DECODED+')
                  and (baseline or (item['PROVENANCE'].startswith('DECODED+GAN-')
                                    and 'FALLBACK' not in item['PROVENANCE']))]
    successful_pts = {item['PTS'] for item in successful}
    presented_order = bool(presented) and all(
        mapping[right['PTS']]['SEQ'] >= mapping[left['PTS']]['SEQ']
        for left, right in zip(presented, presented[1:]))
    clocks = json.loads((directory / 'clock_sync.json').read_text(encoding='utf-8'))
    before, after = clocks['before'], clocks['after']
    uncertainty = max(before['uncertainty_s'], after['uncertainty_s']) * 1000
    if after['pc_at'] <= before['pc_at']:
        raise ValueError('invalid clock calibration interval')
    def source_age(item):
        fraction = max(0, min(1, (item['AT'] - before['pc_at']) /
                              (after['pc_at'] - before['pc_at'])))
        offset = before['pc_minus_board_s'] + fraction * (
            after['pc_minus_board_s'] - before['pc_minus_board_s'])
        return (item['AT'] - offset - mapping[item['PTS']]['PTS'] / 1e6) * 1000
    ages = [source_age(item) for item in successful]
    pc_begin = begin / 1e6 + before['pc_minus_board_s']
    pc_end = end / 1e6 + after['pc_minus_board_s']
    health = sorted((item for item in records if item.get('TYPE') == event_type
                     and item.get('PTS') in mapping and pc_begin <= item['AT'] < pc_end),
                    key=lambda item: item['AT'])
    all_source_ages = [source_age(item) for item in health]
    health_times = [pc_begin] + [item['AT'] for item in health] + [pc_end]
    gaps = [(b - a) * 1000 for a, b in zip(health_times, health_times[1:])]
    fallback_start = None
    fallback_runs = []
    for item in health:
        if 'FALLBACK' in item.get('PROVENANCE', ''):
            if fallback_start is None:
                fallback_start = item['AT']
        elif fallback_start is not None:
            fallback_runs.append((item['AT'] - fallback_start) * 1000)
            fallback_start = None
    if fallback_start is not None:
        fallback_runs.append((pc_end - fallback_start) * 1000)
    memory = [item for item in read_jsonl(directory / 'memory.jsonl')
              if pc_begin <= item['at'] < pc_end]
    rss_growth = slope = None
    if len(memory) >= 6:
        values = [item['process_tree_rss_mib'] for item in memory]
        rss_growth = statistics.median(values[-3:]) - statistics.median(values[:3])
        times = [item['at'] - memory[0]['at'] for item in memory]
        # A single temporary allocation can tilt OLS despite flat endpoints.
        # Use a robust median slope across pairs at least half a window apart.
        separation = max(10, times[-1] * .5)
        slopes = [(values[j] - values[i]) / (times[j] - times[i]) * 60
                  for i in range(len(times)) for j in range(i + 1, len(times))
                  if times[j] - times[i] >= separation]
        slope = statistics.median(slopes) if slopes else None
    # Published screening thresholds, not a claim of mathematical leak freedom.
    policy = dict(max_source_age_ms=1000, max_rss_growth_mib=64,
                  max_rss_slope_mib_per_min=2, max_fallback_tick_fraction=.1,
                  max_fallback_run_ms=max(1000, 10000 / config['fps']),
                  max_tx_queue_frames=3, max_tx_queue_age_ms=250)
    fallback_fraction = sum('FALLBACK' in item.get('PROVENANCE', '')
                            for item in health) / len(health) if health else 1
    success_fraction = len(successful_pts) / len(received) if received else 0
    wire = summary.get('tx_wire_kbps')
    wire_windows = [summary['transport'].get(f'wire_{window}s_kbps', {}).get('max')
                    for window in (1, 5, 60)]
    burst_bytes = 1238  # Sender default MTU 1200 plus Ethernet wire overhead 38.
    window_caps = [config['cap'] + burst_bytes * .008 / window
                   for window in (1, 5, 60)]
    wire_ok = wire is not None and 0 <= wire <= config['cap'] and all(
        value is not None and value <= cap for value, cap in zip(wire_windows, window_caps))
    whole_link = None
    if config.get('capture_ethernet'):
        from analyze_ethernet_wire import summarize as summarize_ethernet
        whole_link = summarize_ethernet(directory)
        wire_ok = wire_ok and whole_link['whole_link_cap_screening_verified']
    queue_age = summary.get('tx_oldest_queue_age_ms', {}).get('max')
    queue_ok = (summary.get('tx_queue_max', math.inf) <= policy['max_tx_queue_frames']
                and queue_age is not None and queue_age <= policy['max_tx_queue_age_ms'])
    coverage = max(0, (encoded[-1]['PTS'] - begin) / 1e6 + 1 / config['fps'])
    render_records = [item['AT'] for item in records if item.get('TYPE') == event_type]
    render_coverage = (max(0, max(render_records) - render_begin + 1 / config['fps'])
                       if render_begin is not None and render_records else 0)
    files = ['config.json', 'sender.log', 'encoder.jsonl', 'receiver.log', 'packets.jsonl',
             'decoder_validation.json',
             'received.h265', 'enhancer.jsonl', 'clock_sync.json', 'memory.jsonl']
    if (directory / 'sender_runtime.json').exists():
        files.append('sender_runtime.json')
    if (directory / 'received.runtime.json').exists():
        files.append('received.runtime.json')
    if whole_link is not None:
        files.extend(['ethernet_wire.json', 'ethernet_wire.jsonl', 'ethernet_wire.pcap',
                      'ethernet_wire_summary.json'])
        if (directory / 'ethernet_capture_identity.json').exists():
            files.extend(['ethernet_capture_identity.json', 'ethernet_capture_source.py'])
    result = dict(
        wire_cap_verified=wire_ok, ordering_verified=packet_order and marker_order
        and decode_count_matches and decoder_clean and presented_order and pts_drops == 0,
        bounded_queue_verified=queue_ok,
        memory_stable=rss_growth is not None and rss_growth <= policy['max_rss_growth_mib']
        and slope is not None and slope <= policy['max_rss_slope_mib_per_min'],
        source_age_bounded=bool(all_source_ages) and min(all_source_ages) >= -uncertainty
        and max(all_source_ages) + uncertainty <= policy['max_source_age_ms']
        and max(gaps) <= policy['max_source_age_ms']
        and (not live_camera or driver_timestamp_verified),
        no_fallback_storm=success_fraction >= .9 and (baseline or (
            fallback_fraction <= policy['max_fallback_tick_fraction']
            and max(fallback_runs, default=0) <= policy['max_fallback_run_ms'])),
        decoded_fps=len(received) / config['duration_s'],
        unique_presented_fps=len(successful_pts) / config['duration_s'],
        unique_presented_gan_fps=(len(successful_pts) / config['duration_s']
                                  if config['enhancer'] != 'none' else 0),
        gan_success_fraction=success_fraction if config['enhancer'] != 'none' else 0,
        end_to_end_ms=percentiles(ages),
        measurement_seconds=min(coverage, render_coverage, config['duration_s']),
        hardware=hardware, model=('none' if config['enhancer'] == 'none' else
            'RealESRGAN_x2_dynamic.onnx' if config['enhancer'] == 'x2' else
            'RealESRGAN_x4plus_dynamic.onnx'),
        evidence_files=files, policy=policy,
        diagnostics=dict(total_decoded_frames=decoded, total_marker_frames=len(markers),
            packet_order=packet_order, marker_order=marker_order,
            decoder_clean=decoder_clean,
            decoded_timestamp_queue_drops=pts_drops,
            decoder_evidence_reused_by_sha256=reuse_decode,
            live_camera_driver_timestamp_verified=driver_timestamp_verified,
            driver_queue_age_ms=percentiles(driver_ages),
            presented_order=presented_order, fallback_tick_fraction=fallback_fraction,
            fallback_stat_scope=('shared renderer state; conventional-scale baseline has no GAN failures'
                                 if baseline else 'GAN fallback presentation fraction'),
            source_age_ms=percentiles(all_source_ages), output_gap_ms=percentiles(gaps),
            max_fallback_run_ms=max(fallback_runs, default=0),
            steady_rss_growth_mib=rss_growth, steady_rss_slope_mib_per_min=slope,
            clock_uncertainty_ms=uncertainty, source_window_coverage_s=coverage,
            render_window_start=render_begin, render_window_end=render_end,
            render_window_coverage_s=render_coverage,
            capture_retries=sender.count('CAPTURE_RGA_RETRY'),
            **capture_conversion_stats(encoded, begin, end, config['fps'])),
        latency_scope=('V4L2 monotonic buffer timestamp' if driver_timestamp_verified else
                       'software capture-read PTS') + (' to OpenCV imshow/waitKey return'
                       if windowed else ' to scheduled headless presentation')
                       + '; clock uncertainty reported; not optical latency',
        end_to_end_sample_scope=('fresh conventional-scale outputs; HOLD excluded' if baseline else
            'successful fresh GAN outputs only; fallback/HOLD excluded; '
            'diagnostics.source_age_ms covers all-image health'),
        actual_windowed_presentation=windowed,
        whole_link_screening=whole_link,
        wire_scope=('application RTP accounting AND all outbound Ethernet capture; '
                    'management/background traffic included; timestamp scope in whole_link_screening; no physical wire tap' if whole_link is not None else
                    'application H.265 RTP IPv4/Ethernet accounting; management SSH excluded; no wire tap'),
        queue_scope='observed sender queue plus existing latest-only single pending receiver implementation',
        memory_scope='steady RSS endpoints plus median slope across pairs separated by half the observed window',
        presentation_scope='FPS/latency: source-aligned render window; health: calibrated capture-time interval including HOLD/fallback, EOF tail excluded',
        audit_version='render_and_hold_health_v5_whole_link_capture',
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        duration_scope='screening only' if config['duration_s'] < 600 else
            'candidate stability' if config['duration_s'] < 1800 else 'final stability duration',
        evidence_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in files})
    (directory / 'acceptance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return summarize(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    hardware = json.loads(args.hardware.read_text(encoding='utf-8'))
    if not hardware:
        parser.error('hardware identity evidence is required')
    results = []
    for path in sorted(args.root.rglob('config.json')):
        config = json.loads(path.read_text(encoding='utf-8'))
        if config.get('status') != 'RECORDED':
            continue
        result = audit(path.parent, hardware, ffmpeg=args.ffmpeg)
        results.append(result)
        print(path.parent.name, result['status'], flush=True)
    (args.root / 'verified_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

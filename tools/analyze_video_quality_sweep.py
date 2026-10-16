#!/usr/bin/env python3
"""Summarize real logs; missing acceptance evidence never becomes PASS."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


def percentiles(values: list[float]) -> dict:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return dict(p50=None, p95=None, p99=None, max=None)
    def quantile(q):
        pos = (len(values) - 1) * q
        low = int(pos)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (pos - low)
    return dict(p50=quantile(.50), p95=quantile(.95), p99=quantile(.99), max=values[-1])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def packet_rates(records: list[dict], warmup: float, duration: float) -> dict:
    if not records:
        return {'status': 'NOT RUN'}
    start = records[0]['at'] + warmup
    end = start + duration
    selected = [item for item in records if start <= item['at'] < end]
    if not selected or records[-1]['at'] < end - 1:
        return {'status': 'INCOMPLETE', 'reason': 'packet capture does not cover measurement interval'}
    result = {'status': 'RECORDED', 'basis': 'received IPv4 UDP, Ethernet wire accounting',
              'wire_kbps': sum(item['wire_bytes'] for item in selected) * .008 / duration,
              'rtp_payload_kbps': sum(item['rtp_payload_bytes'] for item in selected) * .008 / duration,
              'udp_kbps': sum(item['udp_bytes'] for item in selected) * .008 / duration}
    for window in (1, 5, 60):
        queue = collections.deque()
        total = 0
        values = []
        for item in selected:
            queue.append(item)
            total += item['wire_bytes']
            while queue and queue[0]['at'] <= item['at'] - window:
                total -= queue.popleft()['wire_bytes']
            if item['at'] >= start + window:
                values.append(total * .008 / window)
        if duration == window:
            # An exactly one-window trial ends before the next packet at end;
            # the complete fixed measurement window is still observable.
            values = [sum(item['wire_bytes'] for item in selected) * .008 / window]
        result[f'wire_{window}s_kbps'] = percentiles(values)
    return result


def summarize(directory: Path) -> dict:
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    result = dict(config, directory=str(directory), status='NOT RUN')
    duration = config['duration_s']
    warmup = config['warmup_s']
    result['transport'] = packet_rates(read_jsonl(directory / 'packets.jsonl'), warmup, duration)
    encoded = [item for item in read_jsonl(directory / 'encoder.jsonl') if item.get('TYPE') == 'ENC_FRAME']
    sender_path = directory / 'sender.log'
    sender_text = sender_path.read_text(encoding='utf-8', errors='replace') if sender_path.exists() else ''
    trial_pts = [int(value) for value in re.findall(r'^QUALITY_TX pts_us=(\d+)', sender_text, re.MULTILINE)]
    if trial_pts:
        # Earlier interrupted runs may have appended to the same remote log.
        # Sender timestamps identify this trial without rewriting raw evidence.
        lower, upper = min(trial_pts), max(trial_pts)
        encoded = [item for item in encoded if lower <= item['PTS'] <= upper]
    if encoded:
        begin = encoded[0]['PTS'] + warmup * 1e6
        encoded = [item for item in encoded if begin <= item['PTS'] < begin + duration * 1e6]
        result['es_kbps'] = sum(item['ENCODED_BITS'] for item in encoded) / duration / 1000
        result['idr_bits'] = percentiles([item['ENCODED_BITS'] for item in encoded if item.get('ACTUAL_KEY_FRAME')])
        result['p_bits'] = percentiles([item['ENCODED_BITS'] for item in encoded if not item.get('ACTUAL_KEY_FRAME')])
    worker = [item for item in read_jsonl(directory / 'enhancer.jsonl') if item.get('TYPE') == 'GAN_WORKER']
    if worker:
        if encoded:
            measured_timestamps = {(item['PTS'] * 90 // 1000) & 0xffffffff for item in encoded}
            worker = [item for item in worker if item.get('PTS') in measured_timestamps]
            result['worker_window_basis'] = 'same encoded capture-PTS measurement window'
        else:
            begin = min(item['ARRIVAL'] for item in worker) + warmup
            worker = [item for item in worker if begin <= item['ARRIVAL'] < begin + duration]
            result['worker_window_basis'] = 'worker arrival window; capture alignment unavailable'
        successful = {item['SEQ'] for item in worker if item.get('OUTPUT_READY') and not item.get('DROPPED')}
        result['worker_unique_ready_fps'] = len(successful) / duration
        result['infer_ms'] = percentiles([item['INFER_MS'] for item in worker if item.get('INFER_RETURNED')])
        result['provider'] = sorted({item.get('PROVIDER', 'unknown') for item in worker})
        result['precision'] = sorted({item.get('EXECUTION_PRECISION', 'unknown') for item in worker})
    log = (directory / 'receiver.log').read_text(encoding='utf-8', errors='replace') if (directory / 'receiver.log').exists() else ''
    samples = re.findall(r'decode_fps=([\d.]+).*?lost=(\d+) reorder=(\d+) decode_errors=(\d+)', log)
    structured_status = [item for item in read_jsonl(directory / 'enhancer.jsonl')
                         if item.get('TYPE') == 'QUALITY_RX_STATUS']
    if structured_status:
        result['decoded_fps_log_mean'] = statistics.fmean(item['DECODE_FPS'] for item in structured_status)
        result['errors_max'] = [max(item[key] for item in structured_status)
                                for key in ('LOST', 'REORDER', 'DECODE_ERRORS')]
    elif samples:
        result['decoded_fps_log_mean'] = statistics.fmean(float(item[0]) for item in samples)
        result['errors_max'] = [max(int(item[i]) for item in samples) for i in (1, 2, 3)]
    tx_path = directory / 'sender.log'
    if tx_path.exists():
        tx_records = []
        for line in tx_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('QUALITY_TX '):
                tx_records.append({key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)})
        if tx_records:
            start = tx_records[0]['pts_us'] + warmup * 1e6
            tx_records = [item for item in tx_records if start <= item['pts_us'] <= start + duration * 1e6]
            if len(tx_records) > 1:
                elapsed = (tx_records[-1]['pts_us'] - tx_records[0]['pts_us']) / 1e6
                result['tx_measurement_seconds'] = elapsed
                result['tx_wire_kbps'] = (tx_records[-1]['wire_bytes'] - tx_records[0]['wire_bytes']) * .008 / elapsed
                result['tx_rtp_payload_kbps'] = (tx_records[-1]['rtp_payload_bytes'] - tx_records[0]['rtp_payload_bytes']) * .008 / elapsed
                result['tx_queue_max'] = max(item['queue'] for item in tx_records)
                result['tx_oldest_queue_age_ms'] = percentiles([item['oldest_queue_age_us'] / 1000
                    for item in tx_records if 'oldest_queue_age_us' in item])
                result['tx_drop_p'] = tx_records[-1]['drop_p'] - tx_records[0]['drop_p']
                result['tx_drop_idr'] = tx_records[-1]['drop_idr'] - tx_records[0]['drop_idr']
    # Explicit verification is separate from inferred log statistics: successful
    # worker output is not necessarily presented; receiver wire omits lost packets.
    evidence_path = directory / 'acceptance.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else {}
    required = ('wire_cap_verified', 'ordering_verified', 'bounded_queue_verified',
                'memory_stable', 'source_age_bounded', 'no_fallback_storm',
                'decoded_fps', 'unique_presented_gan_fps', 'gan_success_fraction',
                'end_to_end_ms', 'measurement_seconds', 'hardware', 'model', 'evidence_files')
    result['missing_evidence'] = [key for key in required if evidence.get(key) is None]
    result['acceptance'] = evidence
    if not result['missing_evidence']:
        flags = required[:6]
        numeric = evidence['decoded_fps'] >= config['fps'] * .9
        if config['enhancer'] != 'none':
            numeric &= evidence['unique_presented_gan_fps'] >= config['fps'] * .9 and evidence['gan_success_fraction'] >= .9
        files_exist = bool(evidence['evidence_files']) and all((directory / value).exists() for value in evidence['evidence_files'])
        evidence_hashes = evidence.get('evidence_sha256')
        hashes_valid = evidence_hashes is None or (files_exist and all(
            evidence_hashes.get(name) == hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in evidence['evidence_files']))
        result['evidence_hashes_verified'] = hashes_valid if evidence_hashes is not None else None
        observed_wire = result.get('tx_wire_kbps')
        e2e = evidence['end_to_end_ms']
        latency_keys = ('p50', 'p95', 'p99', 'max')
        latency_valid = isinstance(e2e, dict) and all(isinstance(e2e.get(key), (int, float))
            and math.isfinite(e2e[key]) and e2e[key] >= 0 for key in latency_keys)
        if latency_valid:
            latency_valid = [e2e[key] for key in latency_keys] == sorted(e2e[key] for key in latency_keys)
        worker_valid = config['enhancer'] == 'none' or result.get('worker_unique_ready_fps', 0) >= config['fps'] * .9
        result['status'] = 'PASS' if (all(evidence[key] is True for key in flags) and numeric and files_exist and hashes_valid
            and latency_valid and worker_valid
            and evidence['measurement_seconds'] >= duration and observed_wire is not None
            and 0 <= observed_wire <= config['cap'] and result.get('errors_max') == [0, 0, 0]
            and result.get('tx_drop_p') == result.get('tx_drop_idr') == 0
            and result['transport']['status'] == 'RECORDED'
            and result.get('tx_measurement_seconds', 0) >= duration - 1
            and config.get('status') == 'RECORDED') else 'FAIL'
    quality_path = directory / 'quality.json'
    result['quality'] = json.loads(quality_path.read_text(encoding='utf-8')) if quality_path.exists() else None
    result['utilization'] = result.get('tx_wire_kbps', 0) / config['cap'] if 'tx_wire_kbps' in result else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(path.parent) for path in sorted(args.root.rglob('config.json'))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'{len(results)} trials summarized; PASS={sum(item["status"] == "PASS" for item in results)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

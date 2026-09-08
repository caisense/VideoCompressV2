#!/usr/bin/env python3
"""Summarize a GAN enhancer JSONL run without inventing missing metrics."""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def numeric_field(records: list[dict[str, Any]], name: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            values.append(number)
    return values


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
        records.append(value)
    return records


def read_receiver_metrics(path: Path) -> dict[str, Any]:
    wire = []
    rtp = []
    caps = []
    pattern = re.compile(
        r"rtp_kbps=(?P<rtp>[0-9.]+).*?wire_kbps=(?P<wire>[0-9.]+).*?"
        r"link_cap_kbps=(?P<cap>[0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        rtp.append(float(match.group("rtp")))
        wire.append(float(match.group("wire")))
        caps.append(float(match.group("cap")))
    if not wire:
        return {"samples": 0, "h265_rtp_kbps": None, "wire_kbps": None,
                "link_cap_kbps": None}
    return {
        "samples": len(wire),
        "h265_rtp_kbps": {
            "mean": statistics.fmean(rtp), "p95": percentile(rtp, 95), "max": max(rtp),
        },
        "wire_kbps": {
            "mean": statistics.fmean(wire), "p95": percentile(wire, 95), "max": max(wire),
        },
        "link_cap_kbps": max(caps),
    }


def normalized_drop_reason(record: dict[str, Any]) -> str | None:
    reason = str(record.get("DROP_REASON", record.get("REASON", "")))
    return {
        "replaced_pending": "replaced_pending",
        "stale_before_infer": "stale_before",
        "stale_before_predicted": "stale_before_predicted",
        "stale_after_infer": "stale_after",
        "inference_error": "inference_error",
        "output_replaced": "output_replaced",
        "OUTPUT_TOO_OLD": "presentation_old",
        "PRESENTATION_SEQUENCE_OLD": "presentation_sequence_old",
        "RECOVERY_NOT_FRESH": "recovery_not_fresh",
        "RECOVERY_TOO_OLD": "recovery_too_old",
        "RECOVERY_SEQUENCE_OLD": "recovery_sequence_old",
        "RECOVERY_SEQUENCE_FUTURE": "recovery_sequence_future",
        "RECOVERY_SEQUENCE_UNKNOWN": "recovery_sequence_unknown",
    }.get(reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--receiver-log", type=Path, default=None,
                        help="optional headless receiver log for H.265/wire/link metrics")
    parser.add_argument("--latency-budget-ms", type=float, default=None)
    args = parser.parse_args()
    records = read_records(args.log)
    worker_records = [
        item for item in records
        if item.get("TYPE") in (None, "GAN_WORKER") and
        ("SEQ" in item or "OUTPUT_READY" in item)
    ]
    worker_drop_events = [
        item for item in records if item.get("TYPE") == "GAN_WORKER_DROP"
    ]
    presentation_events = [
        item for item in records if item.get("TYPE") == "GAN_PRESENTATION_DROP"
    ]
    state_events = [item for item in records if item.get("TYPE") == "GAN_STATE"]
    run_end_events = [item for item in records if item.get("TYPE") == "GAN_RUN_END"]
    hard_stall_events = [item for item in records if item.get("TYPE") == "GAN_HARD_STALL"]
    ready = [item for item in worker_records
             if item.get("OUTPUT_READY") and not item.get("DROPPED")]
    dropped = [item for item in worker_records if item.get("DROPPED")]
    inference_records = [
        item for item in worker_records
        if item.get("INFER_RETURNED") or (
            float(item.get("INFER_MS", 0.0) or 0.0) > 0.0 and
            item.get("DROP_REASON") not in (
                "stale_before_infer", "stale_before_predicted",
            )
        )
    ]
    latencies = [float(item["TOTAL_PC_MS"]) for item in ready
                 if item.get("TOTAL_PC_MS") is not None]
    all_total = [float(item["TOTAL_PC_MS"]) for item in inference_records
                 if item.get("TOTAL_PC_MS") is not None]
    infer = [float(item["INFER_MS"]) for item in inference_records
             if item.get("INFER_MS") is not None]
    dropped_after_infer = [
        float(item["TOTAL_PC_MS"]) for item in dropped
        if item.get("DROP_REASON") == "stale_after_infer" and
        item.get("TOTAL_PC_MS") is not None
    ]
    dropped_after_infer_infer = [
        float(item["INFER_MS"]) for item in dropped
        if item.get("DROP_REASON") == "stale_after_infer" and
        item.get("INFER_MS") is not None
    ]
    queue_wait = [float(item["QUEUE_WAIT"]) for item in worker_records
                  if item.get("QUEUE_WAIT") is not None]
    predicted_infer = [
        float(item["PREDICTED_INFER_MS"]) for item in worker_records
        if float(item.get("PREDICTED_INFER_MS", 0.0) or 0.0) > 0.0
    ]
    predicted_total = [
        float(item["PREDICTED_TOTAL_MS"]) for item in worker_records
        if float(item.get("PREDICTED_TOTAL_MS", 0.0) or 0.0) > 0.0
    ]
    raw_min_values = numeric_field(worker_records, "RAW_MIN")
    raw_max_values = numeric_field(worker_records, "RAW_MAX")
    output_luma_values = numeric_field(worker_records, "OUTPUT_LUMA_MEAN")
    postprocess_mode_counts = dict(collections.Counter(
        str(item["POSTPROCESS_MODE"])
        for item in worker_records
        if item.get("POSTPROCESS_MODE") not in (None, "")
    ))
    clipped_low_total = sum(
        int(float(item.get("CLIP_LOW_COUNT", 0) or 0))
        for item in worker_records
        if item.get("CLIP_LOW_COUNT") is not None
    )
    clipped_high_total = sum(
        int(float(item.get("CLIP_HIGH_COUNT", 0) or 0))
        for item in worker_records
        if item.get("CLIP_HIGH_COUNT") is not None
    )
    arrivals = [float(item["ARRIVAL"]) for item in worker_records
                if item.get("ARRIVAL") is not None]
    output_sequences = [int(item["SEQ"]) for item in ready if "SEQ" in item]
    event_times = [
        float(item["AT"]) for item in records
        if item.get("AT") is not None
    ]
    timeline = arrivals + event_times
    duration = max(timeline) - min(timeline) if len(timeline) >= 2 else 0.0
    backend = next((item.get("ENHANCER") for item in worker_records if item.get("ENHANCER")), None)
    provider = next((item.get("PROVIDER") for item in worker_records if item.get("PROVIDER")), None)
    source_fps_values = [
        float(item["INPUT_FPS"]) for item in worker_records
        if float(item.get("INPUT_FPS", 0.0) or 0.0) > 0.0
    ]
    source_fps = statistics.fmean(source_fps_values) if source_fps_values else None
    budget_values = [
        float(item["EFFECTIVE_BUDGET_MS"]) for item in worker_records
        if item.get("EFFECTIVE_BUDGET_MS") is not None
    ]
    output_budget_ms = (
        statistics.fmean(budget_values) if budget_values else args.latency_budget_ms
    )
    drop_reason_counts = {
        "replaced_pending": 0,
        "stale_before": 0,
        "stale_before_predicted": 0,
        "stale_after": 0,
        "inference_error": 0,
        "output_replaced": 0,
        "presentation_old": 0,
        "presentation_sequence_old": 0,
        "recovery_not_fresh": 0,
        "recovery_too_old": 0,
        "recovery_sequence_old": 0,
        "recovery_sequence_future": 0,
        "recovery_sequence_unknown": 0,
    }
    for item in dropped:
        reason = normalized_drop_reason(item)
        if reason is not None:
            drop_reason_counts[reason] += 1
    for item in worker_drop_events + presentation_events:
        reason = normalized_drop_reason(item)
        if reason is not None:
            drop_reason_counts[reason] += 1
    drop_denominator = len(worker_records) or 1
    drop_reason_percent = {
        reason: count * 100.0 / drop_denominator
        for reason, count in drop_reason_counts.items()
    }
    soft_events = [item for item in state_events if item.get("STATE") == "SOFT_FALLBACK"]
    recovered_events = [item for item in state_events if item.get("STATE") == "RECOVERED"]
    fallback_durations = [
        float(item["FALLBACK_DURATION_MS"])
        for item in recovered_events if item.get("FALLBACK_DURATION_MS") is not None
    ]
    fallback_durations.extend(
        float(item["FALLBACK_DURATION_MS"])
        for item in run_end_events
        if item.get("FALLBACK_ACTIVE") and item.get("FALLBACK_DURATION_MS") is not None
    )
    fallback_total_duration_ms = sum(fallback_durations)
    fallback_longest_duration_ms = max(fallback_durations) if fallback_durations else None
    if not fallback_durations and soft_events and recovered_events:
        paired = zip(soft_events, recovered_events)
        fallback_total_duration_ms = sum(
            max(0.0, float(recovered.get("AT", 0.0)) - float(start.get("AT", 0.0))) * 1000.0
            for start, recovered in paired
            if start.get("AT") is not None and recovered.get("AT") is not None
        )
    fallback_percent = (
        fallback_total_duration_ms * 100.0 / (duration * 1000.0)
        if duration > 0.0 else 0.0
    )
    stale_recovery_rejects = sum(
        1 for item in presentation_events if item.get("REASON") == "RECOVERY_NOT_FRESH"
    )
    recovery_rejection_reasons = dict(collections.Counter(
        str(item.get("REASON")) for item in presentation_events
        if str(item.get("REASON", "")).startswith("RECOVERY_")
    ))
    recovery_reset_events = [
        item for item in records if item.get("TYPE") == "GAN_RECOVERY_RESET"
    ]
    hard_run_ages = [
        float(item["RUN_AGE_MS"]) for item in hard_stall_events
        if item.get("RUN_AGE_MS") is not None
    ]
    result: dict[str, Any] = {
        "status": "PASS",
        "log": str(args.log),
        "records": len(records),
        "worker_records": len(worker_records),
        "backend": backend,
        "provider": provider,
        "input_fps_mean": source_fps,
        "source_fps": source_fps,
        "output_budget_ms": output_budget_ms,
        "duration_s": duration,
        "submitted": len(worker_records),
        "completed": len(ready),
        "inference_completed": len(inference_records),
        "worker_dropped": len(dropped) + len(worker_drop_events),
        "presentation_dropped": len(presentation_events),
        "dropped": len(dropped) + len(worker_drop_events),
        "drop_percent": (len(dropped) + len(worker_drop_events)) * 100.0 /
        len(worker_records) if worker_records else 0.0,
        "enhanced_fps": len(ready) / duration if duration > 0 else None,
        "unique_output_sequences": len(set(output_sequences)),
        "duplicate_output_sequences": len(output_sequences) - len(set(output_sequences)),
        "drop_reason_counts": drop_reason_counts,
        "drop_reason_percent": drop_reason_percent,
        "latency_ms": summary(latencies),
        "infer_ms": summary(infer),
        "infer_all_ms": summary(infer),
        "inference_all": summary(infer),
        "all_total_ms": summary(all_total),
        "successful_total_ms": summary(latencies),
        "dropped_after_infer_ms": summary(dropped_after_infer),
        "dropped_after_infer_infer_ms": summary(dropped_after_infer_infer),
        "postprocess_mode_counts": postprocess_mode_counts,
        "raw_output_range": {
            "min": min(raw_min_values) if raw_min_values else None,
            "max": max(raw_max_values) if raw_max_values else None,
        },
        "clipped_low_total": clipped_low_total,
        "clipped_high_total": clipped_high_total,
        "output_luma": summary(output_luma_values),
        "queue_wait_ms": {
            "p50": percentile(queue_wait, 50),
            "p95": percentile(queue_wait, 95),
            "max": max(queue_wait) if queue_wait else None,
        },
        "prediction": {
            "stale_before_predicted": drop_reason_counts["stale_before_predicted"],
            "predicted_infer_ms": summary(predicted_infer),
            "predicted_total_ms": summary(predicted_total),
        },
        "fallback": {
            "soft_fallback_entries": len(soft_events),
            "fallback_total_duration_ms": fallback_total_duration_ms,
            "fallback_longest_duration_ms": fallback_longest_duration_ms,
            "fallback_percent": fallback_percent,
            "recoveries": len(recovered_events),
        },
        "soft_fallback_entries": len(soft_events),
        "fallback_total_duration_ms": fallback_total_duration_ms,
        "fallback_longest_duration_ms": fallback_longest_duration_ms,
        "fallback_percent": fallback_percent,
        "recoveries": len(recovered_events),
        "hard_stall": {
            "count": len(hard_stall_events),
            "max_run_age_ms": max(hard_run_ages) if hard_run_ages else 0.0,
            "sequences": [item.get("SEQ") for item in hard_stall_events],
        },
        "hard_stall_count": len(hard_stall_events),
        "hard_stall_max_run_age_ms": max(hard_run_ages) if hard_run_ages else 0.0,
        "recovery_stale_rejects": stale_recovery_rejects,
        "recovery_rejection_reasons": recovery_rejection_reasons,
        "recovery_worker_resets": len(recovery_reset_events),
    }
    if args.receiver_log is not None:
        result["transport"] = read_receiver_metrics(args.receiver_log)
    if args.latency_budget_ms is not None:
        result["latency_budget_ms"] = args.latency_budget_ms
        result["latency_budget_pass"] = bool(
            not latencies or max(latencies) <= args.latency_budget_ms
        )
        if not result["latency_budget_pass"]:
            result["status"] = "FAIL"
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

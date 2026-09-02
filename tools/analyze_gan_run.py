#!/usr/bin/env python3
"""Summarize a GAN enhancer JSONL run without inventing missing metrics."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--receiver-log", type=Path, default=None,
                        help="optional headless receiver log for H.265/wire/link metrics")
    parser.add_argument("--latency-budget-ms", type=float, default=None)
    args = parser.parse_args()
    records = read_records(args.log)
    ready = [item for item in records if item.get("OUTPUT_READY") and not item.get("DROPPED")]
    dropped = [item for item in records if item.get("DROPPED")]
    latencies = [float(item["TOTAL_PC_MS"]) for item in ready if "TOTAL_PC_MS" in item]
    infer = [float(item["INFER_MS"]) for item in ready if "INFER_MS" in item]
    queue_wait = [float(item["QUEUE_WAIT"]) for item in records if "QUEUE_WAIT" in item]
    arrivals = [float(item["ARRIVAL"]) for item in records if "ARRIVAL" in item]
    output_sequences = [int(item["SEQ"]) for item in ready if "SEQ" in item]
    duration = max(arrivals) - min(arrivals) if len(arrivals) >= 2 else 0.0
    backend = next((item.get("ENHANCER") for item in records if item.get("ENHANCER")), None)
    provider = next((item.get("PROVIDER") for item in records if item.get("PROVIDER")), None)
    result: dict[str, Any] = {
        "status": "PASS",
        "log": str(args.log),
        "records": len(records),
        "backend": backend,
        "provider": provider,
        "input_fps_mean": (statistics.fmean(float(item["INPUT_FPS"])
                            for item in records if float(item.get("INPUT_FPS", 0)) > 0)
                           if any(float(item.get("INPUT_FPS", 0)) > 0 for item in records) else None),
        "duration_s": duration,
        "submitted": len(records),
        "completed": len(ready),
        "dropped": len(dropped),
        "drop_percent": len(dropped) * 100.0 / len(records) if records else 0.0,
        "enhanced_fps": len(ready) / duration if duration > 0 else None,
        "unique_output_sequences": len(set(output_sequences)),
        "duplicate_output_sequences": len(output_sequences) - len(set(output_sequences)),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else None,
        },
        "infer_ms": {
            "p50": percentile(infer, 50),
            "p95": percentile(infer, 95),
            "p99": percentile(infer, 99),
        },
        "queue_wait_ms": {
            "p50": percentile(queue_wait, 50),
            "p95": percentile(queue_wait, 95),
            "max": max(queue_wait) if queue_wait else None,
        },
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

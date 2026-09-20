#!/usr/bin/env python3
"""Align sender ENC_FRAME and receiver GAN_WORKER JSONL, then summarize GOP phase."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def records(path: Path, record_type: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if item.get("TYPE") == record_type:
                yield item


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def corr(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    pairs = [(number(row.get(left)), number(row.get(right))) for row in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "UNKNOWN" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", type=Path)
    parser.add_argument("gan", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    enc = list(records(args.encoder, "ENC_FRAME"))
    gan = list(records(args.gan, "GAN_WORKER"))
    # SEQ+GENERATION is the primary cross-host key. PTS is retained in the
    # merged output and used as a fallback when both logs expose the same unit.
    gan_by_key = {(r.get("SEQ"), r.get("GENERATION")): r for r in gan if r.get("INFER_RETURNED")}
    gan_by_pts = {(r.get("PTS"), r.get("GENERATION")): r for r in gan if r.get("INFER_RETURNED")}
    merged: list[dict[str, Any]] = []
    for sender in enc:
        receiver = gan_by_key.get((sender.get("SEQ"), sender.get("GENERATION")))
        if receiver is None:
            receiver = gan_by_pts.get((sender.get("PTS"), sender.get("GENERATION")))
        row = dict(sender)
        if receiver:
            row["DEC_SHARP"] = receiver.get("INPUT_LAPLACIAN_VARIANCE")
            row["GAN_SHARP"] = receiver.get("OUTPUT_LAPLACIAN_VARIANCE")
            row["DEC_EDGE"] = receiver.get("INPUT_EDGE_ENERGY")
            row["GAN_EDGE"] = receiver.get("OUTPUT_EDGE_ENERGY")
            row["GAN_CLIP_HIGH"] = receiver.get("CLIP_HIGH_COUNT")
            row["GAN_RAW_MAX"] = receiver.get("RAW_MAX")
        merged.append(row)

    phases: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        if isinstance(row.get("GOP_POSITION"), int):
            phases[row["GOP_POSITION"]].append(row)

    print("POS TYPE COUNT AVG_BITS P50_BITS P95_BITS AVG_QP ROI% DEC_SHARP GAN_SHARP")
    phase_output = []
    for position in sorted(phases):
        rows = phases[position]
        values = lambda key: [v for v in (number(r.get(key)) for r in rows) if v is not None]
        types = defaultdict(int)
        for row in rows:
            types[str(row.get("FRAME_TYPE", "UNKNOWN"))] += 1
        frame_type = max(types, key=types.get)
        entry = {
            "position": position, "type": frame_type, "count": len(rows),
            "avg_bits": mean(values("ENCODED_BITS")),
            "p50_bits": percentile(values("ENCODED_BITS"), 0.50),
            "p95_bits": percentile(values("ENCODED_BITS"), 0.95),
            "avg_qp": mean(values("AVG_QP")),
            "roi_percent": None if not values("ROI_AREA_RATIO") else mean(values("ROI_AREA_RATIO")) * 100.0,
            "decoded_sharpness": mean(values("DEC_SHARP")),
            "enhanced_sharpness": mean(values("GAN_SHARP")),
        }
        phase_output.append(entry)
        print(f"{position:>3} {frame_type:>4} {len(rows):>5} {fmt(entry['avg_bits']):>8} "
              f"{fmt(entry['p50_bits']):>8} {fmt(entry['p95_bits']):>8} {fmt(entry['avg_qp']):>6} "
              f"{fmt(entry['roi_percent']):>5} {fmt(entry['decoded_sharpness']):>9} "
              f"{fmt(entry['enhanced_sharpness']):>9}")

    correlations = {
        "gop_position_vs_decoded_sharpness": corr(merged, "GOP_POSITION", "DEC_SHARP"),
        "gop_position_vs_gan_sharpness": corr(merged, "GOP_POSITION", "GAN_SHARP"),
        "qp_vs_decoded_sharpness": corr(merged, "AVG_QP", "DEC_SHARP"),
        "encoded_bits_vs_decoded_sharpness": corr(merged, "ENCODED_BITS", "DEC_SHARP"),
        "roi_area_vs_decoded_sharpness": corr(merged, "ROI_AREA_RATIO", "DEC_SHARP"),
    }
    print("\nCORRELATIONS")
    for key, value in correlations.items():
        print(f"{key}={fmt(value, 4)}")
    print(f"matched_receiver_frames={sum('DEC_SHARP' in row for row in merged)}/{len(merged)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"phase": phase_output, "correlations": correlations,
                                         "sender_frames": len(enc), "gan_frames": len(gan)},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

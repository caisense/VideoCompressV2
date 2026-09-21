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


def slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    mx = statistics.fmean(x for x, _ in points)
    my = statistics.fmean(y for _, y in points)
    denominator = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", type=Path)
    parser.add_argument("gan", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
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
    decoded_phase = [(float(entry["position"]), entry["decoded_sharpness"])
                     for entry in phase_output if entry["decoded_sharpness"] is not None]
    qp_phase = [(float(entry["position"]), entry["avg_qp"])
                for entry in phase_output if entry["avg_qp"] is not None and entry["position"] > 0]
    decoded_values = [value for _, value in decoded_phase]
    early_values = [entry["decoded_sharpness"] for entry in phase_output
                    if 1 <= entry["position"] <= 3 and entry["decoded_sharpness"] is not None]
    late_values = [entry["decoded_sharpness"] for entry in phase_output
                   if 12 <= entry["position"] <= 15 and entry["decoded_sharpness"] is not None]
    pos0 = next((entry["decoded_sharpness"] for entry in phase_output
                 if entry["position"] == 0), None)
    i_bits = [number(row.get("ENCODED_BITS")) for row in merged if row.get("FRAME_TYPE") in ("IDR", "I")]
    p_bits = [number(row.get("ENCODED_BITS")) for row in merged if row.get("FRAME_TYPE") == "P"]
    i_bits = [value for value in i_bits if value is not None]
    p_bits = [value for value in p_bits if value is not None]
    p_mean = mean(p_bits)
    summary = {
        "BREATH_AMPLITUDE": max(decoded_values) - min(decoded_values) if decoded_values else None,
        "BREATH_STD": statistics.pstdev(decoded_values) if len(decoded_values) > 1 else None,
        "EARLY_P_TROUGH": min(early_values) if early_values else None,
        "POS0_SHARPNESS": pos0,
        "POS1_3_AVG": mean(early_values),
        "POS12_15_AVG": mean(late_values),
        "QP_PHASE_STD": statistics.pstdev([value for _, value in qp_phase]) if len(qp_phase) > 1 else None,
        "QP_PHASE_SLOPE": slope(qp_phase),
        "IDR_AVG_BITS": mean(i_bits),
        "P_AVG_BITS": p_mean,
        "IDR_P_BIT_RATIO": (mean(i_bits) / p_mean) if i_bits and p_mean else None,
    }
    print("\nBREATH_METRICS")
    for key, value in summary.items():
        print(f"{key}={fmt(value, 4)}")
    print("\nCORRELATIONS")
    for key, value in correlations.items():
        print(f"{key}={fmt(value, 4)}")
    print(f"matched_receiver_frames={sum('DEC_SHARP' in row for row in merged)}/{len(merged)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"phase": phase_output, "correlations": correlations,
                                         "summary": summary,
                                         "sender_frames": len(enc), "gan_frames": len(gan)},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plot:
        positions = [entry["position"] for entry in phase_output]
        series = [
            ("Decoded sharpness", [entry["decoded_sharpness"] for entry in phase_output], "#1f77b4"),
            ("Average QP", [entry["avg_qp"] for entry in phase_output], "#ff7f0e"),
            ("Encoded bits", [entry["avg_bits"] for entry in phase_output], "#2ca02c"),
        ]
        width, height, left, plot_width, panel_height = 900, 840, 100, 750, 210
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{args.encoder.stem}</text>']
        for panel, (label, values, color) in enumerate(series):
            top = 55 + panel * 260
            valid = [float(value) for value in values if value is not None]
            low, high = min(valid), max(valid)
            if high == low:
                high = low + 1.0
            points = []
            for position, value in zip(positions, values):
                if value is None:
                    continue
                x = left + plot_width * position / max(1, max(positions))
                y = top + panel_height - panel_height * (float(value) - low) / (high - low)
                points.append(f"{x:.1f},{y:.1f}")
            svg += [
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="#444"/>',
                f'<line x1="{left}" y1="{top + panel_height}" x2="{left + plot_width}" y2="{top + panel_height}" stroke="#444"/>',
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>',
                f'<text x="15" y="{top + panel_height / 2:.1f}" font-family="sans-serif" font-size="14">{label}</text>',
                f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-family="sans-serif" font-size="12">{high:.1f}</text>',
                f'<text x="{left - 8}" y="{top + panel_height}" text-anchor="end" font-family="sans-serif" font-size="12">{low:.1f}</text>',
            ]
        svg.append(f'<text x="450" y="825" text-anchor="middle" font-family="sans-serif" font-size="14">GOP position 0..{max(positions)}</text>')
        svg.append('</svg>')
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        args.plot.write_text("\n".join(svg), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

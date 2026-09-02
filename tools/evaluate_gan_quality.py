#!/usr/bin/env python3
"""Compute reproducible full-frame and optional fixed-ROI video metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    error = np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2)
    return 99.0 if error <= 1.0e-12 else 10.0 * math.log10((255.0 ** 2) / float(error))


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    mean_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mean_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    variance_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left * mean_left
    variance_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right * mean_right
    covariance = (cv2.GaussianBlur(left * right, (11, 11), 1.5) - mean_left * mean_right)
    c1 = 6.5025
    c2 = 58.5225
    score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2) /
             ((mean_left * mean_left + mean_right * mean_right + c1) *
              (variance_left + variance_right + c2)))
    return float(np.mean(score))


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_roi(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError("ROI must be x,y,width,height with positive width/height")
    return tuple(parts)  # type: ignore[return-value]


def crop(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return frame
    x, y, width, height = roi
    x = max(0, min(frame.shape[1] - 1, x))
    y = max(0, min(frame.shape[0] - 1, y))
    return frame[y:min(frame.shape[0], y + height), x:min(frame.shape[1], x + width)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="ground-truth/reference video")
    parser.add_argument("output", type=Path, help="captured GAN output video")
    parser.add_argument("--roi", default=None, help="optional fixed ROI x,y,width,height")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--result", type=Path, default=None)
    args = parser.parse_args()
    roi = parse_roi(args.roi)
    reference = cv2.VideoCapture(str(args.reference))
    output = cv2.VideoCapture(str(args.output))
    if not reference.isOpened() or not output.isOpened():
        parser.error("both reference and output videos must be readable")
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    sharpness_values: list[float] = []
    temporal_jumps: list[float] = []
    previous_output = None
    frames = 0
    while args.max_frames <= 0 or frames < args.max_frames:
        ok_reference, reference_frame = reference.read()
        ok_output, output_frame = output.read()
        if not ok_reference or not ok_output:
            break
        if reference_frame.shape[:2] != output_frame.shape[:2]:
            output_frame = cv2.resize(output_frame,
                                       (reference_frame.shape[1], reference_frame.shape[0]),
                                       interpolation=cv2.INTER_LANCZOS4)
        reference_roi = crop(reference_frame, roi)
        output_roi = crop(output_frame, roi)
        psnr_values.append(psnr(reference_roi, output_roi))
        ssim_values.append(ssim(reference_roi, output_roi))
        gray = cv2.cvtColor(output_roi, cv2.COLOR_BGR2GRAY)
        sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if previous_output is not None:
            temporal_jumps.append(float(np.mean(
                (output_roi.astype(np.float32) - previous_output.astype(np.float32)) ** 2)))
        previous_output = output_roi.copy()
        frames += 1
    reference.release()
    output.release()
    if frames == 0:
        parser.error("no paired frames were available")
    result = {
        "frames": frames,
        "roi": roi,
        "psnr_db": {"mean": statistics.fmean(psnr_values), "p50": percentile(psnr_values, 50),
                     "p95": percentile(psnr_values, 95)},
        "ssim": {"mean": statistics.fmean(ssim_values), "p50": percentile(ssim_values, 50),
                 "p95": percentile(ssim_values, 95)},
        "laplacian_variance": {"mean": statistics.fmean(sharpness_values),
                               "p50": percentile(sharpness_values, 50)},
        "temporal_jump_mse": {
            "mean": statistics.fmean(temporal_jumps) if temporal_jumps else None,
            "p95": percentile(temporal_jumps, 95) if temporal_jumps else None,
            "max": max(temporal_jumps) if temporal_jumps else None,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="", flush=True)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


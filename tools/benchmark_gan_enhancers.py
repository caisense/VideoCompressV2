#!/usr/bin/env python3
"""Benchmark a GAN full-frame backend with explicit CUDA/CPU reporting.

This script never labels a CPU run as an RTX result.  Missing models or an
inactive CUDA provider are reported as NOT RUN when requested by the caller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from full_frame_enhancer import FullFrameEnhancer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("none", "esrnet", "esrgan"), required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup calls excluded from the measured statistics")
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.measured <= 0 or args.threads < 0:
        parser.error("warmup must be non-negative, measured must be positive, threads non-negative")
    if args.backend != "none" and args.model is None:
        parser.error("--model is required for esrnet/esrgan")
    try:
        enhancer = FullFrameEnhancer(
            args.backend,
            model_path=args.model,
            require_cuda=args.require_cuda,
            threads=args.threads,
            warmup=0,
        )
        result = enhancer.benchmark(args.warmup, args.measured)
        result["status"] = "PASS"
    except (OSError, RuntimeError, ValueError) as error:
        result = {
            "status": "NOT RUN",
            "backend": args.backend,
            "model": None if args.model is None else str(args.model),
            "reason": str(error),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


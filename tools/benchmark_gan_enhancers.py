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
    parser.add_argument(
        "--execution-provider", choices=("auto", "cuda", "tensorrt-fp16"),
        default="auto",
    )
    parser.add_argument(
        "--input-size", choices=("256x144", "320x180", "640x360"),
        default="256x144",
        help="decoded source size; use 320x180 with scale-factor=4 for native 720p",
    )
    parser.add_argument("--scale-factor", type=int, choices=(2, 4), default=2,
                        help="model's native spatial scale (2 for x2, 4 for x4)")
    parser.add_argument("--trt-cache-dir", default="runs/tensorrt_cache")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.measured <= 0 or args.threads < 0:
        parser.error("warmup must be non-negative, measured must be positive, threads non-negative")
    if args.backend != "none" and args.model is None:
        parser.error("--model is required for esrnet/esrgan")
    if args.backend == "none" and args.execution_provider != "auto":
        parser.error("--execution-provider requires --backend=esrnet or esrgan")
    if args.require_cuda and args.execution_provider == "tensorrt-fp16":
        parser.error("--require-cuda conflicts with --execution-provider=tensorrt-fp16")
    input_size = tuple(int(part) for part in args.input_size.split("x"))
    native_size = (input_size[0] * args.scale_factor,
                   input_size[1] * args.scale_factor)
    output_size = native_size if native_size == (1280, 720) else (640, 360)
    execution_provider = (
        "cuda" if args.require_cuda and args.execution_provider == "auto"
        else args.execution_provider
    )
    try:
        enhancer = FullFrameEnhancer(
            args.backend,
            model_path=args.model,
            input_size=input_size,
            native_size=native_size,
            output_size=output_size,
            require_cuda=args.require_cuda,
            execution_provider=execution_provider,
            tensorrt_cache_dir=args.trt_cache_dir,
            threads=args.threads,
            warmup=0,
        )
        result = enhancer.benchmark(args.warmup, args.measured)
        result["scale_factor"] = args.scale_factor
        result["status"] = "PASS"
    except (OSError, RuntimeError, ValueError) as error:
        result = {
            "status": "NOT RUN",
            "backend": args.backend,
            "model": None if args.model is None else str(args.model),
            "execution_mode": execution_provider,
            "execution_precision": (
                "TensorRT FP16 requested" if execution_provider == "tensorrt-fp16"
                else "CUDA requested" if execution_provider == "cuda"
                else "ONNX Runtime auto"
            ),
            "input": list(input_size),
            "native": list(native_size),
            "output": list(output_size),
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


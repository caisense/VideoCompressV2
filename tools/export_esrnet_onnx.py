#!/usr/bin/env python3
"""Export a real Real-ESRNet x2 checkpoint to the GAN ONNX contract.

This is an exporter, not a model substitute: it refuses to copy or rename the
Real-ESRGAN asset.  A checkpoint containing only a state dict must be paired
with a factory that constructs the matching Real-ESRNet architecture.

Example::

    python tools/export_esrnet_onnx.py \
      --checkpoint RealESRNet_x2plus.pth \
      --factory my_esrnet_factory:create_model \
      --output model/RealESRNet_x2_dynamic.onnx
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("--factory must be module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if factory is None:
        raise ValueError(f"factory not found: {spec}")
    return factory


def checkpoint_state(checkpoint: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(checkpoint, Mapping):
        for key in ("params_ema", "params", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    return checkpoint if isinstance(checkpoint, Mapping) else None


def first_tensor(value: Any):
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for ESRNet export") from error
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--factory", default=None,
                        help="module:function that returns the matching ESRNet model")
    parser.add_argument("--factory-args", default="{}",
                        help="JSON object passed as keyword arguments to --factory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=144)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.opset < 11 or args.input_width <= 0 or args.input_height <= 0:
        parser.error("opset must be >= 11 and input dimensions must be positive")
    try:
        factory_args = json.loads(args.factory_args)
        if not isinstance(factory_args, dict):
            raise ValueError("--factory-args must be a JSON object")
    except json.JSONDecodeError as error:
        parser.error(f"invalid --factory-args JSON: {error}")

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit("PyTorch is required for ESRNet export") from error

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    model = checkpoint if isinstance(checkpoint, torch.nn.Module) else None
    if model is None and isinstance(checkpoint, Mapping):
        candidate = checkpoint.get("model")
        if isinstance(candidate, torch.nn.Module):
            model = candidate
    if model is None:
        if args.factory is None:
            raise SystemExit(
                "checkpoint contains no serialized torch.nn.Module; provide --factory module:function"
            )
        model = load_factory(args.factory)(**factory_args)
        state = checkpoint_state(checkpoint)
        if state is None:
            raise SystemExit("checkpoint has no state dict for the supplied ESRNet factory")
        # Common training wrappers prefix keys with ``module.``.  Try the
        # original state first, then the unwrapped spelling, and fail closed.
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            unwrapped = {
                key[7:] if key.startswith("module.") else key: value
                for key, value in state.items()
            }
            model.load_state_dict(unwrapped, strict=True)
    model = model.eval().cpu()
    dummy = torch.zeros(1, 3, args.input_height, args.input_width, dtype=torch.float32)
    with torch.no_grad():
        output = first_tensor(model(dummy))
    if output is None or output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
        raise SystemExit(f"ESRNet factory output must be NCHW 3-channel, got {None if output is None else tuple(output.shape)}")
    expected = (args.input_height * 2, args.input_width * 2)
    if tuple(output.shape[2:]) != expected:
        raise SystemExit(
            f"ESRNet x2 contract failed: expected NCHW spatial {expected}, got {tuple(output.shape[2:])}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("ONNX export produced no file")
    print(
        f"exported Real-ESRNet x2 ONNX: {args.output} "
        f"input={args.input_width}x{args.input_height} output={args.input_width * 2}x{args.input_height * 2}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


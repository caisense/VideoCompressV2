#!/usr/bin/env python3
"""PC-side full-frame enhancement for the H.265-only GAN profile.

The module deliberately has no knowledge of RB/1, semantic masks, reference
JPEGs, registration, or ROI compositing.  Every backend consumes the same
decoded 256x144 BGR frame and produces a native 512x288 frame before the
shared 640x360 Lanczos presentation resize.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import site
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Optional, Tuple

import cv2
import numpy as np


Size = Tuple[int, int]  # width, height
_CUDA_DLL_HANDLES = []
_CUDA_DLL_PATHS = set()


def _prepare_cuda_runtime_paths() -> None:
    """Expose pip-installed CUDA/cuDNN DLLs to the Windows loader."""
    if os.name != "nt":
        return
    try:
        roots = list(site.getsitepackages())
    except AttributeError:
        roots = []
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)
    libraries = (
        "cuda_runtime", "cuda_nvrtc", "cublas", "cudnn", "cufft", "curand", "nvjitlink"
    )
    directories = []
    for root in roots:
        for library in libraries:
            directory = Path(root) / "nvidia" / library / "bin"
            if directory.is_dir():
                directories.append(directory)
    if not directories:
        return
    current = os.environ.get("PATH", "")
    known = {os.path.normcase(os.path.normpath(entry))
             for entry in current.split(os.pathsep) if entry}
    additions = []
    for directory in directories:
        value = str(directory)
        normalized = os.path.normcase(os.path.normpath(value))
        if normalized not in known:
            additions.append(value)
            known.add(normalized)
        if hasattr(os, "add_dll_directory"):
            try:
                if normalized not in _CUDA_DLL_PATHS:
                    _CUDA_DLL_HANDLES.append(os.add_dll_directory(value))
                    _CUDA_DLL_PATHS.add(normalized)
            except (OSError, AttributeError):
                pass
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + ([current] if current else []))


def _percentile(values: Deque[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


@dataclasses.dataclass(frozen=True)
class EnhancementInput:
    frame: np.ndarray
    source_sequence: int
    rtp_timestamp: Optional[int]
    arrived_at: float
    input_fps: float


@dataclasses.dataclass(frozen=True)
class EnhancementOutput:
    frame: np.ndarray
    source_sequence: int
    rtp_timestamp: Optional[int]
    arrived_at: float
    completed_at: float
    queue_wait_ms: float
    infer_ms: float
    total_pc_ms: float
    enhancer: str
    provider: str


class FullFrameEnhancer:
    """Common preprocessing/postprocessing wrapper for none/ESRNet/ESRGAN."""

    VALID_BACKENDS = ("none", "esrnet", "esrgan")

    def __init__(
        self,
        backend: str,
        model_path: Optional[str | Path] = None,
        input_size: Size = (256, 144),
        native_size: Size = (512, 288),
        output_size: Size = (640, 360),
        require_cuda: bool = False,
        threads: int = 2,
        warmup: int = 4,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"unknown full-frame enhancer: {backend}")
        if any(int(value) <= 0 for value in (*input_size, *native_size, *output_size)):
            raise ValueError("enhancer dimensions must be positive")
        if threads < 0 or warmup < 0:
            raise ValueError("enhancer threads/warmup must be non-negative")
        self.backend = backend
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.native_size = (int(native_size[0]), int(native_size[1]))
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.require_cuda = bool(require_cuda)
        self.session = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.provider = "Lanczos4" if backend == "none" else "CPUExecutionProvider"
        self.providers = (self.provider,)
        self._logger = logger or (lambda message: print(message, flush=True))

        if backend != "none":
            if model_path is None:
                raise ValueError(f"--gan-{backend}-model is required for {backend}")
            self.model_path = Path(model_path).expanduser()
            if not self.model_path.is_file():
                raise FileNotFoundError(f"{backend} ONNX model not found: {self.model_path}")
            self._load_onnx(threads)
        else:
            self.model_path = None

        if warmup:
            self.warmup(warmup)

    def _load_onnx(self, threads: int) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("onnxruntime is required for ESRNet/ESRGAN") from error

        _prepare_cuda_runtime_paths()
        available = tuple(ort.get_available_providers())
        if self.require_cuda and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "--gan-require-cuda requested, but CUDAExecutionProvider is unavailable; "
                f"available={list(available)}"
            )
        if "CUDAExecutionProvider" in available:
            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                try:
                    preload(directory="")
                except Exception as error:
                    self._logger(f"GAN CUDA DLL preload unavailable: {error}")
        cuda_requested = "CUDAExecutionProvider" in available
        # Probe CUDA in an EP-only session.  If CPU is listed in the same
        # session, ORT can silently retry a failed CUDA kernel on CPU while
        # still reporting CUDA first in get_providers().
        requested = ["CUDAExecutionProvider"] if cuda_requested else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=requested
            )
        except Exception as error:
            if self.require_cuda and cuda_requested:
                raise RuntimeError(
                    "--gan-require-cuda requested, but CUDA session creation failed: "
                    f"{error}"
                ) from error
            if not cuda_requested:
                raise RuntimeError(f"full-frame enhancer session creation failed: {error}") from error
            self._logger(
                f"GAN CUDA session creation failed; falling back to CPUExecutionProvider: {error}"
            )
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        active = tuple(self.session.get_providers())
        if self.require_cuda and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "--gan-require-cuda requested, but ONNX Runtime did not activate CUDA; "
                f"active={list(active)}"
            )
        self.providers = active
        self.provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in active
            else "CPUExecutionProvider"
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("full-frame ONNX enhancer must have one input and one output")
        input_info = inputs[0]
        output_info = outputs[0]
        if getattr(input_info, "type", None) != "tensor(float)":
            raise ValueError(
                f"full-frame enhancer input must be float32, got {getattr(input_info, 'type', None)}"
            )
        if getattr(output_info, "type", None) not in ("tensor(float)", "tensor(float16)"):
            raise ValueError(
                f"full-frame enhancer output must be float/float16, got {getattr(output_info, 'type', None)}"
            )
        if len(input_info.shape) != 4 or input_info.shape[1] not in (3, "3"):
            raise ValueError(f"expected NCHW RGB input with three channels, got {input_info.shape}")
        height, width = self.input_size[1], self.input_size[0]
        if input_info.shape[2] not in (None, "height", "H", height):
            raise ValueError(f"model input height {input_info.shape[2]} != {height}")
        if input_info.shape[3] not in (None, "width", "W", width):
            raise ValueError(f"model input width {input_info.shape[3]} != {width}")
        if len(output_info.shape) != 4 or output_info.shape[1] not in (3, "3"):
            raise ValueError(f"expected NCHW RGB output with three channels, got {output_info.shape}")
        self.input_name = input_info.name
        self.output_name = output_info.name

        # Some Windows ORT/CUDA installations report a CUDA EP during session
        # construction but fail only when the first kernel is executed (for
        # example, a missing cuDNN engine DLL).  Probe the real live tensor
        # before allowing the worker to advertise that provider.
        probe = np.zeros((1, 3, self.input_size[1], self.input_size[0]), dtype=np.float32)
        try:
            probe_output = self.session.run([self.output_name], {self.input_name: probe})[0]
            self._postprocess_native(probe_output)
        except Exception as error:
            if self.require_cuda and "CUDAExecutionProvider" in active:
                raise RuntimeError(
                    "--gan-require-cuda requested, but CUDA inference probe failed: "
                    f"{error}"
                ) from error
            if "CUDAExecutionProvider" not in active:
                raise RuntimeError(f"full-frame enhancer inference probe failed: {error}") from error
            self._logger(
                f"GAN CUDA inference probe failed; falling back to CPUExecutionProvider: {error}"
            )
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            active = tuple(self.session.get_providers())
            if "CPUExecutionProvider" not in active:
                raise RuntimeError(
                    "CUDA probe failed and CPUExecutionProvider is unavailable; "
                    f"active={list(active)}"
                )
            probe_output = self.session.run([self.output_name], {self.input_name: probe})[0]
            self._postprocess_native(probe_output)
            self.providers = active
            self.provider = "CPUExecutionProvider"

        self._logger(
            f"GAN enhancer provider selected: {self.provider}; backend={self.backend}; "
            f"model={self.model_path}; input={input_info.name}:{input_info.type}{input_info.shape}; "
            f"output={output_info.name}:{output_info.type}{output_info.shape}; "
            f"active_providers={list(active)}"
        )

    def warmup(self, count: int = 4) -> None:
        if count <= 0:
            return
        synthetic = np.zeros(
            (self.input_size[1], self.input_size[0], 3), dtype=np.uint8
        )
        for index in range(count):
            # Keep the warmup shape/type identical to a live decoded frame.
            synthetic[:, :, 0] = (index * 17) & 0xFF
            self.enhance(synthetic)
        self._logger(f"GAN enhancer warmup complete: {count} frames; provider={self.provider}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("enhancer input must be a BGR HxWx3 numpy array")
        if (frame.shape[1], frame.shape[0]) != self.input_size:
            frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor / 255.0

    def _postprocess_native(self, output: np.ndarray) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
            raise ValueError(f"enhancer output must be NCHW with batch=1/channels=3, got {array.shape}")
        array = array[0].transpose(1, 2, 0)
        if array.shape[1] != self.native_size[0] or array.shape[0] != self.native_size[1]:
            raise ValueError(
                f"enhancer native output must be {self.native_size[0]}x{self.native_size[1]}, "
                f"got {array.shape[1]}x{array.shape[0]}"
            )
        array = array.astype(np.float32, copy=False)
        minimum = float(np.nanmin(array))
        maximum = float(np.nanmax(array))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("enhancer output contains NaN/Inf")
        if minimum < 0.0 and minimum >= -1.1 and maximum <= 1.1:
            array = (array + 1.0) * 127.5
        elif maximum <= 1.5:
            array = array * 255.0
        elif maximum > 255.0:
            raise ValueError(f"unsupported enhancer output range [{minimum}, {maximum}]")
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    def enhance_native(self, frame: np.ndarray) -> np.ndarray:
        if self.backend == "none":
            resized = cv2.resize(frame, self.native_size, interpolation=cv2.INTER_LANCZOS4)
            return np.ascontiguousarray(resized)
        if self.session is None or self.input_name is None or self.output_name is None:
            raise RuntimeError("ONNX enhancer session is not initialized")
        tensor = self._preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self._postprocess_native(output)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        native = self.enhance_native(frame)
        if (native.shape[1], native.shape[0]) != self.output_size:
            native = cv2.resize(native, self.output_size, interpolation=cv2.INTER_LANCZOS4)
        return np.ascontiguousarray(native)

    def benchmark(self, warmup: int = 10, measured: int = 200) -> dict:
        if warmup < 0 or measured <= 0:
            raise ValueError("benchmark requires warmup >= 0 and measured > 0")
        frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        for _ in range(warmup):
            self.enhance(frame)
        samples = []
        for _ in range(measured):
            started = time.perf_counter()
            self.enhance(frame)
            samples.append((time.perf_counter() - started) * 1000.0)
        values = collections.deque(samples)
        return {
            "backend": self.backend,
            "provider": self.provider,
            "warmup": warmup,
            "measured": measured,
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
            "p99_ms": _percentile(values, 99),
            "mean_ms": statistics.fmean(samples),
            "input": list(self.input_size),
            "native": list(self.native_size),
            "output": list(self.output_size),
        }


class LatestOnlyEnhancerWorker:
    """One running item plus one replaceable pending item.

    A frame that is too old before or after inference is dropped.  The worker
    never queues an unbounded backlog and stores at most one unconsumed output,
    so display latency remains bounded even when the model is slower than the
    encoded source cadence.
    """

    def __init__(
        self,
        enhancer: Any,
        max_latency_ms: int = 100,
        debug_log: Optional[str | Path] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if max_latency_ms <= 0:
            raise ValueError("max enhancer latency must be positive")
        self.enhancer = enhancer
        self.max_latency_ms = float(max_latency_ms)
        self.logger = logger or (lambda message: print(message, flush=True))
        self.provider = getattr(enhancer, "provider", "unknown")
        self.backend = getattr(enhancer, "backend", "unknown")
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.pending: Optional[EnhancementInput] = None
        self.output: Optional[EnhancementOutput] = None
        self.running = False
        self.running_started_at: Optional[float] = None
        self.running_sequence: Optional[int] = None
        self.stopping = False
        self.closed = False
        self.error: Optional[str] = None
        self.thread = threading.Thread(target=self._run, name="gan-enhancer", daemon=True)
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.replaced_pending = 0
        self.stale_before_infer = 0
        self.stale_after_infer = 0
        self.output_replaced = 0
        self.latencies: Deque[float] = collections.deque(maxlen=2000)
        self.queue_waits: Deque[float] = collections.deque(maxlen=2000)
        self.infer_latencies: Deque[float] = collections.deque(maxlen=2000)
        self.completion_times: Deque[float] = collections.deque()
        self.last_output: Optional[EnhancementOutput] = None
        self.debug_handle = None
        if debug_log is not None:
            path = Path(debug_log).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_handle = path.open("a", encoding="utf-8", buffering=1)
        self.thread.start()

    def _debug(self, item: EnhancementInput, *, queue_wait_ms: float = 0.0,
               infer_ms: float = 0.0, total_pc_ms: float = 0.0,
               dropped: bool = False, drop_reason: str = "") -> None:
        if self.debug_handle is None:
            return
        record = {
            "SEQ": item.source_sequence,
            "PTS": item.rtp_timestamp,
            "ARRIVAL": item.arrived_at,
            "INPUT_FPS": item.input_fps,
            "ENHANCER": self.backend,
            "PROVIDER": self.provider,
            "QUEUE_WAIT": queue_wait_ms,
            "INFER_MS": infer_ms,
            "TOTAL_PC_MS": total_pc_ms,
            "DROPPED": bool(dropped),
            "DROP_REASON": drop_reason,
            "OUTPUT_READY": not dropped,
        }
        self.debug_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def submit(
        self,
        frame: np.ndarray,
        source_sequence: int,
        rtp_timestamp: Optional[int] = None,
        arrived_at: Optional[float] = None,
        input_fps: float = 0.0,
    ) -> bool:
        if not isinstance(frame, np.ndarray):
            raise ValueError("enhancer worker frame must be a numpy array")
        arrived_at = time.monotonic() if arrived_at is None else float(arrived_at)
        immutable = np.ascontiguousarray(frame.copy())
        immutable.setflags(write=False)
        item = EnhancementInput(
            frame=immutable,
            source_sequence=int(source_sequence),
            rtp_timestamp=rtp_timestamp,
            arrived_at=arrived_at,
            input_fps=float(input_fps),
        )
        with self.condition:
            if self.stopping:
                return False
            self.submitted += 1
            if self.pending is not None:
                self.replaced_pending += 1
                self.dropped += 1
                self._debug(self.pending, dropped=True, drop_reason="replaced_pending")
            self.pending = item
            self.condition.notify()
            return True

    def poll_output(self) -> Optional[EnhancementOutput]:
        with self.lock:
            output = self.output
            self.output = None
            return output

    def _drop(self, item: EnhancementInput, reason: str, queue_wait_ms: float = 0.0,
              infer_ms: float = 0.0, total_pc_ms: float = 0.0) -> None:
        with self.lock:
            self.dropped += 1
            if reason == "stale_before_infer":
                self.stale_before_infer += 1
            elif reason == "stale_after_infer":
                self.stale_after_infer += 1
        self._debug(item, queue_wait_ms=queue_wait_ms, infer_ms=infer_ms,
                    total_pc_ms=total_pc_ms, dropped=True, drop_reason=reason)

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.pending is not None)
                if self.stopping and self.pending is None and not self.running:
                    break
                item = self.pending
                self.pending = None
            if item is None:
                continue
            started = time.monotonic()
            with self.condition:
                self.running = True
                self.running_started_at = started
                self.running_sequence = item.source_sequence
            queue_wait_ms = (started - item.arrived_at) * 1000.0
            if queue_wait_ms > self.max_latency_ms:
                self._drop(item, "stale_before_infer", queue_wait_ms=queue_wait_ms)
                with self.condition:
                    self.running = False
                    self.running_started_at = None
                    self.running_sequence = None
                    self.condition.notify_all()
                continue
            infer_started = time.perf_counter()
            try:
                enhanced = self.enhancer.enhance(item.frame)
                infer_ms = (time.perf_counter() - infer_started) * 1000.0
                completed_at = time.monotonic()
                total_pc_ms = (completed_at - item.arrived_at) * 1000.0
                if total_pc_ms > self.max_latency_ms:
                    self._drop(item, "stale_after_infer", queue_wait_ms,
                               infer_ms, total_pc_ms)
                else:
                    output = EnhancementOutput(
                        frame=np.ascontiguousarray(enhanced),
                        source_sequence=item.source_sequence,
                        rtp_timestamp=item.rtp_timestamp,
                        arrived_at=item.arrived_at,
                        completed_at=completed_at,
                        queue_wait_ms=queue_wait_ms,
                        infer_ms=infer_ms,
                        total_pc_ms=total_pc_ms,
                        enhancer=self.backend,
                        provider=self.provider,
                    )
                    with self.condition:
                        if self.output is not None:
                            self.output_replaced += 1
                            self.dropped += 1
                        self.output = output
                        self.last_output = output
                        self.completed += 1
                        self.latencies.append(total_pc_ms)
                        self.queue_waits.append(queue_wait_ms)
                        self.infer_latencies.append(infer_ms)
                        self.completion_times.append(completed_at)
                    self._debug(item, queue_wait_ms=queue_wait_ms, infer_ms=infer_ms,
                                total_pc_ms=total_pc_ms)
            except Exception as error:  # keep the RTP/decoder process alive for diagnostics
                message = f"{type(error).__name__}: {error}"
                with self.lock:
                    self.error = message
                self.logger(f"GAN enhancer worker error: {message}")
                self._drop(item, "inference_error", queue_wait_ms)
            finally:
                with self.condition:
                    self.running = False
                    self.running_started_at = None
                    self.running_sequence = None
                    self.condition.notify_all()
        with self.lock:
            self.closed = True

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            while self.completion_times and self.completion_times[0] < now - 10.0:
                self.completion_times.popleft()
            rolling_1s = sum(timestamp >= now - 1.0
                             for timestamp in self.completion_times)
            rolling_5s = sum(timestamp >= now - 5.0
                             for timestamp in self.completion_times)
            running_age_ms = (0.0 if self.running_started_at is None else
                              max(0.0, (now - self.running_started_at) * 1000.0))
            return {
                "backend": self.backend,
                "provider": self.provider,
                "submitted": self.submitted,
                "completed": self.completed,
                "dropped": self.dropped,
                "drop_percent": (self.dropped * 100.0 / self.submitted
                                  if self.submitted else 0.0),
                "replaced_pending": self.replaced_pending,
                "stale_before_infer": self.stale_before_infer,
                "stale_after_infer": self.stale_after_infer,
                "output_replaced": self.output_replaced,
                "p50_ms": _percentile(self.latencies, 50),
                "p95_ms": _percentile(self.latencies, 95),
                "p99_ms": _percentile(self.latencies, 99),
                "last_infer_ms": (self.last_output.infer_ms if self.last_output else 0.0),
                "last_total_pc_ms": (self.last_output.total_pc_ms if self.last_output else 0.0),
                "last_queue_wait_ms": (self.last_output.queue_wait_ms if self.last_output else 0.0),
                "rolling_1s_fps": float(rolling_1s),
                "rolling_5s_fps": rolling_5s / 5.0,
                "rolling_10s_fps": len(self.completion_times) / 10.0,
                "running": self.running,
                "running_age_ms": running_age_ms,
                "running_sequence": self.running_sequence,
                "pending": self.pending is not None,
                "output_ready": self.output is not None,
                "max_latency_ms": self.max_latency_ms,
                "error": self.error,
                "closed": self.closed,
            }

    def stop(self) -> None:
        with self.condition:
            if self.stopping:
                thread = self.thread
            else:
                self.stopping = True
                self.pending = None
                self.condition.notify_all()
                thread = self.thread
        if thread.is_alive():
            thread.join(timeout=max(2.0, self.max_latency_ms / 1000.0 + 2.0))
        if self.debug_handle is not None:
            self.debug_handle.close()
            self.debug_handle = None

    close = stop

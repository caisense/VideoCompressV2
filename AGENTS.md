# Repository Guidelines

## Project Scope

VideoCompressV2 is a low-bandwidth, low-latency video transmission and enhancement system centered on an RK3588 sender and PC/RK3588 receivers. The sender handles capture, YOLOv8-Seg inference, ROI generation, H.265 encoding, transport, audio, and preview. The Windows receiver handles H.265 decode, HUD/telemetry, optional reconstruction, and GAN enhancement.

End-to-end latency, physical wire bitrate, frame ordering, and temporal quality stability are system-level constraints. Optimize the complete pipeline rather than an isolated function.

## Project Structure & Module Organization

The RK3588 sender lives in `cpp/`. Modules are grouped by responsibility: `capture/`, `inference/`, `roi/`, `encoder/`, `transport/`, `audio/`, and `preview/`; `main.cc` wires the pipeline together. Native unit tests are in `cpp/tests/`.

`board_receiver/` contains the optional RK3588 receiver application and its tests. Windows/Python receiver, enhancement, telemetry, benchmark, and experiment utilities are under `tools/`. Models and labels belong in `model/`; vendor libraries and headers are under `cpp/3rdparty/`.

Treat `artifacts/`, `runs/`, `cpp/build/`, install directories, accelerator caches, generated logs, screenshots, and benchmark outputs as generated artifacts, not source.

Before editing, inspect the current implementation and nearby tests. Do not assume an older plan, prior chat, issue description, or public repository snapshot exactly matches the working tree.

## Build, Test, and Development Commands

Build the sender from the Ubuntu cross-compilation environment:

```bash
cd cpp
ROI_CODEC2_SOURCE_DIR=/path/to/codec2-1.2.0 ./build-linux.sh
```

The script targets RK3588 with `/opt/atk-dlrk3588-toolchain` and installs to `cpp/install/rk3588_linux_aarch64/atk_rknn_yolov8_seg_cam/`.

Run host-native C++ regression tests with:

```bash
cd cpp && ./tests/build_roi_tests.sh
```

Build the standalone board receiver with:

```bash
cd board_receiver && ./build-linux.sh
```

Validate Python changes in the intended Conda environment:

```powershell
conda run -n videocompress python -m py_compile tools\live_h265_hud.py
```

When environment or dependency issues are relevant, verify the actual interpreter used by the receiver. Prefer `python -m pip` or `conda run -n videocompress ...`; do not assume `python`, `pip`, and the receiver process resolve to the same environment.

## Coding Style & Naming Conventions

Use C++11, four-space indentation, braces on the same line, and existing RAII patterns. Types use `PascalCase`; functions and variables use `camelCase` or the surrounding module's established style; private members end in `_`. Keep headers self-contained and namespace code under `roi_h265`.

Python follows PEP 8 with `snake_case`.

Prefer focused configuration flags and preserve existing profile defaults unless the current task explicitly changes them. Do not introduce a new flag when an existing configuration mechanism already expresses the behavior cleanly.

Keep hot-path logging lightweight. High-frequency per-frame diagnostics should be optional, structured, or rate-limited outside benchmark/debug runs.

## Compatibility & Change Discipline

Existing modes, profiles, and defaults are compatibility-sensitive. Do not silently redefine an existing mode while implementing a new experiment or backend.

Keep changes scoped to the requested goal. Before changing shared behavior:

1. identify the current call path and configuration source;
2. identify existing defaults and tests;
3. preserve backward-compatible behavior unless the task explicitly requests a breaking change;
4. add diagnostics before making performance claims;
5. validate the actual runtime path, not just configuration strings.

Do not treat CLI flags, HUD text, or configuration placeholders alone as a completed implementation. A feature is complete only when the requested runtime behavior is active and observable.

Avoid silent fallback. If an accelerator, model, device, or provider is unavailable, fail clearly or enter an explicitly labeled fallback state according to the mode's contract.

## Low-Latency Receiver & GAN Rules

Receiver enhancement paths must remain latency-bounded. Do not introduce unbounded frame queues to improve apparent throughput.

For asynchronous inference or multi-worker processing, prefer latest-oriented bounded buffering: keep only the work that can still affect near-real-time output, replace stale pending work when appropriate, and avoid long FIFO backlogs.

If workers can finish out of order, restore presentation order by sequence/PTS with a bounded reorder policy. Do not wait indefinitely for a missing or slow frame, and do not let one stalled worker freeze the whole receiver.

For multi-GPU inference, use independent per-device inference contexts/sessions unless the implementation explicitly requires another architecture. Do not assume separate GPU memories form one shared memory pool.

## Backend, Precision & HUD Semantics

Keep these concepts separate in code, telemetry, and documentation:

- model storage/input dtype;
- inference backend/provider;
- execution precision;
- fallback backend.

For example, an FP32 ONNX model may still execute supported layers through an FP16-optimized backend. Do not infer runtime precision only from the model file type.

HUD and logs must describe the backend that is actually running. If an optimized backend falls back to CUDA or CPU, expose that truthfully instead of continuing to display the requested backend as if it were active.

When image scaling is involved, distinguish:

- decoded/source resolution;
- AI-native output resolution;
- final presentation/display resolution.

Do not describe a conventional post-scale as native GAN restoration.

## Transport & Bitrate Invariants

The project's bandwidth constraint is the physical wire rate, not only the H.265 elementary-stream bitrate. Do not conflate:

- H.265 payload bitrate;
- RTP/UDP payload bitrate;
- physical Ethernet wire rate;
- total audio/video wire rate.

Audio and video may share the same physical cap. Receiver-side enhancement work must not silently change sender bitrate, GOP, ROI allocation, or audio reservation unless the current task explicitly requires a sender-side change.

When a change claims to respect a bitrate cap, validate the measured physical wire rate, not only the encoder target bitrate.

## Encoder / GOP / Temporal Quality

Periodic blur or sharpness oscillation must be diagnosed from evidence, not assumed to be a GAN problem.

When changing GOP, rate control, I/P allocation, ROI QP, super-frame, re-encode, or related encoder settings, record the relevant frame type, frame bits, QP, sequence/GOP position, receive FPS, and physical wire rate where available. Compare temporal/GOP-phase behavior as well as average quality.

Avoid changing several independent rate-control variables at once unless the experiment is explicitly designed for that purpose. A local quality improvement is not acceptable if it violates the end-to-end bitrate or receive-FPS constraints.

## Testing & Benchmarking

Name native tests `test_<feature>.cc` and add them to `build_roi_tests.sh`. Tests compile with `-Wall -Wextra -Werror`, so warnings are failures.

For transport, rate-control, or GAN changes, include real board-to-PC evidence when hardware behavior matters: receive FPS, physical wire rate, loss/reorder/decode errors, queue latency, and relevant structured logs or phase analysis.

For receiver/GAN work, validate at least:

1. syntax/import/startup in the intended environment;
2. a short functional run;
3. a sustained run long enough to expose queue buildup, stale frames, fallback loops, or memory growth;
4. representative visual A/B output when image quality changes;
5. latency distribution (`p50`, `p95`, `p99`, and max), not average latency alone.

For performance-sensitive changes, record the exact hardware, mode, source resolution/FPS, model, backend, precision, and relevant sender bitrate/cap. When comparing backends or precision modes, use the same decoded input frames where practical.

Do not claim a performance target is met from a few seconds of testing. Do not claim a bug is fixed merely because the application starts; verify the task's actual latency, throughput, bitrate, ordering, stability, and quality criteria.

## Commit & Pull Request Guidelines

Recent commits use concise, outcome-oriented subjects, for example `fallback死锁 fix` or `增加多个档位`. Keep each commit scoped and describe the user-visible result.

Pull requests should summarize behavior changes, list exact test commands, identify hardware/profile combinations tested, and link logs or screenshots for HUD or visual-quality changes.

Never commit credentials, machine-specific paths, generated binaries, accelerator caches, models, or run logs unless the repository explicitly tracks them.

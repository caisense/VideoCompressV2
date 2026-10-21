"""Motion-interpolated GAN runtime for live_h265_hud --fps.

RTP supplies source geometry and cadence. One replaceable input, eight timed
outputs, a finite 200ms playout delay and expired-output rejection bound lag.
The ordinary receiver remains the default when --fps is absent.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import textwrap
import threading
import time

import cv2
import numpy as np

from full_frame_enhancer import ProcessFullFrameEnhancer
from live_h265_hud import (
    DecodedPtsQueue, HevcRtpDepacketizer, ProfileSwitchGate, RtpStats,
    ffmpeg_frame_rate_args, gan_effective_output_budget_ms,
    gan_profile_sizes, gan_scale, is_gan_profile, postprocess_frame,
    profile_switch_key, read_bmp_frame, receiver_gan_profile,
)
from motion_interpolation import MotionPair, grid_between

PLAYOUT_MS = 200.0


def interpolation_settings(args, profile):
    """Select the actual RTP-selected model/geometry; never invent a GAN hint."""
    if not is_gan_profile(profile):
        raise ValueError("--fps requires a sender GAN hint (for example --gan=x2)")
    if int(profile["fps"]) <= 0:
        raise ValueError("invalid source frame rate")
    if args.gan_enhancer == "esrnet" and gan_scale(profile) != 2:
        raise ValueError("ESRNet interpolation requires a native x2 stream")
    source, native, output = gan_profile_sizes(profile)
    model = (args.gan_esrnet_model if args.gan_enhancer == "esrnet" else
             args.gan_esrgan_x4_model if gan_scale(profile) == 4 else args.gan_esrgan_model)
    budget = gan_effective_output_budget_ms(
        float(profile["fps"]), args.gan_output_latency_budget_ms,
        args.gan_output_latency_factor, args.gan_max_inference_latency_ms, profile)
    return dict(source=source, native=native, output=output, model=model,
                source_fps=float(profile["fps"]), budget_ms=budget)


def _open_optional(path, mode):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode, **({"encoding": "utf8", "buffering": 1} if "b" not in mode else {}))


def run_interpolated_receiver(args, listen_port):
    cv2.setNumThreads(args.gan_threads or 2)
    stats, pts_queue = RtpStats(), DecodedPtsQueue()
    stop, profile_ready, calibrated = (threading.Event() for _ in range(3))
    condition, output_lock, log_lock = threading.Condition(), threading.Lock(), threading.Lock()
    pending, outputs = collections.deque(maxlen=1), collections.deque(maxlen=8)
    counts = collections.Counter()
    state = dict(profile=None, settings=None, enhancer=None, phase="WAITING FOR GAN PROFILE",
                 running_since=None, precision="", provider="", origin=None)
    histories = {key: collections.deque(maxlen=512) for key in ("sr", "flow", "warp", "latency")}
    presented = collections.deque()
    log = _open_optional(args.gan_debug_log, "w")
    packets_log = _open_optional(args.quality_packet_log, "w")
    video = _open_optional(args.quality_video_log, "wb")

    def event(event_type, **fields):
        if event_type == "ERROR":
            counts["fatal_errors"] += 1
            state["phase"] = "ERROR"
            print("INTERPOLATION ERROR: " + str(fields), flush=True)
        if log is not None:
            with log_lock:
                log.write(json.dumps(dict(type=event_type, TYPE="FPS_"+event_type,
                                          at=time.perf_counter(), **fields))+"\n")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    receiver.bind(("0.0.0.0", listen_port))
    receiver.settimeout(.2)
    command = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning",
               "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "1024",
               "-f", "hevc", "-i", "pipe:0", "-an", "-sn", "-dn",
               *ffmpeg_frame_rate_args(args.ffmpeg), "-pix_fmt", "bgr24",
               "-f", "image2pipe", "-vcodec", "bmp", "pipe:1"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def receive():
        gate, depacketizer = ProfileSwitchGate(), HevcRtpDepacketizer()
        try:
            while not stop.is_set():
                try:
                    packet, _ = receiver.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                stats.on_packet(packet)
                if packets_log is not None:
                    offset = RtpStats._payload_offset(packet)
                    packets_log.write(json.dumps(dict(at=time.perf_counter(), udp_bytes=len(packet),
                        rtp_payload_bytes=max(0, len(packet)-(offset or len(packet))),
                        wire_bytes=RtpStats._wire_bytes(len(packet)),
                        rtp_timestamp=int.from_bytes(packet[4:8], "big"),
                        marker=bool(packet[1] & 128)))+"\n")
                profile = stats.snapshot()["profile"]
                if profile is not None:
                    profile = receiver_gan_profile(profile, args.ganq_scale, False)
                    if state["profile"] is None:
                        settings = interpolation_settings(args, profile)
                        state.update(profile=profile, settings=settings, phase="GAN WARMING")
                        event("PROFILE", profile=profile, settings=settings, output_fps=args.fps)
                        profile_ready.set()
                    elif profile_switch_key(profile) != profile_switch_key(state["profile"]):
                        raise ValueError("source profile changed during --fps; restart receiver for the new profile")
                generation, ready = gate.feed(packet, profile)
                if generation is not None:
                    depacketizer.reset()
                for item in ready:
                    data = depacketizer.feed(item)
                    if data:
                        if video is not None:
                            video.write(data)
                        if item[1] & 128:
                            pts_queue.push_rtp_timestamp(int.from_bytes(item[4:8], "big"))
                        process.stdin.write(data)
        except Exception as error:
            if not stop.is_set():
                event("ERROR", stage="receive", error=str(error))
                stop.set()

    def decode():
        last_pts, elapsed, sequence, anchors = None, 0., 0, []
        try:
            while not stop.is_set():
                frame = read_bmp_frame(process.stdout)
                if frame is None:
                    if not stop.is_set():
                        raise RuntimeError("unexpected decoder EOF")
                    break
                now, pts = time.perf_counter(), pts_queue.pop()
                if pts is None:
                    raise ValueError("missing decoded source RTP timestamp")
                width, height = state["settings"]["source"]
                if frame.shape != (height, width, 3):
                    raise ValueError("decoded dimensions disagree with source RTP metadata")
                stats.on_decoded_frame()
                if last_pts is not None:
                    delta = (pts-last_pts) & 0xffffffff
                    if delta == 0 or delta > 90000:
                        raise ValueError("non-monotonic or >1s RTP timestamp gap; restart sender/receiver")
                    elapsed += delta/90000
                last_pts, sequence = pts, sequence+1
                if not calibrated.is_set():
                    anchors.append(now-elapsed)
                    if len(anchors) == 30:
                        state["origin"] = statistics.median(anchors)
                        calibrated.set()
                        event("CLOCK_CALIBRATED", samples=30, decode_clock_origin=state["origin"])
                event("DECODE", seq=sequence, pts=elapsed)
                with condition:
                    if pending:
                        counts["input_replaced"] += 1
                    pending.append(dict(frame=frame, pts=elapsed, seq=sequence, decoded=now))
                    condition.notify()
        except Exception as error:
            if not stop.is_set():
                event("ERROR", stage="decode", error=str(error))
                stop.set()

    def errors():
        for raw in iter(process.stderr.readline, b""):
            message = raw.decode(errors="replace").strip()
            if message:
                print("[ffmpeg] " + message, flush=True)
            if any(term in raw.lower() for term in (b"error", b"invalid", b"could not find ref")):
                stats.on_decode_error()

    def work():
        previous, next_index, enhancer = None, 0, None
        try:
            while not profile_ready.wait(.2):
                if stop.is_set():
                    return
            if stop.is_set():
                return
            settings = state["settings"]
            enhancer = ProcessFullFrameEnhancer(
                backend=args.gan_enhancer, model_path=settings["model"],
                input_size=settings["source"], native_size=settings["native"],
                output_size=settings["output"], execution_provider=args.gan_execution_provider,
                tensorrt_cache_dir=args.gan_trt_cache_dir, threads=args.gan_threads,
                warmup=args.gan_warmup)
            if stop.is_set():
                enhancer.close()
                return
            state.update(enhancer=enhancer, provider=enhancer.provider,
                         precision=enhancer.execution_precision, phase="CLOCK CALIBRATING")
            event("BACKEND", sr_provider=enhancer.provider, sr_precision=enhancer.execution_precision,
                  sr_model=str(Path(settings["model"]).resolve()),
                  interpolation="OpenCV DIS FAST CPU bidirectional warp", source=list(settings["source"]),
                  native=list(settings["native"]), output=list(settings["output"]),
                  output_fps=args.fps, playout_delay_ms=PLAYOUT_MS)
            print(f"FPS READY: source={settings['source']} native={settings['native']} "
                  f"output={settings['output']} target={args.fps:g} CPU DIS "
                  f"playout={PLAYOUT_MS:g}ms; {enhancer.provider}; {enhancer.execution_precision}", flush=True)
            while not stop.is_set():
                with condition:
                    condition.wait_for(lambda: bool(pending) or stop.is_set(), timeout=.2)
                    if not pending:
                        continue
                    current = pending.pop()
                began = time.perf_counter()
                state["running_since"] = began
                current["sr"] = enhancer.enhance(current["frame"])
                completed = time.perf_counter()
                state["running_since"] = None
                histories["sr"].append((completed-began)*1000)
                event("SR", seq=current["seq"], pts=current["pts"], infer_ms=(completed-began)*1000,
                      queue_ms=(began-current["decoded"])*1000, total_ms=(completed-current["decoded"])*1000)
                if (completed-current["decoded"])*1000 > settings["budget_ms"]:
                    counts["sr_expired"] += 1
                    event("SR_DROP", seq=current["seq"], reason="source_age_budget")
                    previous = None
                    continue
                counts["sr_completed"] += 1
                if previous is None or not calibrated.is_set():
                    previous = current
                    continue
                if current["pts"]-previous["pts"] > .25:
                    counts["source_gap"] += 1
                    previous = current
                    continue
                began_pair = time.perf_counter()
                pair = MotionPair(previous["frame"], current["frame"], previous["sr"], current["sr"])
                flow_ms = (time.perf_counter()-began_pair)*1000
                histories["flow"].append(flow_ms)
                for index, timestamp, alpha in grid_between(previous["pts"], current["pts"], next_index, args.fps):
                    next_index = index+1
                    due = state["origin"]+timestamp+PLAYOUT_MS/1000
                    if time.perf_counter() > due+.033:
                        counts["compute_late"] += 1
                        event("GRID_DROP", index=index, reason="compute_late")
                        continue
                    began_render = time.perf_counter()
                    image, kind = pair.render(alpha)
                    warp_ms = (time.perf_counter()-began_render)*1000
                    histories["warp"].append(warp_ms)
                    job = dict(index=index, pts=timestamp, due=due, frame=image, kind=kind,
                               a_seq=previous["seq"], b_seq=current["seq"], alpha=alpha,
                               a_decoded=previous["decoded"], b_decoded=current["decoded"])
                    with output_lock:
                        if len(outputs) == outputs.maxlen:
                            counts["output_replaced"] += 1
                        outputs.append(job)
                    event("GRID_READY", index=index, pts=timestamp, kind=kind,
                          flow_ms=flow_ms, warp_ms=warp_ms, queue_depth=len(outputs))
                previous = current
                state["phase"] = "FLOW ACTIVE"
        except Exception as error:
            if not stop.is_set():
                event("ERROR", stage="super_resolution_interpolation", error=str(error))
                stop.set()
        finally:
            if enhancer is not None:
                enhancer.close()

    threads = [threading.Thread(target=target, daemon=True, name="fps-"+target.__name__)
               for target in (receive, decode, errors, work)]
    for thread in threads:
        thread.start()
    window = "RK3588 H.265 HUD - motion interpolation"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"INTERPOLATION LISTENING: UDP {listen_port}, --fps={args.fps:g}, CPU DIS, "
          f"playout={PLAYOUT_MS:g}ms; waiting for sender GAN metadata", flush=True)
    event("START", target_fps=args.fps, playout_delay_ms=PLAYOUT_MS, headless=args.headless)
    if args.quality_video_log is not None:
        paths = [Path(__file__), Path(__file__).with_name("live_h265_hud.py"),
                 Path(__file__).with_name("motion_interpolation.py")]
        args.quality_video_log.with_suffix(".runtime.json").write_text(json.dumps(dict(
            python=sys.executable, python_version=sys.version, headless=args.headless,
            source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), indent=2))
    started, last_status, last_idle, last_index = time.perf_counter(), 0., 0., -1
    previous_canvas, snapshot_saved = None, False
    try:
        while not stop.is_set() and (args.duration <= 0 or time.perf_counter()-started < args.duration):
            now, candidate = time.perf_counter(), None
            with output_lock:
                while outputs and outputs[0]["due"] <= now:
                    if candidate is not None:
                        counts["presentation_skip"] += 1
                        event("GRID_DROP", index=candidate["index"], reason="superseded_due")
                    candidate = outputs.popleft()
            if candidate is not None:
                if now-candidate["due"] > .033:
                    counts["presentation_late"] += 1
                    event("GRID_DROP", index=candidate["index"], reason="presentation_late")
                    candidate = None
                elif candidate["index"] <= last_index:
                    raise RuntimeError("interpolated frame presentation order violation")
            while presented and presented[0] < now-1:
                presented.popleft()
            key = 255
            if candidate is not None or now-last_idle >= .1:
                values, settings = stats.snapshot(), state["settings"]
                if candidate is not None:
                    canvas = postprocess_frame(candidate["frame"], args.rotate, False)
                    if args.scale > 1:
                        canvas = cv2.resize(canvas, (canvas.shape[1]*args.scale, canvas.shape[0]*args.scale),
                                            interpolation=cv2.INTER_LANCZOS4)
                    previous_canvas = canvas.copy()
                else:
                    canvas = (previous_canvas.copy() if previous_canvas is not None else
                              np.zeros((360, 640, 3), dtype=np.uint8))
                running_ms = 0 if state["running_since"] is None else (now-state["running_since"])*1000
                phase = ("SR HARD STALL (no fresh output)" if running_ms > args.gan_hard_stall_ms else state["phase"])
                lines = [f"INTERP DIS CPU | target {args.fps:g} effective {len(presented):g} fps",
                         f"RX {values['rx_fps']:.1f} DEC {values['decode_fps']:.1f} | {phase}",
                         f"FRAME {candidate['kind'] if candidate is not None else 'HOLD (no new output)'}"]
                if settings:
                    source, native, output = settings["source"], settings["native"], settings["output"]
                    lines += [f"{state['profile']['name']} SRC {source[0]}x{source[1]} @{settings['source_fps']:g}",
                              f"AI NATIVE {native[0]}x{native[1]} OUT {output[0]}x{output[1]}",
                              f"DISPLAY {canvas.shape[1]}x{canvas.shape[0]} {args.rotate}"]
                lines += [f"SR {state['precision'] or 'WARMING'}", f"Playout {PLAYOUT_MS:g}ms | fresh FLOW / real only",
                          f"WIRE {values['wire_kbps']:.1f}kbps LOSS {values['lost']} REO {values['reordered']} ERR {values['decode_errors']}",
                          f"SR {counts['sr_completed']} FLOW {counts['flow_display']} REAL {counts['real_display']} CUT {counts['cut_display']}",
                          f"DROP late {counts['compute_late']+counts['presentation_late']} SR {counts['sr_expired']} replace {counts['input_replaced']}"]
                if histories["latency"]:
                    lines.append(f"PC reference median {statistics.median(histories['latency']):.0f}ms (not camera-to-glass)")
                wrapped = [part for line in lines for part in textwrap.wrap(line, max(25, int(canvas.shape[1]/6.8)))]
                for n, line in enumerate(wrapped):
                    cv2.putText(canvas, line, (8, 18+n*17), cv2.FONT_HERSHEY_SIMPLEX, .36, (80,255,80), 1, cv2.LINE_AA)
                if not args.headless:
                    if candidate is not None and last_index < 0:
                        cv2.resizeWindow(window, canvas.shape[1], canvas.shape[0])
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(1) & 255
                last_idle = now
                if candidate is not None:
                    shown = time.perf_counter()
                    latency = (shown-(candidate["a_decoded"]*(1-candidate["alpha"])+candidate["b_decoded"]*candidate["alpha"]))*1000
                    histories["latency"].append(latency)
                    if candidate["kind"] in ("FLOW", "REAL_A", "REAL_B"):
                        presented.append(shown)
                    counts[{"FLOW":"flow_display", "REAL_A":"real_display", "REAL_B":"real_display"}.get(candidate["kind"], "cut_display")] += 1
                    event("DISPLAY", **{k:v for k,v in candidate.items() if k not in ("frame", "due", "a_decoded", "b_decoded")},
                          headless=args.headless, lateness_ms=(shown-candidate["due"])*1000,
                          pc_grid_latency_ms=(shown-state["origin"]-candidate["pts"])*1000,
                          interpolated_decode_latency_ms=latency,
                          past_frame_age_ms=(shown-candidate["a_decoded"])*1000,
                          future_frame_age_ms=(shown-candidate["b_decoded"])*1000)
                    last_index = candidate["index"]
                    if not snapshot_saved and args.quality_video_log is not None:
                        cv2.imwrite(str(args.quality_video_log.with_suffix(".hud.png")), canvas)
                        snapshot_saved = True
            if key in (27, ord('q')):
                break
            if now-last_status >= 5:
                event("STATUS", stats=stats.snapshot(), pts_queue=pts_queue.snapshot(), counters=dict(counts), effective_fps=len(presented))
                print(f"FPS target={args.fps:g} effective={len(presented)} state={state['phase']} "
                      f"loss={stats.snapshot()['lost']} errors={stats.snapshot()['decode_errors']}", flush=True)
                last_status = now
            time.sleep(.001)
    except Exception as error:
        event("ERROR", stage="presentation", error=str(error))
    finally:
        stop.set()
        with condition:
            condition.notify_all()
        receiver.close()
        if state["enhancer"] is not None:
            state["enhancer"].close()
        try:
            process.stdin.close()
            process.wait(3)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(3)
        for thread in threads:
            thread.join(4)
        event("FINISH", stats=stats.snapshot(), counters=dict(counts), elapsed=time.perf_counter()-started)
        if not args.headless:
            cv2.destroyAllWindows()
        for stream in (log, packets_log, video):
            if stream is not None:
                stream.close()
    return 1 if counts["fatal_errors"] else 0

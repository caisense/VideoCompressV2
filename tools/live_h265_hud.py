#!/usr/bin/env python3
"""Native-PC H.265 RTP viewer with live transport and decoder HUD metrics."""

from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

import cv2
import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from full_frame_enhancer import FullFrameEnhancer, LatestOnlyEnhancerWorker
from rebuild_receiver import RebuildComposer, RebuildReceiver, SuperResolver


DEFAULT_VIDEO_PORT = 5004


def gan_profile_sizes(profile: dict) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Return RTP-selected source, x2-native, and fixed presentation sizes."""
    width = int(profile.get("width", 0))
    height = int(profile.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"GAN profile has invalid input size {width}x{height}")
    return (width, height), (width * 2, height * 2), (640, 360)


def gan_profile_signature(profile: dict) -> Tuple[int, int, int, int]:
    """Identity of an enhancer/decoder generation, including its cadence."""
    input_size, _, _ = gan_profile_sizes(profile)
    return (
        int(profile.get("generation", -1)),
        input_size[0],
        input_size[1],
        int(profile.get("fps", 0)),
    )


def gan_target_kbps_text(profile: Optional[dict]) -> str:
    """Format optional GAN target metadata without rejecting legacy senders."""
    target = None if profile is None else profile.get("target_bitrate_kbps")
    return "--" if target is None else str(int(target))


def gan_link_cap_kbps(profile: Optional[dict]) -> int:
    """Read advertised GAN CAP, retaining the documented old-stream fallback."""
    advertised = None if profile is None else profile.get("link_cap_kbps")
    if advertised is not None and int(advertised) > 0:
        return int(advertised)
    # Both the 8-byte legacy record and the former TARGET+reserved tail were
    # the original 100 kbps GAN profile.  This is a compatibility fallback,
    # not the active link-rate source for new senders.
    return 100


def gan_bitrate_hud_line(profile: Optional[dict], values: dict) -> str:
    """Keep configured target and observed receive rates visibly distinct."""
    return (
        f"TARGET {gan_target_kbps_text(profile)}  RTP {values['rtp_kbps']:.1f}  "
        f"WIRE {values['wire_kbps']:.1f}  CAP {gan_link_cap_kbps(profile)} kbps"
    )


def gan_waiting_worker_snapshot() -> dict:
    """HUD-safe zero metrics while a new generation waits for its IDR."""
    return {
        "backend": "warming",
        "provider": "--",
        "rolling_1s_fps": 0.0,
        "max_latency_ms": 0.0,
        "infer_all_p50_ms": 0.0,
        "infer_all_p95_ms": 0.0,
        "infer_all_p99_ms": 0.0,
        "infer_all_max_ms": 0.0,
        "last_finished_infer_ms": 0.0,
        "good_total_p50_ms": 0.0,
        "good_total_p95_ms": 0.0,
        "good_total_p99_ms": 0.0,
        "good_total_max_ms": 0.0,
        "last_good_total_ms": 0.0,
        "last_queue_wait_ms": 0.0,
        "running": False,
        "running_age_ms": 0.0,
        "pending": False,
        "dropped": 0,
        "drop_percent": 0.0,
        "drop_reason_counts": {
            "replaced_pending": 0,
            "output_replaced": 0,
            "stale_before_infer": 0,
            "stale_before_predicted": 0,
            "stale_after_infer": 0,
            "inference_error": 0,
        },
    }


class RtpStats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.samples: Deque[Tuple[float, int, int]] = collections.deque()
        self.markers: Deque[float] = collections.deque()
        self.p_frames: Deque[float] = collections.deque()
        self.i_frames: Deque[float] = collections.deque()
        self.decoded: Deque[float] = collections.deque()
        self.last_sequence: Optional[int] = None
        self.lost_packets = 0
        self.reordered_packets = 0
        self.duplicate_packets = 0
        self.total_packets = 0
        self.decode_errors = 0
        self.last_idr_time: Optional[float] = None
        self.total_p_frames = 0
        self.total_i_frames = 0
        self._access_unit_has_vcl = False
        self._access_unit_is_i = False
        self.stream_profile: Optional[dict] = None

    @staticmethod
    def _payload_offset(packet: bytes) -> Optional[int]:
        if len(packet) < 12 or packet[0] >> 6 != 2:
            return None
        offset = 12 + 4 * (packet[0] & 0x0F)
        if offset > len(packet):
            return None
        if packet[0] & 0x10:
            if offset + 4 > len(packet):
                return None
            words = int.from_bytes(packet[offset + 2:offset + 4], "big")
            offset += 4 + words * 4
        return offset if offset <= len(packet) else None

    @staticmethod
    def _profile_extension(packet: bytes) -> Optional[dict]:
        if len(packet) < 16 or not packet[0] & 0x10:
            return None
        offset = 12 + 4 * (packet[0] & 0x0F)
        if offset + 4 > len(packet) or packet[offset:offset + 2] != b"RO":
            return None
        words = int.from_bytes(packet[offset + 2:offset + 4], "big")
        payload_end = offset + 4 + words * 4
        if payload_end > len(packet):
            return None
        payload = packet[offset + 4:payload_end]
        if len(payload) < 8 or payload[0] != 1:
            return None
        names = {0: "low", 1: "medium", 2: "high", 3: "rebuild", 4: "gan"}
        name = names.get(payload[1], "unknown")
        target_bitrate_kbps = None
        link_cap_kbps = None
        # GAN v1 originally had an eight-byte core.  The current sender then
        # appended TARGET plus two reserved bytes; new senders reuse those
        # final two bytes as CAP without increasing the 12-byte tail layout.
        if name == "gan":
            if len(payload) >= 10:
                advertised_target = int.from_bytes(payload[8:10], "big")
                if advertised_target > 0:
                    target_bitrate_kbps = advertised_target
            if len(payload) >= 12:
                advertised_cap = int.from_bytes(payload[10:12], "big")
                if advertised_cap > 0:
                    link_cap_kbps = advertised_cap
        return {
            "name": name,
            "width": int.from_bytes(payload[2:4], "big"),
            "height": int.from_bytes(payload[4:6], "big"),
            "fps": payload[6],
            "generation": payload[7],
            "target_bitrate_kbps": target_bitrate_kbps,
            "link_cap_kbps": link_cap_kbps,
        }

    @staticmethod
    def _wire_bytes(udp_payload_bytes: int) -> int:
        return 38 + max(46, udp_payload_bytes + 28)

    @staticmethod
    def _nal_type(packet: bytes) -> Optional[int]:
        offset = RtpStats._payload_offset(packet)
        if offset is None or offset + 2 > len(packet):
            return None
        nal_type = (packet[offset] >> 1) & 0x3F
        if nal_type == 49 and offset + 3 <= len(packet):  # RFC 7798 FU
            if not packet[offset + 2] & 0x80:
                return None
            nal_type = packet[offset + 2] & 0x3F
        return nal_type

    @staticmethod
    def _is_idr(packet: bytes) -> bool:
        nal_type = RtpStats._nal_type(packet)
        if nal_type is None:
            return False
        return 16 <= nal_type <= 21

    @staticmethod
    def _vcl_type(packet: bytes) -> Optional[int]:
        """Return the HEVC VCL NAL type carried by one RTP packet, if any."""
        nal_type = RtpStats._nal_type(packet)
        if nal_type is None:
            return None
        return nal_type if nal_type <= 31 else None

    def on_packet(self, packet: bytes, now: Optional[float] = None) -> bool:
        if len(packet) < 12 or packet[0] >> 6 != 2:
            return False
        now = time.monotonic() if now is None else now
        sequence = int.from_bytes(packet[2:4], "big")
        profile = self._profile_extension(packet)
        with self.lock:
            profile_changed = False
            if profile is not None:
                if self.stream_profile is None or \
                        profile["generation"] != self.stream_profile["generation"]:
                    profile_changed = self.stream_profile is not None
                    self._access_unit_has_vcl = False
                    self._access_unit_is_i = False
                self.stream_profile = profile
            self.total_packets += 1
            self.samples.append((now, len(packet), self._wire_bytes(len(packet))))
            if packet[1] & 0x80:
                self.markers.append(now)
            if self._is_idr(packet):
                self.last_idr_time = now
            vcl_type = self._vcl_type(packet)
            if vcl_type is not None:
                self._access_unit_has_vcl = True
                self._access_unit_is_i = self._access_unit_is_i or 16 <= vcl_type <= 21
            if packet[1] & 0x80:
                if self._access_unit_has_vcl:
                    if self._access_unit_is_i:
                        self.i_frames.append(now)
                        self.total_i_frames += 1
                    else:
                        self.p_frames.append(now)
                        self.total_p_frames += 1
                self._access_unit_has_vcl = False
                self._access_unit_is_i = False
            if self.last_sequence is not None:
                delta = (sequence - self.last_sequence) & 0xFFFF
                if delta == 0:
                    self.duplicate_packets += 1
                elif delta < 0x8000:
                    self.lost_packets += max(0, delta - 1)
                else:
                    self.reordered_packets += 1
            if self.last_sequence is None or 0 < ((sequence - self.last_sequence) & 0xFFFF) < 0x8000:
                self.last_sequence = sequence
            self._trim(now)
            return profile_changed

    def on_decoded_frame(self) -> None:
        now = time.monotonic()
        with self.lock:
            self.decoded.append(now)
            self._trim(now)

    def on_decode_error(self) -> None:
        with self.lock:
            self.decode_errors += 1

    def _trim(self, now: float) -> None:
        cutoff = now - 1.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.markers and self.markers[0] < cutoff:
            self.markers.popleft()
        while self.p_frames and self.p_frames[0] < cutoff:
            self.p_frames.popleft()
        while self.i_frames and self.i_frames[0] < cutoff:
            self.i_frames.popleft()
        while self.decoded and self.decoded[0] < cutoff:
            self.decoded.popleft()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            self._trim(now)
            packet_count = len(self.samples)
            payload_total = sum(item[1] for item in self.samples)
            return {
                "rx_fps": float(len(self.markers)),
                "p_fps": float(len(self.p_frames)),
                "i_fps": float(len(self.i_frames)),
                "p_frames": self.total_p_frames,
                "i_frames": self.total_i_frames,
                "decode_fps": float(len(self.decoded)),
                "rtp_kbps": payload_total * 8.0 / 1000.0,
                "wire_kbps": sum(item[2] for item in self.samples) * 8.0 / 1000.0,
                "pps": float(packet_count),
                "packet_last_bytes": self.samples[-1][1] if self.samples else 0,
                "packet_avg_bytes": payload_total / packet_count if packet_count else 0.0,
                "packet_max_bytes": max((item[1] for item in self.samples), default=0),
                "lost": self.lost_packets,
                "reordered": self.reordered_packets,
                "duplicates": self.duplicate_packets,
                "packets": self.total_packets,
                "decode_errors": self.decode_errors,
                "idr_age": None if self.last_idr_time is None else now - self.last_idr_time,
                "profile": None if self.stream_profile is None else dict(self.stream_profile),
            }


def profile_switch_key(profile: Optional[dict]) -> Tuple[Any, ...]:
    """Decoder identity; dimensions matter when a sender process restarts."""
    if profile is None:
        return (-1, "legacy", 0, 0, 0)
    return (
        int(profile.get("generation", -1)),
        str(profile.get("name", "unknown")),
        int(profile.get("width", 0)),
        int(profile.get("height", 0)),
        int(profile.get("fps", 0)),
    )


class ProfileSwitchGate:
    """Buffer the first complete IDR of each RTP decoder profile identity."""

    def __init__(self) -> None:
        self.active_generation: Optional[int] = None
        self.active_profile_key: Optional[Tuple[Any, ...]] = None
        self.pending_generation: Optional[int] = None
        self.pending_profile_key: Optional[Tuple[Any, ...]] = None
        self.pending_packets = []
        self.pending_has_idr = False

    def feed(self, packet: bytes, profile: Optional[dict]) -> Tuple[Optional[int], list]:
        # Generation -1 keeps compatibility with senders that do not carry
        # the project-specific profile extension.
        generation = -1 if profile is None else profile["generation"]
        key = profile_switch_key(profile)
        if self.active_profile_key == key:
            return None, [packet]
        if self.pending_profile_key != key:
            self.pending_generation = generation
            self.pending_profile_key = key
            self.pending_packets = []
            self.pending_has_idr = False
        self.pending_packets.append(packet)
        self.pending_has_idr = self.pending_has_idr or RtpStats._is_idr(packet)
        if not packet[1] & 0x80:
            return None, []
        if not self.pending_has_idr:
            self.pending_packets = []
            return None, []
        buffered = self.pending_packets
        self.active_generation = generation
        self.active_profile_key = key
        self.pending_generation = None
        self.pending_profile_key = None
        self.pending_packets = []
        self.pending_has_idr = False
        return generation, buffered


class HevcRtpDepacketizer:
    """Convert the RFC 7798 packet forms emitted by our sender to Annex-B."""

    START_CODE = b"\x00\x00\x00\x01"

    def __init__(self) -> None:
        self.fu_expected_sequence: Optional[int] = None

    def reset(self) -> None:
        self.fu_expected_sequence = None

    def feed(self, packet: bytes) -> bytes:
        offset = RtpStats._payload_offset(packet)
        if offset is None or offset + 2 > len(packet):
            return b""
        end = len(packet)
        if packet[0] & 0x20:  # RTP padding
            padding = packet[-1]
            if padding == 0 or padding > end - offset:
                return b""
            end -= padding
        payload = packet[offset:end]
        if len(payload) < 2:
            return b""
        nal_type = (payload[0] >> 1) & 0x3F
        if nal_type == 48:  # Aggregation packet
            output = bytearray()
            cursor = 2
            while cursor + 2 <= len(payload):
                size = int.from_bytes(payload[cursor:cursor + 2], "big")
                cursor += 2
                if size == 0 or cursor + size > len(payload):
                    return b""
                output.extend(self.START_CODE)
                output.extend(payload[cursor:cursor + size])
                cursor += size
            return bytes(output) if cursor == len(payload) else b""
        if nal_type != 49:  # A complete NAL unit
            self.fu_expected_sequence = None
            return self.START_CODE + payload
        if len(payload) < 3:
            self.fu_expected_sequence = None
            return b""
        sequence = int.from_bytes(packet[2:4], "big")
        fu_header = payload[2]
        start = bool(fu_header & 0x80)
        end_fragment = bool(fu_header & 0x40)
        if start:
            first_byte = (payload[0] & 0x81) | ((fu_header & 0x3F) << 1)
            self.fu_expected_sequence = (sequence + 1) & 0xFFFF
            if end_fragment:
                self.fu_expected_sequence = None
            return self.START_CODE + bytes((first_byte, payload[1])) + payload[3:]
        if self.fu_expected_sequence != sequence:
            self.fu_expected_sequence = None
            return b""
        self.fu_expected_sequence = None if end_fragment else (sequence + 1) & 0xFFFF
        return payload[3:]


def parse_sdp(path: Path) -> Tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^m=video\s+(\d+)\s+RTP/AVP\s+96\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"SDP has no H.265 RTP/AVP 96 video port: {path}")
    return text, int(match.group(1))


def resolve_video_port(sdp: Optional[Path], requested_port: Optional[int]) -> int:
    """Resolve the UDP listen port without requiring an SDP file.

    The HUD receives and depacketizes RFC 7798 itself, so it only used the
    legacy SDP for its ``m=video`` port.  Keep that input compatible while
    allowing the common fixed-port path to start directly.
    """
    if requested_port is not None and not 1 <= requested_port <= 65535:
        raise ValueError("--udp-port must be a valid UDP port (1..65535)")
    if sdp is None:
        return DEFAULT_VIDEO_PORT if requested_port is None else requested_port
    _, sdp_port = parse_sdp(sdp)
    if requested_port is not None and requested_port != sdp_port:
        raise ValueError(
            f"--udp-port {requested_port} does not match SDP video port {sdp_port}"
        )
    return sdp_port


def ffmpeg_frame_rate_args_from_help(help_text: str) -> Tuple[str, ...]:
    """Return a supported output frame-pacing option for this FFmpeg build.

    FFmpeg 4.x commonly exposes the legacy ``-vsync`` option while newer
    builds expose ``-fps_mode``. Passing either option to a build that does
    not know it aborts the decoder before the first packet is received, so
    select from the actual help text instead of assuming a version.
    """
    if re.search(r"(?m)^\s*-fps_mode(?:\s|:|\[|$)", help_text):
        return ("-fps_mode", "passthrough")
    if re.search(r"(?m)^\s*-vsync(?:\s|:|$)", help_text):
        return ("-vsync", "0")
    return ()


def ffmpeg_frame_rate_args(ffmpeg: str) -> Tuple[str, ...]:
    """Probe FFmpeg and return a compatible frame-pacing argument pair."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-h", "full"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return ffmpeg_frame_rate_args_from_help(result.stdout or "")


class LatestFrame:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.updated = 0.0
        self.sequence = 0
        self.rtp_timestamp: Optional[int] = None

    def put(self, frame: np.ndarray, rtp_timestamp: Optional[int] = None) -> int:
        with self.lock:
            self.frame = frame
            self.updated = time.monotonic()
            self.sequence += 1
            self.rtp_timestamp = rtp_timestamp
            return self.sequence

    def get(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), self.updated)

    def get_versioned(self) -> Tuple[Optional[np.ndarray], float, int, Optional[int]]:
        with self.lock:
            return (None if self.frame is None else self.frame.copy(),
                    self.updated, self.sequence, self.rtp_timestamp)

    def clear(self) -> None:
        with self.lock:
            self.frame = None
            self.updated = 0.0
            self.sequence += 1
            self.rtp_timestamp = None


class DecodedPtsQueue:
    """Bounded FIFO handoff from complete RTP AUs to decoded BMP frames.

    Consuming the newest timestamp associated picture *N* with *N+1* whenever
    FFmpeg had already queued the next AU.  There are no B frames here, so
    decoded output normally follows submission order and consumes the oldest
    pending timestamp.  The bound prevents a decoder stall from creating an
    unbounded queue; the receiver's sliding PTS calibration handles recovery.
    """

    def __init__(self, max_pending: int = 16) -> None:
        if max_pending < 2:
            raise ValueError("decoded PTS queue must hold at least two entries")
        self.lock = threading.Lock()
        self.max_pending = max_pending
        self.items: Deque[int] = collections.deque()
        self.dropped = 0

    def push_rtp_timestamp(self, timestamp: int) -> None:
        with self.lock:
            while len(self.items) >= self.max_pending:
                self.items.popleft()
                self.dropped += 1
            self.items.append(timestamp & 0xFFFFFFFF)

    def pop(self) -> Optional[int]:
        with self.lock:
            if not self.items:
                return None
            return self.items.popleft()

    def snapshot(self) -> dict:
        with self.lock:
            return {"pending": len(self.items), "dropped": self.dropped}

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


class PresentationStats:
    """Honest output provenance for a 12 Hz display fed by a slower decoder."""

    def __init__(self) -> None:
        self.samples: Deque[float] = collections.deque()
        self.total = 0
        self.decoded = 0
        self.held = 0
        self.last_source_sequence: Optional[int] = None
        self.last_provenance = "WAIT"
        self.generation: Optional[int] = None

    def present(self, source_sequence: int, spatial: str, generation: int,
                now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        if self.generation != generation:
            self.total = self.decoded = self.held = 0
            self.last_source_sequence = None
            self.generation = generation
        is_new = source_sequence != self.last_source_sequence
        temporal = "DECODED" if is_new else "HOLD"
        self.total += 1
        if is_new:
            self.decoded += 1
            self.last_source_sequence = source_sequence
        else:
            self.held += 1
        self.samples.append(now)
        while self.samples and self.samples[0] < now - 1.0:
            self.samples.popleft()
        self.last_provenance = f"{temporal}+{spatial}"
        return self.last_provenance

    def snapshot(self) -> dict:
        now = time.monotonic()
        while self.samples and self.samples[0] < now - 1.0:
            self.samples.popleft()
        ratio = 0.0 if self.total == 0 else self.held * 100.0 / self.total
        return {
            "fps": float(len(self.samples)),
            "total": self.total,
            "decoded": self.decoded,
            "held": self.held,
            "held_percent": ratio,
            "provenance": self.last_provenance,
        }


def read_exact(stream, size: int) -> Optional[bytes]:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def read_bmp_frame(stream) -> Optional[np.ndarray]:
    """Read one self-delimiting BMP image from FFmpeg's image2pipe output.

    Rawvideo has no frame boundary, so reading it with an incorrect configured
    width or height silently joins fragments of different frames into tiled
    images.  BMP carries its own byte length; decoding it here makes the HUD
    safe across sender resolution changes.
    """
    header = read_exact(stream, 14)
    if header is None or header[:2] != b"BM":
        return None
    frame_size = int.from_bytes(header[2:6], "little")
    if frame_size < 54 or frame_size > 64 * 1024 * 1024:
        return None
    payload = read_exact(stream, frame_size - len(header))
    if payload is None:
        return None
    return cv2.imdecode(np.frombuffer(header + payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def postprocess_frame(frame: np.ndarray, rotation: str, denoise: bool) -> np.ndarray:
    """Apply display-only orientation and edge-preserving smoothing."""
    if rotation == "ccw90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "cw90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "180":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    if denoise:
        # Bilateral filtering reduces block/ringing noise while retaining ROI
        # edges; this is display-only and does not alter the H.265 stream.
        frame = cv2.bilateralFilter(frame, 5, 24.0, 24.0)
    return frame


def gan_effective_output_budget_ms(
    source_fps: float,
    requested_budget_ms: float = 0.0,
    latency_factor: float = 1.15,
    legacy_budget_ms: Optional[float] = None,
) -> float:
    """Resolve the output-age budget for one encoded source profile.

    The old fixed 100 ms budget is intentionally not the default anymore.
    An explicit legacy value still wins so existing scripts can opt into the
    old policy while receiving a clear log line from the receiver.
    """
    if legacy_budget_ms is not None:
        if legacy_budget_ms <= 0.0:
            raise ValueError("legacy GAN latency budget must be positive")
        return float(legacy_budget_ms)
    if requested_budget_ms < 0.0:
        raise ValueError("GAN output latency budget must be non-negative")
    if requested_budget_ms > 0.0:
        return float(requested_budget_ms)
    if latency_factor <= 0.0:
        raise ValueError("GAN output latency factor must be positive")
    fps = max(1.0, float(source_fps))
    return float(latency_factor) * 1000.0 / fps


def gan_fallback_stale_after_ms(source_fps: float, max_latency_ms: float) -> float:
    """Bound a stalled enhancer to two source-frame intervals at most.

    The worker's latency budget rejects a late result only after its provider
    call returns.  This separate display budget prevents one stuck CUDA/ORT
    call from leaving the last enhanced canvas on screen indefinitely.
    """
    fps = max(1.0, float(source_fps))
    return max(2000.0 / fps, 1.5 * max(1.0, float(max_latency_ms)))


class GanAction:
    """Actions returned by :class:`GanFallbackController`."""

    KEEP_GAN = "KEEP_GAN"
    USE_LANCZOS = "USE_LANCZOS"
    # The candidate is safe to paint.  During recovery this can happen before
    # the controller has collected all recovery-good samples.
    ACCEPT_GAN = "ACCEPT_GAN"
    DISCARD_STALE_GAN = "DISCARD_STALE_GAN"


class GanControllerDecision:
    __slots__ = (
        "action", "state", "good_age_ms", "run_age_ms", "soft_threshold_ms",
        "output_age_ms", "recovery_streak", "events",
    )

    def __init__(self, action: str, state: str, good_age_ms: Optional[float],
                 run_age_ms: float, soft_threshold_ms: float,
                 output_age_ms: Optional[float], recovery_streak: int,
                 events: Tuple[dict, ...] = ()) -> None:
        self.action = action
        self.state = state
        self.good_age_ms = good_age_ms
        self.run_age_ms = run_age_ms
        self.soft_threshold_ms = soft_threshold_ms
        self.output_age_ms = output_age_ms
        self.recovery_streak = recovery_streak
        self.events = events


class GanFallbackController:
    """Pure state/age policy for the asynchronous full-frame GAN path.

    The decoder never calls into this controller while holding its socket or
    decoder locks.  It only decides which already-decoded source should be
    displayed.  This makes the soft-overdue, hard-stall, freshness and
    recovery policies deterministic in unit tests and keeps fallback off the
    H.265 decode path.
    """

    HEALTHY = "HEALTHY"
    SOFT_FALLBACK = "SOFT_FALLBACK"
    HARD_STALLED = "HARD_STALLED"

    def __init__(
        self,
        output_budget_ms: float,
        hard_stall_ms: float = 500.0,
        recovery_good_frames: int = 3,
        recovery_max_sequence_lag: int = 1,
    ) -> None:
        if output_budget_ms <= 0.0:
            raise ValueError("GAN output budget must be positive")
        if hard_stall_ms <= 0.0:
            raise ValueError("GAN hard-stall threshold must be positive")
        if recovery_good_frames <= 0:
            raise ValueError("GAN recovery frame count must be positive")
        if recovery_max_sequence_lag < 0:
            raise ValueError("GAN recovery sequence lag must be non-negative")
        self.output_budget_ms = float(output_budget_ms)
        self.hard_stall_ms = float(hard_stall_ms)
        self.recovery_good_frames = int(recovery_good_frames)
        self.recovery_max_sequence_lag = int(recovery_max_sequence_lag)
        self.state = self.HEALTHY
        self.last_good_completed_at: Optional[float] = None
        self.last_good_sequence: Optional[int] = None
        self.fallback_started_at: Optional[float] = None
        self.fallback_displayed_sequence: Optional[int] = None
        self.recovery_streak = 0
        self.current_run_age_ms = 0.0
        self.last_output_age_ms: Optional[float] = None
        self.last_output_arrived_at: Optional[float] = None
        self.soft_fallback_entries = 0
        self.recoveries = 0
        self.hard_stall_count = 0
        self.max_run_age_ms = 0.0
        self.hard_stall_sequences = []
        self.presentation_stale_drops = 0
        self.recovery_stale_drops = 0
        self._hard_stall_reported_sequence: Optional[int] = None
        self._hard_stall_returned_sequences = set()
        self._last_drop_counts = collections.Counter()

    @property
    def fallback_active(self) -> bool:
        return self.state in (self.SOFT_FALLBACK, self.HARD_STALLED)

    def _transition(self, state: str, now: float, good_age_ms: Optional[float],
                    run_age_ms: float, soft_threshold_ms: float) -> Optional[dict]:
        if self.state == state:
            return None
        previous = self.state
        if state in (self.SOFT_FALLBACK, self.HARD_STALLED):
            if self.fallback_started_at is None:
                self.fallback_started_at = now
        elif state == self.HEALTHY:
            self.fallback_started_at = None
            self.fallback_displayed_sequence = None
        self.state = state
        if state == self.SOFT_FALLBACK:
            self.soft_fallback_entries += 1
            return {
                "TYPE": "GAN_STATE",
                "STATE": self.SOFT_FALLBACK,
                "PREVIOUS_STATE": previous,
                "AT": now,
                "GOOD_AGE_MS": good_age_ms,
                "RUN_AGE_MS": run_age_ms,
                "SOFT_THRESHOLD_MS": soft_threshold_ms,
                "EFFECTIVE_BUDGET_MS": self.output_budget_ms,
            }
        if state == self.HARD_STALLED:
            return {
                "TYPE": "GAN_STATE",
                "STATE": self.HARD_STALLED,
                "PREVIOUS_STATE": previous,
                "AT": now,
                "GOOD_AGE_MS": good_age_ms,
                "RUN_AGE_MS": run_age_ms,
                "HARD_STALL_MS": self.hard_stall_ms,
                "EFFECTIVE_BUDGET_MS": self.output_budget_ms,
            }
        return None

    def note_lanczos_display(self, source_sequence: Optional[int]) -> None:
        """Remember source identities already shown by the live fallback."""
        if source_sequence is None:
            return
        sequence = int(source_sequence)
        if (self.fallback_displayed_sequence is None or
                sequence > self.fallback_displayed_sequence):
            self.fallback_displayed_sequence = sequence

    def _reset_recovery_on_worker_drops(self, worker_snapshot: dict) -> Tuple[str, ...]:
        counts = worker_snapshot.get("drop_reason_counts", {}) or {}
        resets = []
        for reason in (
            "stale_before_infer",
            "stale_after_infer",
            "inference_error",
        ):
            current = int(counts.get(reason, 0))
            if current > int(self._last_drop_counts.get(reason, 0)):
                self.recovery_streak = 0
                resets.append(reason)
        # ``stale_before_predicted`` has no GAN result at all: it deliberately
        # avoided a provider call.  It must not erase already observed fresh
        # recovery outputs, or the predictive latest-only policy could make
        # the three-good-output recovery target impossible to reach.
        for key, value in counts.items():
            self._last_drop_counts[key] = int(value)
        return tuple(resets)

    def _hard_return_event(self, now: float, worker_snapshot: dict) -> Optional[dict]:
        sequence = worker_snapshot.get("last_finished_sequence")
        if sequence is None or worker_snapshot.get("running"):
            return None
        sequence = int(sequence)
        if (self._hard_stall_reported_sequence != sequence or
                sequence in self._hard_stall_returned_sequences):
            return None
        self._hard_stall_returned_sequences.add(sequence)
        return {
            "TYPE": "GAN_HARD_STALL_RETURNED",
            "SEQ": sequence,
            "AT": now,
            "RUN_AGE_MS": float(worker_snapshot.get("last_finished_infer_ms", 0.0)),
            "HARD_STALL_MS": self.hard_stall_ms,
        }

    def _recovery_freshness(
        self,
        sequence: int,
        latest_sequence: Optional[int],
        output_age_ms: float,
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """Separate recovery evidence from whether an output can be painted.

        A one-frame lag may be safe evidence that CUDA has recovered, even if
        a newer live Lanczos frame has already reached the display.  The
        presentation rule is deliberately evaluated separately below.
        """
        lag = None if latest_sequence is None else int(latest_sequence) - sequence
        if output_age_ms > self.output_budget_ms:
            return False, lag, "RECOVERY_TOO_OLD"
        if lag is None:
            return False, lag, "RECOVERY_SEQUENCE_UNKNOWN"
        if lag < 0:
            return False, lag, "RECOVERY_SEQUENCE_FUTURE"
        if lag > self.recovery_max_sequence_lag:
            return False, lag, "RECOVERY_SEQUENCE_OLD"
        return True, lag, None

    def update(
        self,
        now: float,
        source_fps: float,
        latest_sequence: Optional[int],
        worker_snapshot: dict,
        candidate: Optional[Any] = None,
    ) -> GanControllerDecision:
        """Consume one renderer tick and at most one completed GAN output."""
        now = float(now)
        fps = max(1.0, float(source_fps))
        soft_threshold_ms = gan_fallback_stale_after_ms(
            fps, self.output_budget_ms)
        running_age_ms = max(0.0, float(worker_snapshot.get("running_age_ms", 0.0)))
        self.current_run_age_ms = running_age_ms
        self.max_run_age_ms = max(self.max_run_age_ms, running_age_ms)
        worker_drop_resets = self._reset_recovery_on_worker_drops(worker_snapshot)
        events = []
        if worker_drop_resets and self.fallback_active:
            events.append({
                "TYPE": "GAN_RECOVERY_RESET",
                "REASON": "WORKER_" + "+".join(worker_drop_resets).upper(),
                "AT": now,
            })

        return_event = self._hard_return_event(now, worker_snapshot)
        if return_event is not None:
            events.append(return_event)

        running_sequence = worker_snapshot.get("running_sequence")
        if (running_age_ms >= self.hard_stall_ms and running_sequence is not None and
                int(running_sequence) != self._hard_stall_reported_sequence):
            sequence = int(running_sequence)
            self._hard_stall_reported_sequence = sequence
            self.hard_stall_count += 1
            self.hard_stall_sequences.append(sequence)
            event = {
                "TYPE": "GAN_HARD_STALL",
                "SEQ": sequence,
                "AT": now,
                "RUN_AGE_MS": running_age_ms,
                "HARD_STALL_MS": self.hard_stall_ms,
            }
            events.append(event)
            transition = self._transition(
                self.HARD_STALLED, now,
                None if self.last_good_completed_at is None else
                max(0.0, (now - self.last_good_completed_at) * 1000.0),
                running_age_ms, soft_threshold_ms)
            if transition is not None:
                events.append(transition)

        good_age_ms = (
            None if self.last_good_completed_at is None else
            max(0.0, (now - self.last_good_completed_at) * 1000.0)
        )
        output_age_ms: Optional[float] = None
        if candidate is not None:
            arrived_at = getattr(candidate, "arrived_at", None)
            if arrived_at is not None:
                self.last_output_arrived_at = float(arrived_at)
                output_age_ms = max(0.0, (now - self.last_output_arrived_at) * 1000.0)
            else:
                output_age_ms = max(0.0, float(getattr(candidate, "total_pc_ms", 0.0)))
            self.last_output_age_ms = output_age_ms
            sequence = int(getattr(candidate, "source_sequence", -1))
            fresh_for_recovery, lag, freshness_reason = self._recovery_freshness(
                sequence, latest_sequence, output_age_ms)
            # Do not mistake a same-source GAN result for an old frame.  It
            # may upgrade the currently displayed Lanczos version of exactly
            # that source.  A strictly older source still cannot overwrite a
            # newer fallback frame, but can remain recovery evidence if its
            # lag is within the configured recovery allowance.
            safe_to_present = (
                self.fallback_displayed_sequence is None or
                sequence >= self.fallback_displayed_sequence
            )
            if not fresh_for_recovery:
                self.presentation_stale_drops += 1
                if self.fallback_active:
                    self.recovery_stale_drops += 1
                    self.recovery_streak = 0
                    reason = str(freshness_reason or "RECOVERY_NOT_FRESH")
                else:
                    reason = "OUTPUT_TOO_OLD"
                events.append({
                    "TYPE": "GAN_PRESENTATION_DROP",
                    "REASON": reason,
                    "SEQ": sequence,
                    "AT": now,
                    "SEQUENCE_LAG": lag,
                    "OUTPUT_AGE_MS": output_age_ms,
                    "EFFECTIVE_BUDGET_MS": self.output_budget_ms,
                    "MAX_SEQUENCE_LAG": self.recovery_max_sequence_lag,
                    "FRESH_FOR_RECOVERY": False,
                    "SAFE_TO_PRESENT": safe_to_present,
                })
                return GanControllerDecision(
                    GanAction.DISCARD_STALE_GAN, self.state, good_age_ms, running_age_ms,
                    soft_threshold_ms, output_age_ms, self.recovery_streak,
                    tuple(events),
                )

            self.last_good_completed_at = float(candidate.completed_at)
            self.last_good_sequence = sequence
            good_age_ms = max(0.0, (now - self.last_good_completed_at) * 1000.0)
            if self.fallback_active:
                self.recovery_streak += 1
                if not safe_to_present:
                    # This result is still valid evidence that the provider
                    # caught up (for example lag=1), but a newer Lanczos source
                    # is already visible and must remain immutable.
                    self.presentation_stale_drops += 1
                    events.append({
                        "TYPE": "GAN_PRESENTATION_DROP",
                        "REASON": "PRESENTATION_SEQUENCE_OLD",
                        "SEQ": sequence,
                        "AT": now,
                        "SEQUENCE_LAG": lag,
                        "OUTPUT_AGE_MS": output_age_ms,
                        "EFFECTIVE_BUDGET_MS": self.output_budget_ms,
                        "MAX_SEQUENCE_LAG": self.recovery_max_sequence_lag,
                        "FRESH_FOR_RECOVERY": True,
                        "SAFE_TO_PRESENT": False,
                    })
                if self.recovery_streak >= self.recovery_good_frames:
                    previous = self.state
                    fallback_duration_ms = (
                        0.0 if self.fallback_started_at is None else
                        max(0.0, (now - self.fallback_started_at) * 1000.0)
                    )
                    self.recoveries += 1
                    self.state = self.HEALTHY
                    self.fallback_started_at = None
                    self.fallback_displayed_sequence = None
                    completed_recovery_streak = self.recovery_streak
                    self.recovery_streak = 0
                    events.append({
                        "TYPE": "GAN_STATE",
                        "STATE": "RECOVERED",
                        "PREVIOUS_STATE": previous,
                        "AT": now,
                        "RECOVERY_STREAK": completed_recovery_streak,
                        "GOOD_AGE_MS": good_age_ms,
                        "FALLBACK_DURATION_MS": fallback_duration_ms,
                        "FALLBACK_ACTIVE": False,
                    })
                    return GanControllerDecision(
                        (GanAction.ACCEPT_GAN if safe_to_present else GanAction.KEEP_GAN),
                        self.state, good_age_ms,
                        running_age_ms, soft_threshold_ms, output_age_ms,
                        self.recovery_streak, tuple(events),
                    )
                return GanControllerDecision(
                    (GanAction.ACCEPT_GAN if safe_to_present else GanAction.KEEP_GAN),
                    self.state, good_age_ms, running_age_ms,
                    soft_threshold_ms, output_age_ms, self.recovery_streak,
                    tuple(events),
                )
            return GanControllerDecision(
                GanAction.ACCEPT_GAN, self.state, good_age_ms, running_age_ms,
                soft_threshold_ms, output_age_ms, self.recovery_streak,
                tuple(events),
            )

        good_age_ms = (
            None if self.last_good_completed_at is None else
            max(0.0, (now - self.last_good_completed_at) * 1000.0)
        )
        if (self.state == self.HEALTHY and good_age_ms is not None and
                good_age_ms >= soft_threshold_ms):
            transition = self._transition(
                self.SOFT_FALLBACK, now, good_age_ms, running_age_ms,
                soft_threshold_ms)
            if transition is not None:
                events.append(transition)
        action = (
            GanAction.USE_LANCZOS if self.fallback_active or
            self.last_good_completed_at is None else GanAction.KEEP_GAN
        )
        return GanControllerDecision(
            action, self.state, good_age_ms, running_age_ms,
            soft_threshold_ms, output_age_ms, self.recovery_streak,
            tuple(events),
        )

    def snapshot(self, now: Optional[float] = None, source_fps: float = 0.0) -> dict:
        now = time.monotonic() if now is None else float(now)
        good_age_ms = (
            None if self.last_good_completed_at is None else
            max(0.0, (now - self.last_good_completed_at) * 1000.0)
        )
        soft_threshold_ms = gan_fallback_stale_after_ms(
            max(1.0, float(source_fps)), self.output_budget_ms)
        fallback_duration_ms = (
            0.0 if self.fallback_started_at is None else
            max(0.0, (now - self.fallback_started_at) * 1000.0)
        )
        last_output_age_ms = self.last_output_age_ms
        if self.last_output_arrived_at is not None:
            last_output_age_ms = max(0.0, (now - self.last_output_arrived_at) * 1000.0)
        return {
            "state": self.state,
            "fallback_active": self.fallback_active,
            "output_budget_ms": self.output_budget_ms,
            "soft_threshold_ms": soft_threshold_ms,
            "hard_stall_ms": self.hard_stall_ms,
            "good_age_ms": good_age_ms,
            "run_age_ms": self.max_run_age_ms,
            "current_run_age_ms": self.current_run_age_ms,
            "last_output_age_ms": last_output_age_ms,
            "recovery_streak": self.recovery_streak,
            "recovery_good_frames": self.recovery_good_frames,
            "recovery_max_sequence_lag": self.recovery_max_sequence_lag,
            "soft_fallback_entries": self.soft_fallback_entries,
            "recoveries": self.recoveries,
            "hard_stall_count": self.hard_stall_count,
            "hard_stall_max_run_age_ms": self.max_run_age_ms,
            "hard_stall_sequences": list(self.hard_stall_sequences),
            "presentation_stale_drops": self.presentation_stale_drops,
            "recovery_stale_drops": self.recovery_stale_drops,
            "fallback_duration_ms": fallback_duration_ms,
            "last_good_sequence": self.last_good_sequence,
            "fallback_displayed_sequence": self.fallback_displayed_sequence,
        }

    def finish(self, now: Optional[float] = None) -> dict:
        """Return a final event so an active fallback has a closed interval."""
        now = time.monotonic() if now is None else float(now)
        duration_ms = (
            0.0 if self.fallback_started_at is None else
            max(0.0, (now - self.fallback_started_at) * 1000.0)
        )
        return {
            "TYPE": "GAN_RUN_END",
            "AT": now,
            "STATE": self.state,
            "FALLBACK_ACTIVE": self.fallback_active,
            "FALLBACK_DURATION_MS": duration_ms,
            "SOFT_FALLBACK_ENTRIES": self.soft_fallback_entries,
            "RECOVERIES": self.recoveries,
            "HARD_STALLS": self.hard_stall_count,
        }


def should_use_gan_fallback(frame: Optional[np.ndarray], source_sequence: Optional[int],
                            current_source_sequence: Optional[int], current_updated: float,
                            fallback_active: bool, now: float, source_fps: float,
                            max_latency_ms: float) -> bool:
    """Return whether the current GAN canvas must be refreshed from decoded video."""
    if frame is None or source_sequence is None:
        return False
    if current_source_sequence is None:
        return True
    if fallback_active and source_sequence != current_source_sequence:
        return True
    return (now - current_updated) * 1000.0 >= gan_fallback_stale_after_ms(
        source_fps, max_latency_ms)


def gan_lanczos_fallback(frame: np.ndarray, rotation: str) -> np.ndarray:
    """Keep GAN video live while an asynchronous neural enhancer is stalled."""
    output = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LANCZOS4)
    return postprocess_frame(output, rotation, False)


def display_dimensions(width: int, height: int, rotation: str, scale: int) -> Tuple[int, int]:
    """Return the OpenCV window size after display rotation."""
    if rotation in ("cw90", "ccw90"):
        width, height = height, width
    return width * scale, height * scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sdp", nargs="?", type=Path,
        help="optional legacy H.265 SDP; the HUD only reads its UDP video port",
    )
    parser.add_argument(
        "--video-port", "--udp-port", dest="video_port", type=int, default=None,
        help=f"H.265 RTP UDP listen port; default {DEFAULT_VIDEO_PORT} when SDP is omitted",
    )
    parser.add_argument("--width", type=int, default=320,
                        help="startup canvas width before the first decoded frame")
    parser.add_argument("--height", type=int, default=180,
                        help="startup canvas height before the first decoded frame")
    parser.add_argument("--scale", type=int, default=0,
                        help="window scale; 0 selects 1x for rebuild and 3x otherwise")
    parser.add_argument("--rotate", choices=("none", "cw90", "ccw90", "180"), default="ccw90",
                        help="display rotation; default is counter-clockwise 90 degrees")
    parser.add_argument("--denoise", choices=("on", "off"), default="on",
                        help="display-only edge-preserving smoothing")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--rebuild-port", type=int, default=0,
                        help="RB/1 UDP port; 0 uses video RTP port + 5")
    parser.add_argument("--rebuild-width", type=int, default=640)
    parser.add_argument("--rebuild-height", type=int, default=360)
    parser.add_argument("--rebuild-fps", type=float, default=12.0)
    parser.add_argument("--rebuild-reference-max-age", type=float, default=1.0)
    parser.add_argument("--rebuild-max-sync-ms", type=int, default=100,
                        help="suppress ROI paint when RB/1 state differs from video PTS")
    parser.add_argument("--esrgan", choices=("auto", "off"), default="auto",
                        help="small-ROI Real-ESRGAN; auto falls back to Lanczos4")
    parser.add_argument("--esrgan-model", default=None)
    parser.add_argument("--esrgan-threads", type=int, default=2)
    parser.add_argument("--gan-enhancer", choices=("none", "esrnet", "esrgan"), default="none",
                        help="GAN profile full-frame backend; none is the Lanczos4 baseline")
    parser.add_argument("--gan-esrnet-model", default=None,
                        help="Real-ESRNet x2 ONNX model; required with --gan-enhancer=esrnet")
    parser.add_argument(
        "--gan-esrgan-model",
        default=str(Path(__file__).resolve().parents[1] / "model" / "RealESRGAN_x2_dynamic.onnx"),
        help="Real-ESRGAN x2 ONNX model; used with --gan-enhancer=esrgan",
    )
    parser.add_argument("--gan-require-cuda", action="store_true",
                        help="fail instead of falling back to CPU for a GAN model")
    parser.add_argument("--gan-threads", type=int, default=2)
    parser.add_argument("--gan-warmup", type=int, default=4,
                        help="live-shape model warmup frames before output")
    parser.add_argument(
        "--gan-output-latency-budget-ms", type=float, default=0.0,
        help="accepted source-to-output age budget; 0 selects FPS-aware AUTO",
    )
    parser.add_argument(
        "--gan-output-latency-factor", type=float, default=1.15,
        help="AUTO output budget multiplier over one source-frame period",
    )
    parser.add_argument(
        "--gan-max-inference-latency-ms", type=float, default=None,
        help="deprecated fixed output budget override (use --gan-output-latency-budget-ms)",
    )
    parser.add_argument(
        "--gan-hard-stall-ms", type=float, default=500.0,
        help="RUN_AGE threshold used to report a hard enhancer stall",
    )
    parser.add_argument(
        "--gan-recovery-good-frames", type=int, default=3,
        help="fresh consecutive GAN outputs required to leave fallback",
    )
    parser.add_argument(
        "--gan-recovery-max-sequence-lag", type=int, default=1,
        help="maximum decoded-frame sequence lag for GAN presentation/recovery",
    )
    parser.add_argument("--gan-display-fps", type=float, default=0.0,
                        help="presentation cadence; 0 follows the encoded source fps")
    parser.add_argument("--gan-debug-log", default=None,
                        help="JSONL per-frame enhancer telemetry path")
    parser.add_argument("--rebuild-boxes", choices=("on", "off"), default="off",
                        help="diagnostic target rectangles; off avoids display overlays")
    parser.add_argument("--headless", action="store_true", help="print HUD metrics without opening a window")
    parser.add_argument("--duration", type=float, default=0.0, help="optional run duration in seconds")
    args = parser.parse_args()
    if (args.width <= 0 or args.height <= 0 or args.scale < 0 or
            args.rebuild_width <= 0 or args.rebuild_height <= 0 or
            args.rebuild_fps <= 0 or args.rebuild_reference_max_age <= 0 or
            args.rebuild_max_sync_ms <= 0 or
            args.esrgan_threads < 0 or args.gan_threads < 0 or args.gan_warmup < 0 or
            args.gan_output_latency_budget_ms < 0 or
            args.gan_output_latency_factor <= 0 or
            (args.gan_max_inference_latency_ms is not None and
             args.gan_max_inference_latency_ms <= 0) or
            args.gan_hard_stall_ms <= 0 or args.gan_recovery_good_frames <= 0 or
            args.gan_recovery_max_sequence_lag < 0 or
            args.gan_display_fps < 0):
        parser.error("dimensions/rates/ages must be positive and scale/threads non-negative")

    try:
        listen_port = resolve_video_port(args.sdp, args.video_port)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    input_source = f"SDP {args.sdp}" if args.sdp is not None else "SDP-free"
    print(
        f"H.265 RTP input: UDP {listen_port}, payload type 96, clock 90000 ({input_source})",
        flush=True,
    )
    stats = RtpStats()
    switch_gate = ProfileSwitchGate()
    depacketizer = HevcRtpDepacketizer()
    latest = LatestFrame()
    decoded_pts = DecodedPtsQueue()
    stopping = threading.Event()
    rebuild_port = args.rebuild_port or listen_port + 5
    if rebuild_port < 1 or rebuild_port > 65535 or rebuild_port == listen_port:
        parser.error("rebuild port must be a valid UDP port distinct from video RTP")
    # Rebuild objects are lazy.  In particular, the GAN receiver must not bind
    # the RB/1 port or instantiate any semantic/reference compositor.
    rebuild_receiver: Optional[RebuildReceiver] = None
    resolver: Optional[SuperResolver] = None
    rebuild_composer: Optional[RebuildComposer] = None

    def ensure_rebuild_components() -> bool:
        nonlocal rebuild_receiver, resolver, rebuild_composer
        if rebuild_receiver is not None and rebuild_composer is not None:
            return True
        candidate_receiver = RebuildReceiver(
            rebuild_port, max_sync_ms=args.rebuild_max_sync_ms)
        candidate_resolver = SuperResolver(
            enabled=False, model_path=args.esrgan_model,
            cpu_threads=args.esrgan_threads)
        candidate_composer = RebuildComposer(
            output_size=(args.rebuild_width, args.rebuild_height), resolver=candidate_resolver,
            reference_max_age=args.rebuild_reference_max_age,
            draw_targets=args.rebuild_boxes == "on")
        try:
            candidate_receiver.start(stopping)
        except OSError as error:
            candidate_composer.close()
            print(f"cannot bind RB/1 UDP {rebuild_port}: {error}", file=sys.stderr)
            return False
        rebuild_receiver = candidate_receiver
        resolver = candidate_resolver
        rebuild_composer = candidate_composer
        print(f"RB/1 rebuild companion listening on UDP {rebuild_port}", flush=True)
        return True

    def disable_rebuild_components() -> None:
        nonlocal rebuild_receiver, resolver, rebuild_composer
        if rebuild_receiver is not None:
            rebuild_receiver.close()
        if rebuild_composer is not None:
            rebuild_composer.close()
        rebuild_receiver = None
        rebuild_composer = None
        resolver = None

    presentation = PresentationStats()

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    receiver.bind(("0.0.0.0", listen_port))
    receiver.settimeout(0.2)
    frame_rate_args = ffmpeg_frame_rate_args(args.ffmpeg)
    if frame_rate_args:
        print(f"FFmpeg frame pacing: {' '.join(frame_rate_args)}", flush=True)
    else:
        print("FFmpeg frame pacing: default (no compatible passthrough option)", flush=True)
    command = [
        args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning",
        "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "1024",
        "-f", "hevc",
        "-i", "pipe:0", "-an", "-sn", "-dn",
        *frame_rate_args, "-pix_fmt", "bgr24",
        "-f", "image2pipe", "-vcodec", "bmp", "pipe:1",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    decoder_lock = threading.Lock()
    decoder_state = {
        "process": None, "generation": None, "profile_key": None, "token": 0,
    }
    decoder_threads = []
    gan_worker_lock = threading.Lock()
    gan_worker_holder = {"worker": None, "signature": None}
    gan_controller: Optional[GanFallbackController] = None

    def set_gan_worker(
        worker: Optional[LatestOnlyEnhancerWorker],
        signature: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        with gan_worker_lock:
            gan_worker_holder["worker"] = worker
            gan_worker_holder["signature"] = signature

    def get_gan_worker(profile: Optional[dict]) -> Optional[LatestOnlyEnhancerWorker]:
        with gan_worker_lock:
            worker = gan_worker_holder["worker"]
            signature = gan_worker_holder["signature"]
        if worker is None or profile is None:
            return None
        try:
            return worker if signature == gan_profile_signature(profile) else None
        except (TypeError, ValueError):
            return None

    def gan_profile_decoder_ready(profile: Optional[dict]) -> bool:
        """Only bind a worker after ProfileSwitchGate accepted a full IDR."""
        if profile is None:
            return False
        try:
            generation = int(profile["generation"])
        except (KeyError, TypeError, ValueError):
            return False
        with decoder_lock:
            return (decoder_state["generation"] == generation and
                    decoder_state["profile_key"] == profile_switch_key(profile))

    gan_worker: Optional[LatestOnlyEnhancerWorker] = None
    gan_worker_signature: Optional[Tuple[int, int, int, int]] = None

    def ensure_gan_worker(profile: dict) -> bool:
        nonlocal gan_worker, gan_worker_signature, gan_controller
        try:
            signature = gan_profile_signature(profile)
            input_size, native_size, output_size = gan_profile_sizes(profile)
        except (TypeError, ValueError) as error:
            print(f"GAN enhancer profile invalid: {error}", file=sys.stderr)
            return False
        if gan_worker is not None:
            return gan_worker_signature == signature
        model_path = (
            args.gan_esrnet_model
            if args.gan_enhancer == "esrnet"
            else args.gan_esrgan_model
        )
        try:
            output_budget_ms = gan_effective_output_budget_ms(
                float(profile["fps"]),
                requested_budget_ms=args.gan_output_latency_budget_ms,
                latency_factor=args.gan_output_latency_factor,
                legacy_budget_ms=args.gan_max_inference_latency_ms,
            )
        except ValueError as error:
            print(f"GAN latency budget invalid: {error}", file=sys.stderr)
            return False
        try:
            print(
                f"GAN WARMING: backend={args.gan_enhancer} "
                f"input={input_size[0]}x{input_size[1]} "
                f"warmup={args.gan_warmup}",
                flush=True,
            )
            enhancer = FullFrameEnhancer(
                args.gan_enhancer,
                model_path=model_path,
                input_size=input_size,
                native_size=native_size,
                output_size=output_size,
                require_cuda=args.gan_require_cuda,
                threads=args.gan_threads,
                warmup=args.gan_warmup,
            )
            worker = LatestOnlyEnhancerWorker(
                enhancer,
                max_latency_ms=output_budget_ms,
                debug_log=args.gan_debug_log,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"GAN enhancer initialization failed: {error}", file=sys.stderr)
            return False
        gan_worker = worker
        gan_worker_signature = signature
        gan_controller = GanFallbackController(
            output_budget_ms=output_budget_ms,
            hard_stall_ms=args.gan_hard_stall_ms,
            recovery_good_frames=args.gan_recovery_good_frames,
            recovery_max_sequence_lag=args.gan_recovery_max_sequence_lag,
        )
        set_gan_worker(worker, signature)
        budget_mode = (
            f"LEGACY_FIXED {output_budget_ms:.1f}ms"
            if args.gan_max_inference_latency_ms is not None else
            f"AUTO {output_budget_ms:.1f}ms factor={args.gan_output_latency_factor:.2f}"
            if args.gan_output_latency_budget_ms <= 0.0 else
            f"FIXED {output_budget_ms:.1f}ms"
        )
        print(
            f"GAN READY: full-frame enhancer ready: backend={worker.backend} "
            f"provider={worker.provider} input={input_size[0]}x{input_size[1]} "
            f"native={native_size[0]}x{native_size[1]} "
            f"output={output_size[0]}x{output_size[1]} "
            f"output_budget_ms={output_budget_ms:.1f} ({budget_mode}) "
            f"hard_stall_ms={args.gan_hard_stall_ms:.0f} "
            f"recovery={args.gan_recovery_good_frames}frames/"
            f"lag<={args.gan_recovery_max_sequence_lag}",
            flush=True,
        )
        return True

    def disable_gan_worker() -> None:
        nonlocal gan_worker, gan_worker_signature, gan_controller
        worker = gan_worker
        controller = gan_controller
        gan_worker = None
        gan_worker_signature = None
        gan_controller = None
        set_gan_worker(None)
        if worker is not None:
            if controller is not None:
                worker.debug_event(controller.finish())
            worker.stop()

    def decode_loop(process: subprocess.Popen, token: int) -> None:
        assert process.stdout is not None
        while not stopping.is_set():
            frame = read_bmp_frame(process.stdout)
            if frame is None:
                break
            with decoder_lock:
                current = token == decoder_state["token"]
            if current:
                rtp_timestamp = decoded_pts.pop()
                source_sequence = latest.put(frame, rtp_timestamp)
                stats.on_decoded_frame()
                profile = stats.snapshot()["profile"]
                worker = get_gan_worker(profile)
                if worker is not None and profile is not None:
                    worker.submit(
                        frame, source_sequence, rtp_timestamp,
                        input_fps=float(profile["fps"]),
                        generation=int(profile["generation"]),
                    )

    def stderr_loop(process: subprocess.Popen) -> None:
        assert process.stderr is not None
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode(errors="replace").rstrip()
            if not line:
                continue
            print(f"[ffmpeg] {line}")
            if "Error" in line or "Could not find ref" in line or "Invalid" in line:
                stats.on_decode_error()

    def launch_decoder_locked() -> None:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
            creationflags=creationflags,
        )
        decoder_state["token"] += 1
        token = decoder_state["token"]
        decoder_state["process"] = process
        decode_thread = threading.Thread(
            target=decode_loop, args=(process, token), name=f"hevc-decode-{token}", daemon=True)
        error_thread = threading.Thread(
            target=stderr_loop, args=(process,), name=f"ffmpeg-stderr-{token}", daemon=True)
        decoder_threads.extend((decode_thread, error_thread))
        decode_thread.start()
        error_thread.start()

    def stop_decoder_locked() -> None:
        process = decoder_state["process"]
        decoder_state["process"] = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def ensure_decoder(generation: Optional[int], profile: Optional[dict]) -> None:
        next_profile_key = profile_switch_key(profile)
        with decoder_lock:
            process = decoder_state["process"]
            changed = (decoder_state["profile_key"] is not None and
                       next_profile_key != decoder_state["profile_key"])
            dead = process is None or process.poll() is not None
            if changed or dead:
                stop_decoder_locked()
                latest.clear()
                decoded_pts.clear()
                decoder_state["generation"] = generation
                decoder_state["profile_key"] = next_profile_key
                launch_decoder_locked()
            elif decoder_state["generation"] is None:
                decoder_state["generation"] = generation
                decoder_state["profile_key"] = next_profile_key

    def write_decoder(data: bytes) -> None:
        if not data:
            return
        with decoder_lock:
            process = decoder_state["process"]
            if process is None or process.poll() is not None or process.stdin is None:
                return
            try:
                process.stdin.write(data)
            except (BrokenPipeError, OSError):
                pass

    def proxy_loop() -> None:
        while not stopping.is_set():
            try:
                packet, _ = receiver.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            stats.on_packet(packet)
            profile = stats.snapshot()["profile"]
            restart_generation, ready_packets = switch_gate.feed(packet, profile)
            if restart_generation is not None:
                nal_types = sorted({
                    nal_type for nal_type in (RtpStats._nal_type(item) for item in ready_packets)
                    if nal_type is not None
                })
                print(f"Profile generation {restart_generation}: replay IDR RTP NAL types {nal_types}")
                ensure_decoder(restart_generation, profile)
                depacketizer.reset()
            for ready_packet in ready_packets:
                annex_b = depacketizer.feed(ready_packet)
                if annex_b and ready_packet[1] & 0x80:
                    decoded_pts.push_rtp_timestamp(
                        int.from_bytes(ready_packet[4:8], "big"))
                write_decoder(annex_b)

    threads = [
        threading.Thread(target=proxy_loop, name="rtp-proxy", daemon=True),
    ]
    for thread in threads:
        thread.start()

    window = "RK3588 H.265 live HUD (q/Esc to quit)"
    startup_scale = args.scale or 3
    window_size = display_dimensions(args.width, args.height, args.rotate, startup_scale)
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *window_size)
    started_at = time.monotonic()
    last_report = 0.0
    next_rebuild_present = started_at
    next_gan_present = started_at
    current_canvas: Optional[np.ndarray] = None
    current_updated = 0.0
    current_source_sequence: Optional[int] = None
    current_mode = "normal"
    gan_display_fallback = False
    try:
        while not stopping.is_set():
            now = time.monotonic()
            sync_ms: Optional[int] = None
            if args.duration > 0 and now - started_at >= args.duration:
                break
            frame, updated, source_sequence, source_rtp_timestamp = latest.get_versioned()
            values = stats.snapshot()
            profile = values["profile"]
            is_rebuild = profile is not None and profile["name"] == "rebuild"
            is_gan = profile is not None and profile["name"] == "gan"
            requested_mode = "gan" if is_gan else ("rebuild" if is_rebuild else "normal")
            if current_mode != requested_mode:
                previous_mode = current_mode
                current_canvas = None
                current_source_sequence = None
                next_rebuild_present = now
                next_gan_present = now
                current_mode = requested_mode
                gan_display_fallback = False
                if previous_mode == "rebuild" and not is_rebuild:
                    disable_rebuild_components()
                if previous_mode == "gan" and not is_gan:
                    disable_gan_worker()
                if is_rebuild:
                    if not ensure_rebuild_components():
                        stopping.set()
                        break
                    if args.esrgan != "off" and resolver is not None:
                        resolver.enable_async()

            # The profile record can arrive before the new IDR is complete.
            # Never bind the old worker to that metadata (or display its last
            # output over a new-size source); wait until ProfileSwitchGate has
            # restarted FFmpeg on the complete IDR, then replace the worker.
            if is_gan and profile is not None:
                try:
                    wanted_gan_signature = gan_profile_signature(profile)
                except (TypeError, ValueError) as error:
                    print(f"GAN profile invalid: {error}", file=sys.stderr)
                    stopping.set()
                    break
                needs_gan_worker = (
                    gan_worker is None or gan_worker_signature != wanted_gan_signature
                )
                if gan_worker_signature != wanted_gan_signature:
                    current_canvas = None
                    current_source_sequence = None
                    next_gan_present = now
                    gan_display_fallback = False
                if needs_gan_worker and gan_profile_decoder_ready(profile):
                    if gan_worker is not None:
                        disable_gan_worker()
                    if not ensure_gan_worker(profile):
                        stopping.set()
                        break

            # A completed RB/1 reference reaches this queue on the socket
            # thread.  Give Real-ESRGAN the crop immediately; do not wait for
            # the 12 fps renderer to find a STATE/PTS-valid moment to paint
            # it.  The compositor still applies all existing fail-closed
            # generation, STATE, PTS and registration checks at blend time.
            if is_rebuild and rebuild_receiver is not None and rebuild_composer is not None:
                for reference in rebuild_receiver.take_completed_references():
                    rebuild_composer.prefetch(reference)
                rebuild_composer.prefetch_pending()

            worker = get_gan_worker(profile) if is_gan else None
            if is_gan and worker is not None:
                newest_output = None
                while True:
                    output = worker.poll_output()
                    if output is None:
                        break
                    if output.generation != int(profile["generation"]):
                        worker.debug_event({
                            "TYPE": "GAN_GENERATION_DROP",
                            "SEQ": output.source_sequence,
                            "OUTPUT_GENERATION": output.generation,
                            "ACTIVE_GENERATION": int(profile["generation"]),
                            "AT": now,
                        })
                        continue
                    newest_output = output
                worker_values = worker.snapshot()
                if gan_controller is None:
                    # This should only be reachable during profile startup;
                    # keep the display path fail-safe if initialization races
                    # with the first decoded frame.
                    gan_controller = GanFallbackController(
                        output_budget_ms=float(worker_values["max_latency_ms"]),
                        hard_stall_ms=args.gan_hard_stall_ms,
                        recovery_good_frames=args.gan_recovery_good_frames,
                        recovery_max_sequence_lag=args.gan_recovery_max_sequence_lag,
                    )
                decision = gan_controller.update(
                    now=now,
                    source_fps=float(profile["fps"]),
                    latest_sequence=source_sequence,
                    worker_snapshot=worker_values,
                    candidate=newest_output,
                )
                fallback_entered_this_tick = any(
                    event.get("TYPE") == "GAN_STATE" and
                    event.get("STATE") in (
                        GanFallbackController.SOFT_FALLBACK,
                        GanFallbackController.HARD_STALLED,
                    )
                    for event in decision.events
                )
                for event in decision.events:
                    worker.debug_event(event)
                    event_type = event.get("TYPE")
                    if event_type == "GAN_STATE" and event.get("STATE") == "SOFT_FALLBACK":
                        print(
                            "GAN output overdue GOOD_AGE=%.0fms SOFT_THRESHOLD=%.0fms "
                            "RUN_AGE=%.0fms; using live Lanczos4 fallback" % (
                                float(event.get("GOOD_AGE_MS") or 0.0),
                                float(event.get("SOFT_THRESHOLD_MS") or 0.0),
                                float(event.get("RUN_AGE_MS") or 0.0),
                            ),
                            flush=True,
                        )
                    elif event_type == "GAN_HARD_STALL":
                        print(
                            "GAN HARD STALL seq=%s RUN_AGE=%.0fms threshold=%.0fms; "
                            "fallback remains Lanczos4" % (
                                event.get("SEQ"),
                                float(event.get("RUN_AGE_MS") or 0.0),
                                float(event.get("HARD_STALL_MS") or 0.0),
                            ),
                            flush=True,
                        )
                    elif event_type == "GAN_HARD_STALL_RETURNED":
                        print(
                            "GAN hard-stall seq=%s returned after RUN_AGE=%.0fms" % (
                                event.get("SEQ"),
                                float(event.get("RUN_AGE_MS") or 0.0),
                            ),
                            flush=True,
                        )
                    elif event_type == "GAN_STATE" and event.get("STATE") == "RECOVERED":
                        print(
                            "GAN recovered after %d consecutive fresh outputs" %
                            int(event.get("RECOVERY_STREAK") or 0),
                            flush=True,
                        )

                if decision.action == GanAction.ACCEPT_GAN and newest_output is not None:
                    current_canvas = postprocess_frame(
                        newest_output.frame, args.rotate, False)
                    current_updated = newest_output.completed_at
                    current_source_sequence = newest_output.source_sequence
                    gan_display_fallback = False

                # Once fallback is active, every newly decoded source frame is
                # painted through the live Lanczos path.  A fresh, safe GAN
                # result may upgrade the same source frame while recovery is
                # still collecting its required consecutive samples; retain it
                # until a newer decoded source needs the live fallback.
                if gan_controller.fallback_active and frame is not None and source_sequence is not None:
                    if (current_canvas is None or
                            source_sequence != current_source_sequence or
                            fallback_entered_this_tick):
                        current_canvas = gan_lanczos_fallback(frame, args.rotate)
                        current_updated = updated
                        current_source_sequence = source_sequence
                        gan_display_fallback = True
                        gan_controller.note_lanczos_display(source_sequence)
                elif (current_canvas is None and frame is not None and
                      source_sequence is not None):
                    # Warmup has no last good GAN output yet.  This is a
                    # display-only Lanczos path, not a soft-fallback entry.
                    current_canvas = gan_lanczos_fallback(frame, args.rotate)
                    current_updated = updated
                    current_source_sequence = source_sequence
                    gan_display_fallback = True
                    gan_controller.note_lanczos_display(source_sequence)
                if now >= next_gan_present:
                    interval = 1.0 / (args.gan_display_fps or float(profile["fps"]))
                    skipped = max(0, int((now - next_gan_present) / interval))
                    next_gan_present += (skipped + 1) * interval
                    if current_source_sequence is not None:
                        presentation.present(
                            current_source_sequence,
                            ("GAN-FALLBACK-LANCZOS" if gan_display_fallback else
                             f"GAN-{worker.backend.upper()}"),
                            int(profile["generation"]),
                            now,
                        )
            elif is_rebuild and now >= next_rebuild_present:
                interval = 1.0 / args.rebuild_fps
                skipped = max(0, int((now - next_rebuild_present) / interval))
                next_rebuild_present += (skipped + 1) * interval
                if frame is not None and rebuild_receiver is not None and rebuild_composer is not None:
                    base = frame
                    if args.denoise == "on":
                        base = cv2.bilateralFilter(base, 5, 24.0, 24.0)
                    state, state_generation, references, _, sync_ms = \
                        rebuild_receiver.scene_synced(source_rtp_timestamp)
                    profile_generation = int(profile["generation"])
                    if state_generation != profile_generation:
                        state, references = None, {}
                    receiver_sync = rebuild_receiver.snapshot()
                    rebuilt, spatial = rebuild_composer.render(
                        base, state, profile_generation, references, now,
                        video_rtp_timestamp=source_rtp_timestamp,
                        pts_bias_ms=receiver_sync["pts_bias_ms"])
                    current_canvas = postprocess_frame(rebuilt, args.rotate, False)
                    current_updated = updated
                    current_source_sequence = source_sequence
                    presentation.present(source_sequence, spatial,
                                         profile_generation, now)
            elif not is_rebuild and not is_gan:
                if frame is not None and current_source_sequence != source_sequence:
                    current_canvas = postprocess_frame(
                        frame, args.rotate, args.denoise == "on")
                    current_updated = updated
                    current_source_sequence = source_sequence

            rebuild_values = (
                rebuild_receiver.snapshot()
                if is_rebuild and rebuild_receiver is not None
                else {}
            )
            composer_values = (
                rebuild_composer.snapshot()
                if is_rebuild and rebuild_composer is not None
                else {}
            )
            gan_values = (
                worker.snapshot() if is_gan and worker is not None
                else gan_waiting_worker_snapshot() if is_gan else {}
            )
            gan_controller_values = (
                gan_controller.snapshot(now, float(profile["fps"]))
                if is_gan and gan_controller is not None and profile is not None
                else {}
            )
            presentation_values = presentation.snapshot()
            if args.headless:
                # A headless report can fall between two rebuild presentation
                # ticks (12 fps).  Refresh the synchronized PTS here so the
                # log never hides a live value as ``none`` merely because the
                # reporting wall-clock tick did not render a frame.
                if (is_rebuild and rebuild_receiver is not None and
                        sync_ms is None and source_rtp_timestamp is not None):
                    _, _, _, _, sync_ms = rebuild_receiver.scene_synced(
                        source_rtp_timestamp)
                    # scene_synced may learn a new median bias; report the
                    # bias and sync value from the same observation.
                    rebuild_values = rebuild_receiver.snapshot()
                if now - last_report >= 1.0:
                    profile_text = "unknown" if profile is None else (
                        f"{profile['name']}:{profile['width']}x{profile['height']}@"
                        f"{profile['fps']}:g{profile['generation']}"
                    )
                    rebuild_text = ""
                    if is_rebuild:
                        media_wire = values["wire_kbps"] + rebuild_values["wire_kbps"]
                        rebuild_text = (
                            f" output={args.rebuild_width}x{args.rebuild_height}@"
                            f"{args.rebuild_fps:g} frame={presentation_values['provenance']}"
                            f" temporal_hold={presentation_values['held_percent']:.1f}%"
                            f" rb_wire_kbps={rebuild_values['wire_kbps']:.1f}"
                            f" video_rb_wire_kbps={media_wire:.1f} link_cap_kbps=100"
                            f" video_pps={values['pps']:.1f}"
                            f" video_pkt_avg_B={values['packet_avg_bytes']:.0f}"
                            f" rb_pps={rebuild_values['pps']:.1f}"
                            f" rb_pkt_avg_B={rebuild_values['packet_avg_bytes']:.0f}"
                            f" refs={composer_values['refs_used']}/"
                            f"{rebuild_values['active_references']}"
                            f" reg_drops={composer_values['registration_drops']}"
                            f" content_drops={composer_values['content_drops']}"
                            f" rebuild_area={composer_values['rebuild_percent']:.1f}%"
                            f" chroma={composer_values['chroma_mode']}"
                            f" pts_sync_ms={'none' if sync_ms is None else sync_ms}"
                            f" pts_bias_ms={rebuild_values['pts_bias_ms']}"
                            f" ref_pts_age_ms={composer_values['reference_content_age_ms']}"
                            f" sync_drops={rebuild_values['sync_drops']}"
                            f" fec_recovered={rebuild_values['parity_recovered']}"
                        )
                    elif is_gan:
                        input_size, native_size, output_size = gan_profile_sizes(profile)
                        drop_counts = gan_values["drop_reason_counts"]
                        gan_text = (
                            f" source_fps={profile['fps']}"
                            f" target_kbps={gan_target_kbps_text(profile)}"
                            f" enhanced_fps={gan_values['rolling_1s_fps']:.1f}"
                            f" display_fps={presentation_values['fps']:.1f}"
                            f" enhancer={gan_values['backend']}"
                            f" provider={gan_values['provider']}"
                            f" input={input_size[0]}x{input_size[1]}"
                            f" native={native_size[0]}x{native_size[1]}"
                            f" output={output_size[0]}x{output_size[1]}"
                            f" output_budget_ms={gan_values['max_latency_ms']:.1f}"
                            f" infer_all_p50_ms={gan_values['infer_all_p50_ms']:.1f}"
                            f" infer_all_p95_ms={gan_values['infer_all_p95_ms']:.1f}"
                            f" infer_all_p99_ms={gan_values['infer_all_p99_ms']:.1f}"
                            f" infer_all_max_ms={gan_values['infer_all_max_ms']:.1f}"
                            f" infer_last_ms={gan_values['last_finished_infer_ms']:.1f}"
                            f" good_total_p50_ms={gan_values['good_total_p50_ms']:.1f}"
                            f" good_total_p95_ms={gan_values['good_total_p95_ms']:.1f}"
                            f" good_total_p99_ms={gan_values['good_total_p99_ms']:.1f}"
                            f" good_total_max_ms={gan_values['good_total_max_ms']:.1f}"
                            f" good_last_ms={gan_values['last_good_total_ms']:.1f}"
                            f" queue_wait_ms={gan_values['last_queue_wait_ms']:.1f}"
                            f" running={int(gan_values['running'])}"
                            f" running_age_ms={gan_values['running_age_ms']:.1f}"
                            f" pending={int(gan_values['pending'])}"
                            f" dropped={gan_values['dropped']}"
                            f" dropped_pct={gan_values['drop_percent']:.1f}"
                            f" drop_repl={drop_counts['replaced_pending']}"
                            f" drop_output={drop_counts['output_replaced']}"
                            f" drop_pre={drop_counts['stale_before_infer']}"
                            f" drop_post={drop_counts['stale_after_infer']}"
                            f" drop_err={drop_counts['inference_error']}"
                            f" present_old={gan_controller_values.get('presentation_stale_drops', 0)}"
                            f" state={gan_controller_values.get('state', 'WAIT')}"
                            f" good_age_ms={gan_controller_values.get('good_age_ms')}"
                            f" run_age_ms={gan_values['running_age_ms']:.1f}"
                            f" recovery={gan_controller_values.get('recovery_streak', 0)}/"
                            f"{gan_controller_values.get('recovery_good_frames', 0)}"
                            f" hard_stalls={gan_controller_values.get('hard_stall_count', 0)}"
                            f" rtp_kbps={values['rtp_kbps']:.1f}"
                            f" wire_kbps={values['wire_kbps']:.1f}"
                            f" link_cap_kbps={gan_link_cap_kbps(profile)}"
                        )
                        rebuild_text = gan_text
                    print(
                        f"profile={profile_text} "
                        f"rx_fps={values['rx_fps']:.1f} decode_fps={values['decode_fps']:.1f} "
                        f"p_fps={values['p_fps']:.1f} i_fps={values['i_fps']:.1f} "
                        f"rtp_kbps={values['rtp_kbps']:.1f} wire_kbps={values['wire_kbps']:.1f} "
                        f"lost={values['lost']} reorder={values['reordered']} "
                        f"decode_errors={values['decode_errors']}" + rebuild_text,
                        flush=True,
                    )
                    last_report = now
                time.sleep(0.01)
                continue
            if current_canvas is None:
                blank_width = args.rebuild_width if is_rebuild else (640 if is_gan else args.width)
                blank_height = args.rebuild_height if is_rebuild else (360 if is_gan else args.height)
                canvas = postprocess_frame(
                    np.zeros((blank_height, blank_width, 3), dtype=np.uint8),
                    args.rotate, False)
            else:
                canvas = current_canvas.copy()
            effective_scale = args.scale or (1 if (is_rebuild or is_gan) else 3)
            display_size = (canvas.shape[1] * effective_scale,
                            canvas.shape[0] * effective_scale)
            if not args.headless and display_size != window_size:
                cv2.resizeWindow(window, *display_size)
                window_size = display_size
            if effective_scale != 1:
                canvas = cv2.resize(canvas, display_size, interpolation=cv2.INTER_LANCZOS4)
            if is_rebuild and rebuild_receiver is not None:
                state, _, _, _, sync_ms = rebuild_receiver.scene_synced(
                    source_rtp_timestamp)
                output_width = state.output_width if state is not None else args.rebuild_width
                output_height = state.output_height if state is not None else args.rebuild_height
                output_fps = state.output_fps if state is not None else args.rebuild_fps
                media_wire = values["wire_kbps"] + rebuild_values["wire_kbps"]
                ref_age = composer_values["reference_age"]
                ref_age_text = "none" if ref_age is None else f"{ref_age * 1000.0:.0f}ms"
                ref_pts_age = composer_values["reference_content_age_ms"]
                ref_pts_age_text = "none" if ref_pts_age is None else f"{ref_pts_age:+d}ms"
                sync_text = "none" if sync_ms is None else f"{sync_ms:+d}ms"
                if sync_ms is not None and abs(sync_ms) > args.rebuild_max_sync_ms:
                    sync_text += " DROP"
                lines = [
                    f"RX {values['rx_fps']:.1f} DEC {values['decode_fps']:.1f} "
                    f"OUT {presentation_values['fps']:.1f} fps",
                    f"H265 {values['rtp_kbps']:.1f} RB {rebuild_values['rtp_kbps']:.1f} kbps",
                    f"V+RB WIRE {media_wire:.1f} kbps",
                    "LINK CAP 100 kbps incl audio",
                    f"P/I {values['p_fps']:.1f}/{values['i_fps']:.1f} "
                    f"total {values['p_frames']}/{values['i_frames']}",
                    f"PKT {values['packets']} LOSS {values['lost']} "
                    f"REO {values['reordered']} ERR {values['decode_errors']}",
                    f"PPS V {values['pps']:.1f} RB {rebuild_values['pps']:.1f} "
                    f"S/D/F {rebuild_values['state_pps']:.0f}/"
                    f"{rebuild_values['data_pps']:.0f}/{rebuild_values['parity_pps']:.0f}",
                    f"LEN L/A/M V {values['packet_last_bytes']}/"
                    f"{values['packet_avg_bytes']:.0f}/{values['packet_max_bytes']} B",
                    f"LEN L/A/M RB {rebuild_values['packet_last_bytes']}/"
                    f"{rebuild_values['packet_avg_bytes']:.0f}/"
                    f"{rebuild_values['packet_max_bytes']} B",
                    f"REBUILD Gen {profile['generation']}",
                    f"SRC {profile['width']}x{profile['height']} @{profile['fps']} fps",
                    f"OUT {output_width}x{output_height} @{output_fps} fps",
                    f"FRAME {presentation_values['provenance']}",
                    f"TEMP HOLD {presentation_values['held_percent']:.1f}% "
                    f"({profile['fps']}->{output_fps})",
                    f"REF {composer_values['refs_used']}/"
                    f"{rebuild_values['active_references']} AGE {ref_age_text} "
                    f"PTSAGE {ref_pts_age_text} "
                    f"REGDROP {composer_values['registration_drops']} "
                    f"MATCHDROP {composer_values['content_drops']}",
                    f"ROI AREA {composer_values['rebuild_percent']:.1f}%",
                    f"PTS SYNC {sync_text} BIAS {rebuild_values['pts_bias_ms']:+d}ms "
                    f"DROP#{rebuild_values['sync_drops']}",
                    f"CHROMA {composer_values['chroma_mode']} "
                    f"BOX {args.rebuild_boxes.upper()}",
                    f"FEC {rebuild_values['parity_recovered']} "
                    f"INC {rebuild_values['incomplete']} BAD "
                    f"{rebuild_values['invalid'] + rebuild_values['inconsistent']}",
                    f"SR {composer_values['sr_model'][:24]} "
                    f"{composer_values['sr_done']}/{composer_values['sr_jobs']} "
                    f"P{1 if composer_values['sr_pending'] else 0} "
                    f"S{composer_values['sr_stale']}",
                ]
            elif is_gan:
                input_size, native_size, output_size = gan_profile_sizes(profile)
                output_backend = (
                    "FALLBACK / Lanczos4" if gan_display_fallback else
                    f"{gan_values['backend'].upper()} / {gan_values['provider']}"
                )
                drop_counts = gan_values["drop_reason_counts"]
                decode_age_ms = (
                    None if frame is None else max(0.0, (now - updated) * 1000.0)
                )
                good_age_ms = gan_controller_values.get("good_age_ms")
                good_age_text = "none" if good_age_ms is None else f"{good_age_ms:.0f}"
                output_age_ms = gan_controller_values.get("last_output_age_ms")
                output_age_text = "none" if output_age_ms is None else f"{output_age_ms:.0f}"
                state_text = gan_controller_values.get("state", "WAIT")
                if state_text == GanFallbackController.HARD_STALLED:
                    state_text = "HARD-STALL"
                lines = [
                    f"GAN H265-only RX {values['rx_fps']:.1f} DEC {values['decode_fps']:.1f} "
                    f"ENH {gan_values['rolling_1s_fps']:.1f}  DISP {presentation_values['fps']:.1f} fps",
                    f"INPUT {input_size[0]}x{input_size[1]} "
                    f"NATIVE {native_size[0]}x{native_size[1]}",
                    f"OUTPUT {output_size[0]}x{output_size[1]}  {output_backend}",
                    gan_bitrate_hud_line(profile, values),
                    f"INFER ALL L/P50/P95/P99/MAX "
                    f"{gan_values['last_finished_infer_ms']:.0f}/"
                    f"{gan_values['infer_all_p50_ms']:.0f}/"
                    f"{gan_values['infer_all_p95_ms']:.0f}/"
                    f"{gan_values['infer_all_p99_ms']:.0f}/"
                    f"{gan_values['infer_all_max_ms']:.0f} ms",
                    f"GOOD PC P50/P95/P99/MAX {gan_values['good_total_p50_ms']:.0f}/"
                    f"{gan_values['good_total_p95_ms']:.0f}/"
                    f"{gan_values['good_total_p99_ms']:.0f}/"
                    f"{gan_values['good_total_max_ms']:.0f} ms",
                    f"AGE DEC {('none' if decode_age_ms is None else f'{decode_age_ms:.0f}')} "
                    f"GAN {good_age_text} RUN {gan_values['running_age_ms']:.0f} "
                    f"OUT {output_age_text} ms",
                    f"BUDGET {gan_values['max_latency_ms']:.0f} SOFT "
                    f"{gan_controller_values.get('soft_threshold_ms', 0.0):.0f} ms",
                    f"QUEUE RUN{int(gan_values['running'])} PEND{int(gan_values['pending'])}",
                    f"DROP REPL {drop_counts['replaced_pending']} OUT {drop_counts['output_replaced']} "
                    f"PRE {drop_counts['stale_before_infer']} "
                    f"PRED {drop_counts['stale_before_predicted']} "
                    f"POST {drop_counts['stale_after_infer']} ERR {drop_counts['inference_error']}",
                    f"PRESENT OLD {gan_controller_values.get('presentation_stale_drops', 0)}",
                    f"P/I {values['p_fps']:.1f}/{values['i_fps']:.1f}  "
                    f"PACKETS {values['packets']}  LOSS {values['lost']}  ERR {values['decode_errors']}",
                    f"STATE {state_text}  REC {gan_controller_values.get('recovery_streak', 0)}/"
                    f"{gan_controller_values.get('recovery_good_frames', 0)}  "
                    f"HARDSTALL {gan_controller_values.get('hard_stall_count', 0)}",
                ]
                if profile is not None:
                    lines.append(
                        f"Profile {profile['name']}   {profile['width']}x{profile['height']} @ "
                        f"{profile['fps']} fps   Gen {profile['generation']}"
                    )
            else:
                lines = [
                    f"RX {values['rx_fps']:.1f} fps   Decode {values['decode_fps']:.1f} fps   "
                    f"RTP {values['rtp_kbps']:.1f} kbps   Wire {values['wire_kbps']:.1f} kbps",
                    f"P {values['p_fps']:.1f} fps   I {values['i_fps']:.1f} fps   "
                    f"P/I total {values['p_frames']}/{values['i_frames']}",
                    f"Packets {values['packets']}   Lost {values['lost']}   "
                    f"Reorder {values['reordered']}   Decode errors {values['decode_errors']}",
                    f"UDP {values['pps']:.1f} pps   packet L/A/M "
                    f"{values['packet_last_bytes']}/{values['packet_avg_bytes']:.0f}/"
                    f"{values['packet_max_bytes']} B",
                ]
                if profile is not None:
                    lines.append(
                        f"Profile {profile['name']}   {profile['width']}x{profile['height']} @ "
                        f"{profile['fps']} fps   Gen {profile['generation']}"
                    )
            if frame is None:
                lines.append("WAITING FOR COMPLETE IDR - start/restart the board sender now")
            elif is_gan:
                idr = "none" if values["idr_age"] is None else f"{values['idr_age']:.1f}s ago"
                lines.append(f"Last IDR {idr}")
            else:
                age_ms = max(0.0, (time.monotonic() - current_updated) * 1000.0)
                idr = "none" if values["idr_age"] is None else f"{values['idr_age']:.1f}s ago"
                lines.append(f"Source age {age_ms:.0f} ms   Last IDR {idr}")
            font_scale = 0.40 if (is_rebuild or is_gan) else 0.58
            line_height = 21 if (is_rebuild or is_gan) else 26
            for index, line in enumerate(lines):
                cv2.putText(canvas, line, (8 if is_rebuild else 10,
                            20 + index * line_height if is_rebuild else 24 + index * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (80, 255, 80), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1 if (is_rebuild or is_gan) else 20) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        stopping.set()
        receiver.close()
        disable_rebuild_components()
        disable_gan_worker()
        with decoder_lock:
            stop_decoder_locked()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the explicit 3 fps x 3 enhancer GAN A/B command matrix.

The matrix is intentionally declarative.  It does not pretend to have run a
60-second board capture; use ``--run`` only after the board/capture setup is
known to be ready, or consume the generated commands from an operator shell.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/gan_ab"))
    parser.add_argument("--board-host", default="root@192.168.0.101")
    parser.add_argument("--board-dir", default="/opt/atk/rknn_yolov8_seg_cam")
    parser.add_argument("--pc-host", default="192.168.0.100")
    parser.add_argument("--video-port", type=int, default=5004)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--bitrate-kbps", type=int, default=75)
    parser.add_argument("--esrnet-model", default="model/RealESRNet_x2_dynamic.onnx")
    parser.add_argument("--esrgan-model", default="model/RealESRGAN_x2_dynamic.onnx")
    parser.add_argument("--run", action="store_true",
                        help="reserved for an operator-controlled runner; matrix generation remains side-effect free")
    args = parser.parse_args()
    if args.duration < 60 or args.bitrate_kbps <= 0:
        parser.error("goal acceptance requires duration >= 60 seconds and a positive bitrate")
    if args.run:
        parser.error("--run is intentionally not enabled yet: execute each generated command only after confirming capture paths")

    output_root = args.output_root
    entries = []
    for fps in (8, 10, 12):
        for enhancer in ("none", "esrnet", "esrgan"):
            name = f"fps{fps}_{enhancer}"
            directory = output_root / name
            directory.mkdir(parents=True, exist_ok=True)
            sdp_remote = f"/tmp/gan_{fps}_{enhancer}.sdp"
            remote_sender = (
                f"cd {args.board_dir} && env LD_LIBRARY_PATH=\"$PWD/lib\" "
                f"./rknn_yolov8_seg_cam --mode=gan --gan-fps={fps} "
                f"--gan-video-bitrate-kbps={args.bitrate_kbps} --audio=off --preview=off "
                f"--udp-host={args.pc_host} --udp-port={args.video_port} "
                f"--rtp-sdp-path={sdp_remote} --profile-control="
            )
            sender = f"ssh {shlex.quote(args.board_host)} {shlex.quote(remote_sender)}"
            receiver_model = ""
            if enhancer == "esrnet":
                receiver_model = f" --gan-esrnet-model={shlex.quote(args.esrnet_model)}"
            elif enhancer == "esrgan":
                receiver_model = f" --gan-esrgan-model={shlex.quote(args.esrgan_model)}"
            receiver = (
                f"python tools/live_h265_hud.py pc_sdp/{name}.sdp --headless "
                f"--duration={args.duration} --gan-enhancer={enhancer} "
                f"{'--gan-require-cuda' if enhancer != 'none' else ''}{receiver_model} "
                f"--gan-debug-log={shlex.quote(str(directory / 'enhancer.jsonl'))}"
            )
            entries.append({
                "name": name,
                "fps": fps,
                "enhancer": enhancer,
                "duration_s": args.duration,
                "video_bitrate_kbps": args.bitrate_kbps,
                "sender_command": sender,
                "sdp_copy_command": f"scp {shlex.quote(args.board_host)}:{sdp_remote} pc_sdp/{name}.sdp",
                "receiver_command": receiver,
                "output_directory": str(directory),
                "scenes": ["empty", "person1", "person2", "person3_or_more"],
                "required_outputs": ["clean.mp4", "hud.mp4", "enhancer.jsonl", "metrics.json"],
            })

    output_root.mkdir(parents=True, exist_ok=True)
    matrix_path = output_root / "command_matrix.json"
    matrix_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# GAN A/B command matrix",
        "",
        "Each cell requires at least 60 s for each of the four scene conditions.",
        "The sender bitrate is held at the same ceiling for all fps values.",
        "",
        "| group | fps | enhancer | sender | receiver |",
        "|---|---:|---|---|---|",
    ]
    for entry in entries:
        markdown.append(
            f"| `{entry['name']}` | {entry['fps']} | `{entry['enhancer']}` | "
            f"`{entry['sender_command']}` | `{entry['receiver_command']}` |"
        )
    (output_root / "command_matrix.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"wrote {matrix_path} and {output_root / 'command_matrix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

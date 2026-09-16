#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
exec env LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./board_h265_receiver --udp-port=5004 --display=wayland --fullscreen \
  --rotate=ccw --receive-buffer-bytes=1048576 --reorder-window=32 "$@"


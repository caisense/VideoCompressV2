#!/bin/sh
# Profiles are labeled with their actual completed validation stage.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/launch_video_quality_sender.py" "$@"

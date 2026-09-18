#!/bin/sh
set -eu
CONTROL=${1:-/tmp/roi-rate-profile}
for profile in rate60 rate80 rate100 rate120 rate150 rate180 rate200 rate300; do
  printf '%s\n' "$profile" > "$CONTROL"
  echo "profile=$profile started; observe for 60 seconds"
  sleep 60
done

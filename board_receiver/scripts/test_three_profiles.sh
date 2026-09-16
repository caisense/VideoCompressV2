#!/bin/sh
set -eu
CONTROL=${1:-/tmp/roi-rate-profile}
for profile in low medium high; do
  printf '%s\n' "$profile" > "$CONTROL"
  echo "profile=$profile started; observe for 60 seconds"
  sleep 60
done


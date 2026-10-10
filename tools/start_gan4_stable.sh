#!/bin/sh
# Verified gan4 temporal-quality configuration, CAP120 / 8fps.
# Pass the PC receiver IP, followed by optional sender arguments.
set -eu

if [ "$#" -lt 1 ]; then
    echo "Usage: sh start_gan4_stable.sh RECEIVER_IP [sender options...]" >&2
    exit 2
fi
receiver_ip=$1
shift
app_dir=${GAN4_APP_DIR:-/opt/atk/rknn_yolov8_seg_cam}
cd "$app_dir"
export LD_LIBRARY_PATH="$app_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec ./rknn_yolov8_seg_cam \
    --gan-native-720p=on --rate-profile=gan4 \
    --gan-link-cap-kbps=120 --gan-fps=8 --gan-video-bitrate-kbps=86 \
    --gan-gop-seconds=2 --gan-inference-fps=0 \
    --gan-qp-min-p=28 --gan-qp-max-p=32 --gan-qp-min-i=28 \
    --gan-debreath=off --gan-intra-refresh=off --gan-super-frame=current \
    --model=model/yolov8_seg.rknn --camera-device=/dev/video-camera0 \
    --udp-host="$receiver_ip" --udp-port=5004 \
    --audio=off --preview=off --profile-control= "$@"

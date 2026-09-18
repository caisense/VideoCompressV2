# RK3588 board H.265 receiver

`board_h265_receiver` is an independent, video-only C++ receiver for the
ATK-DLRK3588 Buildroot image. It accepts the same UDP/RTP/H.265 stream as
`tools/live_h265_hud.py`, decodes it with Rockchip MPP, and displays unannotated
video through the board's Weston/Wayland OpenCV HighGUI backend.

## Supported wire format

- UDP, port 5004 by default
- RTP version 2, payload type 96, 90 kHz timestamps
- RFC 7798 Single NAL, AP type 48, and FU type 49
- RTP extension profile `0x524f` (`RO`), metadata version 1
- eight ordinary rate profiles:

| ID | Name | Encoded video | H.265 target / shared wire cap |
| ---: | --- | --- | --- |
| 0 | `rate60` | 320x180 @ 10, grayscale | 42 / 60 kbps |
| 5 | `rate80` | 320x180 @ 10, color | 56 / 80 kbps |
| 6 | `rate100` | 384x216 @ 10, color | 72 / 100 kbps |
| 7 | `rate120` | 480x270 @ 10, color | 88 / 120 kbps |
| 1 | `rate150` | 480x270 @ 15, color | 110 / 150 kbps |
| 8 | `rate180` | 512x288 @ 15, color | 135 / 180 kbps |
| 9 | `rate200` | 512x288 @ 18, color | 150 / 200 kbps |
| 2 | `rate300` | 640x360 @ 20, color | 240 / 300 kbps |

The rate number is the shared physical A/V cap, not the raw H.265 target.
`rate150` is the renamed former medium profile; there is no duplicate 150 kbps
profile. Wire IDs 0/1/2 are retained for old-sender compatibility.

Profiles 3 (`rebuild`), 4 (`gan`), and unknown values are parsed only so they
can be explicitly rejected. Audio, Codec2, image packets, GAN, and inference
metadata are intentionally outside this program.

The decoder identity is `(SSRC, generation, profile, width, height, fps)`.
Startup and identity changes wait for a complete VPS/SPS/PPS/IRAP access unit.
For one SSRC, an older generation cannot roll the active decoder backward.
After the configured idle timeout the session is reset, allowing a restarted
sender whose generation begins again at zero.

## Build

From the supplied Ubuntu SDK VM:

```sh
cd /mnt/hgfs/videoCompressV2/board_receiver
chmod +x build-linux.sh
./build-linux.sh
```

The default toolchain is `/opt/atk-dlrk3588-toolchain`; override it with
`RK3588_TOOLCHAIN_DIR`. The script links only against the Buildroot sysroot and
prints `file` and `readelf` evidence for the installed binary. It does not
bundle the SDK MPP library: board 101 has glibc 2.37 and should use its own
`librockchip_mpp.so.1`.

Protocol-only host tests do not require MPP or OpenCV:

```sh
cmake -S . -B build-tests \
  -DBOARD_RECEIVER_BUILD_APP=OFF \
  -DBOARD_RECEIVER_BUILD_TESTS=ON
cmake --build build-tests --parallel
ctest --test-dir build-tests --output-on-failure
```

## Deploy and run

From Windows, after cross-compilation:

```powershell
.\board_receiver\scripts\deploy_receiver.ps1 -Board 192.168.0.101
```

Passwords are requested interactively and are never stored by the scripts.
On board 101:

```sh
cd /opt/atk/board_receiver
./scripts/start_receiver.sh
```

Equivalent explicit command:

```sh
env XDG_RUNTIME_DIR=/run WAYLAND_DISPLAY=wayland-0 \
  ./board_h265_receiver \
  --udp-port=5004 --display=wayland --fullscreen --rotate=ccw --hud \
  --receive-buffer-bytes=1048576 --reorder-window=32
```

Useful diagnostics are `--headless`, `--max-frames=N`,
`--stats-interval-ms=N`, `--idle-timeout-ms=N`, and `--rotate=none`.
`scripts/start_receiver.sh` enables `--hud` by default. Pass `--no-hud` after
the script name to display clean video. The board HUD contains five green-text
lines with no black mask: this board's locally measured physical transmit bitrate
(`TX`) and locally estimated physical receive bitrate (`RX`) in kbps; cumulative P/I/total
frame counts; packet/loss/reorder/decode-error counters; active
profile/resolution/FPS/generation; and source/IDR age. TX and RX use the same
Ethernet-wire accounting. The local sender publishes TX through the Unix datagram
socket `/tmp/board_tx_wire_rate.sock`; it is not part of the RTP protocol. If no
local sender telemetry arrives for two seconds, the HUD displays `TX --`.
Terminate with SIGINT or SIGTERM.

On board 102, start the existing sender with `--preview=off`, `--audio=off`,
and `--udp-host=192.168.0.101`. Use `--send-queue-frames=16` and
`--send-max-latency-ms=2000`. These are also the normal rate-profile defaults.
Runtime switching supports all eight physical-link caps:

```sh
for profile in rate60 rate80 rate100 rate120 rate150 rate180 rate200 rate300; do
  echo "$profile" > /tmp/roi-rate-profile
done
```

## Two-board full-duplex startup

Both boards must contain the receiver under `/opt/atk/board_receiver`, the
existing sender under `/opt/atk/rknn_yolov8_seg_cam`, and a working
`/dev/video-camera0`. Each board listens on its own UDP port 5004 while its
sender targets port 5004 on the other board, so the two directions do not
conflict.

Use two SSH terminals for each board. Start both receivers first, then both
senders. Keep sender preview disabled so the received video owns the external
display.

### Board 101 (`192.168.0.101`)

Terminal 1 — receive video from board 102:

```sh
cd /opt/atk/board_receiver
./scripts/start_receiver.sh
```

Terminal 2 — send the local camera to board 102:

```sh
cd /opt/atk/rknn_yolov8_seg_cam
rm -f /tmp/roi-rate-profile
mkfifo /tmp/roi-rate-profile

env LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./rknn_yolov8_seg_cam \
  --rate-profile=rate150 \
  --mode=segmentation \
  --model=model/yolov8_seg.rknn \
  --camera-device=/dev/video-camera0 \
  --udp-host=192.168.0.102 \
  --udp-port=5004 \
  --audio=off \
  --preview=off \
  --profile-control=/tmp/roi-rate-profile \
  --send-queue-frames=16 \
  --send-max-latency-ms=2000
```

### Board 102 (`192.168.0.102`)

Terminal 1 — receive video from board 101:

```sh
cd /opt/atk/board_receiver
./scripts/start_receiver.sh
```

Terminal 2 — send the local camera to board 101:

```sh
cd /opt/atk/rknn_yolov8_seg_cam
rm -f /tmp/roi-rate-profile
mkfifo /tmp/roi-rate-profile

env LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./rknn_yolov8_seg_cam \
  --rate-profile=rate150 \
  --mode=segmentation \
  --model=model/yolov8_seg.rknn \
  --camera-device=/dev/video-camera0 \
  --udp-host=192.168.0.101 \
  --udp-port=5004 \
  --audio=off \
  --preview=off \
  --profile-control=/tmp/roi-rate-profile \
  --send-queue-frames=16 \
  --send-max-latency-ms=2000
```

To change the outgoing profile on either board, open another SSH terminal to
that board and write one of these values to its local control FIFO:

```sh
echo rate60  > /tmp/roi-rate-profile
echo rate80  > /tmp/roi-rate-profile
echo rate100 > /tmp/roi-rate-profile
echo rate120 > /tmp/roi-rate-profile
echo rate150 > /tmp/roi-rate-profile
echo rate180 > /tmp/roi-rate-profile
echo rate200 > /tmp/roi-rate-profile
echo rate300 > /tmp/roi-rate-profile
```

The two directions switch independently. Stop each foreground sender and
receiver with `Ctrl+C`. Do not recreate the FIFO while its sender is running.

## Validation

Save commands and password-free logs under `test-results/`. Completion requires
all protocol tests, AArch64 ELF/ABI checks, eight 60-second profile runs,
30 repeated switches, fault recovery, a 30-minute rate300 soak, and direct
external-screen evidence. Decoder logs alone are not screen evidence.

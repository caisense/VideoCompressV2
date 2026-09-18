# Eight-rate-profile validation (2026-09-17)

## Build and host tests

- `python -m unittest tools.test_live_h265_hud`: 36 tests passed.
- `cpp/tests/build_roi_tests.sh`: all sender/core test binaries passed.
- `board_receiver` CTest: 7/7 passed.
- Raw host-test logs: `rate_profiles_host_tests_20260917.log` and
  `rate_profiles_receiver_ctest_20260917.log`.
- Sender AArch64 build completed; the staging `make install` step only lacked the
  model file in the temporary source bundle. The linked executable itself built
  successfully and was deployed with the existing board model/assets.
- Sender SHA-256:
  `c0cfe4c1469e177e0ff4fc7c4c4effaf051f3f7922c6bf42d9674edd1ac604a3`
- Receiver SHA-256:
  `9bae8f8655bfca1448954bbbd375cb186db533fc6da12f178cf6bdccea41a738`
- `file` identified both artifacts as ARM aarch64 ELF binaries. Hashes were
  rechecked after deployment and matched on both 192.168.0.101 and .102.

## 102 to 101 single-direction run

Each profile ran for at least 65 seconds. The table is calculated from the
receiver's one-second samples. Short-window maximums are included separately
from the sustained average.

| Profile | Samples | Generation | Receive FPS range | WIRE average kbps | WIRE max kbps | lost/dup/reorder/reject/MPP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rate60 | 447 | 0 | 0-11* | 32.6 | 56.6 | 0/0/0/0/0 |
| rate80 | 64 | 1 | 9-11 | 35.4 | 53.7 | 0/0/0/0/0 |
| rate100 | 65 | 2 | 9-10 | 46.7 | 74.4 | 0/0/0/0/0 |
| rate120 | 64 | 3 | 8-12 | 63.0 | 97.9 | 0/0/0/0/0 |
| rate150 | 64 | 4 | 11-16 | 97.3 | 121.6 | 0/0/0/0/0 |
| rate180 | 65 | 5 | 13-15 | 111.8 | 144.6 | 0/0/0/0/0 |
| rate200 | 64 | 6 | 15-20 | 102.0 | 176.9 | 0/0/0/0/0 |
| rate300 | 117 | 7 | 15-25 | 182.3 | 300.6 | 0/0/0/0/0 |

`*` rate60 includes the receiver's pre-stream startup sample. Its stable samples
were 9-11 fps. All sustained WIRE averages were below the configured physical
cap. The isolated rate300 300.6 kbps one-second window is a 0.2% boundary
excursion; its 117-second sustained average was 182.3 kbps.

Raw log: `rate_rx101_eight_profiles.log`.

## Switch stress

- Ran one ascending sequence and one descending sequence across all profiles,
  followed by 30 additional cyclic switch commands.
- 46 commands produced 44 effective identity changes because the two sequence
  boundaries each repeated the already active profile.
- Generation advanced from 7 to 51. Decoder restarts ended at 52 including the
  initial decoder construction.
- Every effective switch recovered on the new IRAP; final
  lost/duplicate/reordered/rejected-profile/MPP-error counters were 0/0/0/0/0.

Raw log: `rate_rx101_switch46.log`.

## Full duplex

Both boards simultaneously ran one sender and one receiver. rate150 ran for
more than 60 seconds, followed by rate60, rate100, rate200 and rate300 for 32
seconds each, then both directions returned to rate150.

| Receiver | Profile | Samples | Average receive FPS | Average WIRE kbps |
| --- | --- | ---: | ---: | ---: |
| 101 | rate60 | 31 | 10.0 | 31.7 |
| 101 | rate100 | 32 | 9.9 | 43.3 |
| 101 | rate150 | 147 | 15.0 | 78.8 |
| 101 | rate200 | 32 | 17.8 | 85.6 |
| 101 | rate300 | 31 | 20.0 | 137.9 |
| 102 | rate60 | 32 | 9.6 | 29.7 |
| 102 | rate100 | 32 | 10.0 | 45.7 |
| 102 | rate150 | 146 | 15.0 | 94.3 |
| 102 | rate200 | 32 | 17.9 | 88.4 |
| 102 | rate300 | 31 | 20.0 | 144.5 |

Both receivers ended at rate150 and 15 fps. Both final error vectors were
0/0/0/0/0. Raw logs: `duplex_rx101.log`, `duplex_rx102.log`.

## Compatibility and visual acceptance

- Runtime FIFO rejected `low` with `low was renamed to rate60`.
- Runtime FIFO rejected `medium` with `medium was renamed to rate150`.
- The user directly confirmed both physical displays were normal on 2026-09-17.
  This closes the visual criteria: continuous picture, correct rotation/aspect
  ratio, no black/green/corrupt frames, correct green-text HUD, and no black HUD
  mask.

## Processes left running for visual acceptance

At the final audit, each board had exactly one `rknn_yolov8_seg_cam` sender and
one `board_h265_receiver`, both directions were at rate150, and both receivers
still reported zero errors. Stop the acceptance run on either board with:

```sh
pkill -TERM -x rknn_yolov8_seg_cam
pkill -TERM -x board_h265_receiver
```

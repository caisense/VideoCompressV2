#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
TOOLCHAIN_DIR=${RK3588_TOOLCHAIN_DIR:-/opt/atk-dlrk3588-toolchain}
PREFIX="$TOOLCHAIN_DIR/bin/aarch64-buildroot-linux-gnu"
SYSROOT="$TOOLCHAIN_DIR/aarch64-buildroot-linux-gnu/sysroot"
BUILD_DIR="$ROOT_DIR/build-rk3588"
INSTALL_DIR="$ROOT_DIR/install-rk3588"

test -x "$PREFIX-g++"
test -d "$SYSROOT"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="$PREFIX-gcc" \
  -DCMAKE_CXX_COMPILER="$PREFIX-g++" \
  -DCMAKE_SYSROOT="$SYSROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOARD_RECEIVER_BUILD_APP=ON \
  -DBOARD_RECEIVER_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
cmake --build "$BUILD_DIR" --parallel
cmake --install "$BUILD_DIR"
"$PREFIX-readelf" -h "$INSTALL_DIR/board_h265_receiver"
"$PREFIX-readelf" -d "$INSTALL_DIR/board_h265_receiver"
file "$INSTALL_DIR/board_h265_receiver"


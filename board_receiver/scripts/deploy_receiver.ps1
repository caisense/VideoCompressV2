param([string]$Board = "192.168.0.101")
$ErrorActionPreference = "Stop"
$source = Join-Path $PSScriptRoot "..\install-rk3588\board_h265_receiver"
if (-not (Test-Path -LiteralPath $source)) { throw "Build output not found: $source" }
ssh "root@$Board" "mkdir -p /opt/atk/board_receiver/scripts"
scp $source "root@${Board}:/opt/atk/board_receiver/board_h265_receiver"
scp (Join-Path $PSScriptRoot "start_receiver.sh") "root@${Board}:/opt/atk/board_receiver/scripts/start_receiver.sh"
ssh "root@$Board" "chmod +x /opt/atk/board_receiver/board_h265_receiver /opt/atk/board_receiver/scripts/start_receiver.sh"


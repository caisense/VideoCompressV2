param(
    [int[]]$MaxIProps = @(30, 24, 20, 16),
    [int]$QpMinI = 36,
    [string]$TagPrefix = 'maxi',
    [string]$ExtraArgs = ''
)

$ErrorActionPreference = 'Stop'
$runDir = 'D:\work\videoCompressV2\runs\goal0920a'
$pythonExe = 'D:\Users\57174\anaconda3\envs\videocompress\python.exe'
$askpass = 'D:\work\videoCompressV2\tools\codex_askpass_board.cmd'
New-Item -ItemType Directory -Force $runDir | Out-Null
$env:SSH_ASKPASS = $askpass
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'codex'
$env:PATH = 'D:\Users\57174\anaconda3\envs\videocompress\Library\bin;' +
    'D:\Users\57174\anaconda3\envs\videocompress\Scripts;' + $env:PATH

foreach ($value in $MaxIProps) {
    $tag = "${TagPrefix}${value}"
    $ganLog = Join-Path $runDir "${tag}_gan.jsonl"
    $receiverOut = Join-Path $runDir "${tag}_receiver.log"
    $receiverErr = Join-Path $runDir "${tag}_receiver.err"
    $senderOut = Join-Path $runDir "${tag}_sender.log"
    Remove-Item -LiteralPath $ganLog, $receiverOut, $receiverErr, $senderOut -Force -ErrorAction SilentlyContinue
    $receiverArgs = @(
        'tools\live_h265_hud.py', '--udp-port=5004', '--gan-enhancer=none',
        '--rotate=none', '--headless', '--duration=24', "--gan-debug-log=$ganLog"
    )
    $receiver = Start-Process -FilePath $pythonExe -ArgumentList $receiverArgs `
        -WorkingDirectory 'D:\work\videoCompressV2' -RedirectStandardOutput $receiverOut `
        -RedirectStandardError $receiverErr -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    $remoteLog = "/tmp/goal0920a_${tag}_r2.jsonl"
    $remote = "cd /opt/atk/rknn_yolov8_seg_cam && env LD_LIBRARY_PATH=`$PWD/lib " +
        "./rknn_yolov8_seg_cam.goal0920a --mode=gan --gan-link-cap-kbps=60 --gan-fps=8 " +
        "--gan-inference-fps=0 --model=model/yolov8_seg.rknn --camera-device=/dev/video-camera0 " +
        "--udp-host=192.168.0.128 --udp-port=5004 --profile-control= --audio=off --preview=off " +
        "--max-frames=160 --gan-max-i-prop=$value --gan-min-i-prop=10 --gan-init-ip-ratio=160 " +
        "--gan-qp-min-i=$QpMinI " +
        "--gan-debreath=off --gan-intra-refresh=off $ExtraArgs --encoder-debug-log=$remoteLog"
    & ssh root@192.168.0.101 $remote *> $senderOut
    Wait-Process -Id $receiver.Id -Timeout 35 -ErrorAction SilentlyContinue
    & scp "root@192.168.0.101:$remoteLog" (Join-Path $runDir "${tag}_encoder.jsonl") | Out-Null
    Write-Output "DONE max_i_prop=$value"
}

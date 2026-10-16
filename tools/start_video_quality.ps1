param(
    [ValidateSet(0,60,120,180)][int]$Cap = 0,
    [ValidateSet('auto','none','x2','x4')][string]$Enhancer = 'auto',
    [ValidateSet('cuda','tensorrt-fp16')][string]$Provider = 'tensorrt-fp16',
    [ValidateRange(1,65535)][int]$Port = 5004,
    [string]$Python = '',
    [ValidateRange(0,86400)][int]$Duration = 0,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
if ($Cap -eq 0) {
    $choice = Read-Host 'CAP (60 / 120 / 180)'
    if ($choice -notin @('60','120','180')) { throw 'CAP must be 60, 120 or 180' }
    $Cap = [int]$choice
}
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$config = Get-Content -LiteralPath "$PSScriptRoot/video_quality_profiles.json" -Raw | ConvertFrom-Json
$profile = $config.profiles.PSObject.Properties["$Cap"].Value
if ($Enhancer -eq 'auto') { $Enhancer = $profile.enhancer }
if (-not $Python) { $Python = $env:VIDEO_QUALITY_PYTHON }
if (-not $Python) {
    $runtime = Join-Path $repo 'runs/video_quality_optimization/receiver_runtime.json'
    if (Test-Path -LiteralPath $runtime) {
        $Python = (Get-Content -LiteralPath $runtime -Raw | ConvertFrom-Json).python
    }
}
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
$validation = if ($Enhancer -eq $profile.enhancer -and $Provider -eq $profile.provider) {
    $config.validation_state
} else { 'CUSTOM_UNVALIDATED' }
if ($Enhancer -ne 'none') {
    $modelKey = "gan_$Enhancer"
    $model = $config.models.PSObject.Properties[$modelKey].Value
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $modelStream = [System.IO.File]::OpenRead((Join-Path $repo $model.path))
    try {
        $actual = ([BitConverter]::ToString($sha.ComputeHash($modelStream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $modelStream.Dispose()
        $sha.Dispose()
    }
    if ($actual -ne $model.sha256) { throw "Validated model hash mismatch: $($model.path)" }
}
Write-Host "CAP$Cap $Enhancer $Provider | $validation | Python: $Python"
Write-Host "Validation GPU: $($config.validation_gpu); other GPUs require runtime validation."
if ($CheckOnly) {
    [pscustomobject]@{
        Cap = $Cap; Enhancer = $Enhancer; Provider = $Provider
        Python = $Python; Validation = $validation
        ValidationGpu = $config.validation_gpu
        HardwareStartup = 'NOT RUN'; Port = $Port
    } | ConvertTo-Json
    exit 0
}
$importCheck = "import sys, cv2, numpy, onnxruntime as ort; print('Receiver Python:',sys.executable); print('Available providers:',ort.get_available_providers())"
& $Python -c $importCheck
if ($LASTEXITCODE -ne 0) { throw 'Receiver environment import check failed' }
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = "runs/video_quality_optimization/live_${Cap}_$tag"
New-Item -ItemType Directory -Force $logDir | Out-Null
$scale = if ($Enhancer -eq 'x2') { 2 } else { 4 }
$backend = if ($Enhancer -eq 'none') { 'none' } else { 'esrgan' }
$activeProvider = if ($Enhancer -eq 'none') { 'auto' } else { $Provider }
$receiverArgs = @('tools/live_h265_hud.py', "--udp-port=$Port", '--rotate=ccw90',
    "--gan-enhancer=$backend", "--ganq-scale=$scale", "--gan-execution-provider=$activeProvider",
    '--gan-esrgan-model=model/RealESRGAN_x2_dynamic.onnx',
    '--gan-esrgan-x4-model=model/RealESRGAN_x4plus_dynamic.onnx',
    "--gan-debug-log=$logDir/enhancer.jsonl", "--quality-packet-log=$logDir/packets.jsonl",
    "--quality-video-log=$logDir/received.h265")
if ($Duration -gt 0) { $receiverArgs += "--duration=$Duration" }
@{ Cap = $Cap; Validation = $validation; Python = $Python; Args = $receiverArgs } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$logDir/launcher.json" -Encoding UTF8
& $Python @receiverArgs
exit $LASTEXITCODE

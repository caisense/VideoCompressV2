"""Launch a measured video-only profile with board-local verbose logging."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cap', choices=('60', '120', '180'))
    parser.add_argument('receiver', nargs='?')
    parser.add_argument('--app-dir', default=os.environ.get(
        'VIDEO_QUALITY_APP_DIR', '/opt/atk/video_quality'))
    parser.add_argument('--port', type=int, default=5004)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Optional bounded functional run; zero runs continuously')
    args = parser.parse_args()
    config = json.loads(Path(os.environ.get('VIDEO_QUALITY_PROFILE_CONFIG',
        Path(__file__).with_name('video_quality_profiles.json'))).read_text(encoding='utf-8'))
    profile = config['profiles'][args.cap]
    app_dir = Path(args.app_dir).resolve()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    if args.max_frames < 0:
        parser.error('max-frames must be nonnegative')
    command = [str(app_dir / 'rknn_yolov8_seg_cam'),
        '--rate-profile=ganq' + args.cap,
        '--model=' + config['models']['yolo']['path'],
        '--camera-device=/dev/video-camera0',
        '--udp-host=' + (args.receiver or config['default_receiver_ip']),
        '--udp-port=' + str(args.port), '--audio=off',
        '--audio-reserve-bitrate=0', '--preview=off', '--profile-control=',
        '--adaptive-roi=off', '--gan-inference-fps=0', '--gan-debreath=off',
        '--gan-intra-refresh=off', '--gan-super-frame=current']
    fields = {'width': 'encoder-width', 'height': 'encoder-height',
        'fps': 'gan-fps', 'target': 'gan-video-bitrate-kbps',
        'gop_seconds': 'gan-gop-seconds', 'qp_min': 'gan-qp-min-p',
        'qp_max': 'gan-qp-max-p', 'qp_min_i': 'gan-qp-min-i',
        'qp_max_i': 'qp-max-i'}
    command += ['--{}={}'.format(flag, profile[key]) for key, flag in fields.items()]
    if args.max_frames:
        command.append('--max-frames=' + str(args.max_frames))
    if args.dry_run:
        print(json.dumps({'validation': config['validation_state'],
                          'cwd': str(app_dir), 'argv': command}, indent=2))
        return 0
    for path, expected in (
        (Path(command[0]), config['sender_binary_sha256']),
        (app_dir / config['models']['yolo']['path'], config['models']['yolo']['sha256']),
        (app_dir / config['models']['labels']['path'], config['models']['labels']['sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Validated file hash mismatch: ' + str(path))
    log_dir = app_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / ('CAP' + args.cap + '_' + time.strftime('%Y%m%d_%H%M%S')
                          + '_' + str(time.time_ns()) + '.log')
    environment = os.environ.copy()
    environment['LD_LIBRARY_PATH'] = str(app_dir / 'lib') + (
        ':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
    print('CAP{} {} | sender log: {}'.format(args.cap, config['validation_state'], log_path), flush=True)
    with log_path.open('xb') as log:
        result = subprocess.run(command, cwd=app_dir, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 6000))
            print(log.read().decode(errors='replace'), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

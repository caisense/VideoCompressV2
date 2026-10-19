#!/usr/bin/env python3
"""Generate or execute bounded board/PC trials; defaults only generate commands."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('static_person', 'moving_person', 'multi_person', 'text', 'pan_motion')
OPTIONAL_ENCODER_OPTIONS = {
    'qp_init': 'qp-init', 'qp_ip': 'gan-qp-ip',
    'max_i_prop': 'gan-max-i-prop', 'min_i_prop': 'gan-min-i-prop',
    'init_ip_ratio': 'gan-init-ip-ratio',
    'fqp_min_p': 'gan-fqp-min-p', 'fqp_max_p': 'gan-fqp-max-p',
    'super_priority': 'gan-super-priority',
    'idr_roi_scale_percent': 'gan-idr-roi-scale-percent',
}
OPTIONAL_SELECTED_FIELDS = tuple(OPTIONAL_ENCODER_OPTIONS) + (
    'qp_min_i', 'qp_max_i', 'super_frame_policy')


def sample_clocks(client):
    samples = []
    command = 'python3 -u -c "import sys,time; print(\'READY\'); [(print(time.monotonic()),sys.stdout.flush()) for line in sys.stdin]"'
    stdin, stdout, stderr = client.exec_command(command, timeout=10)
    if stdout.readline().strip() != 'READY':
        raise RuntimeError('clock calibration handshake failed')
    for _ in range(8):
        before = time.monotonic()
        stdin.write('sample\n')
        stdin.flush()
        remote = float(stdout.readline().strip())
        after = time.monotonic()
        samples.append(dict(pc_minus_board_s=(before + after) / 2 - remote,
                            uncertainty_s=(after - before) / 2 + time.get_clock_info('monotonic').resolution,
                            pc_at=(before + after) / 2))
    stdin.channel.shutdown_write()
    stdout.channel.close()
    return min(samples, key=lambda item: item['uncertainty_s'])


def monitor_receiver(pid, stop, path):
    import psutil
    with path.open('w', encoding='utf-8') as output:
        while not stop.is_set():
            try:
                process = psutil.Process(pid)
                processes = [process] + process.children(recursive=True)
                memory = [item.memory_info() for item in processes]
                rss = sum(item.rss for item in memory)
                record = dict(at=time.monotonic(), process_tree_rss_mib=rss / 1048576)
                record['process_tree_private_mib'] = (
                    sum(item.private for item in memory) / 1048576
                    if all(hasattr(item, 'private') for item in memory) else None)
                try:
                    record['process_tree_uss_mib'] = sum(
                        item.memory_full_info().uss for item in processes) / 1048576
                except (AttributeError, psutil.Error):
                    record['process_tree_uss_mib'] = None
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=3)
                record['gpu_total_mib'] = gpu.stdout.strip() if gpu.returncode == 0 else None
                output.write(json.dumps(record) + '\n')
                output.flush()
            except (OSError, psutil.Error, subprocess.TimeoutExpired):
                pass
            stop.wait(5)


def run_password_sender(args, command, txlog, remote_log, directory):
    """Use an environment credential without persisting it in trial records."""
    import paramiko
    username, host = args.board_host.split('@', 1)
    password = os.environ[args.ssh_password_env]
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=username, password=password, timeout=10,
                       banner_timeout=10, look_for_keys=False, allow_agent=False)
        _, stdout, stderr = client.exec_command(
            'cd ' + shlex.quote(args.board_dir)
            + ' && sha256sum rknn_yolov8_seg_cam model/yolov8_seg.rknn && uname -a', timeout=10)
        runtime_output = stdout.read().decode('utf-8', errors='replace')
        runtime_errors = stderr.read().decode('utf-8', errors='replace')
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError('sender runtime identity failed: ' + runtime_errors)
        (directory / 'sender_runtime.json').write_text(json.dumps(dict(
            board_host=host, board_dir=args.board_dir, sha256_and_kernel=runtime_output,
            receiver_python=sys.executable), indent=2), encoding='utf-8')
        clocks = {'before': sample_clocks(client)}
        channel = client.get_transport().open_session()
        channel.set_combine_stderr(True)
        local_sender_log = remote_log + '.sender.log'
        if args.board_local_sender_log:
            command = '(' + command + ') >' + shlex.quote(local_sender_log) + ' 2>&1'
        capture_base = remote_log + '.wire'
        if args.capture_ethernet:
            capture_script = remote_log + '.capture.py'
            capture_source = (ROOT / 'tools/capture_ethernet_wire.py').read_bytes()
            with client.open_sftp() as sftp:
                with sftp.open(capture_script, 'wb') as remote_script:
                    remote_script.write(capture_source)
            (directory / 'ethernet_capture_source.py').write_bytes(capture_source)
            (directory / 'ethernet_capture_identity.json').write_text(json.dumps(dict(
                sha256=hashlib.sha256(capture_source).hexdigest(),
                source='ethernet_capture_source.py', remote_path=capture_script), indent=2))
            capture_command = ('python3 ' + shlex.quote(capture_script)
                               + ' --interface ' + shlex.quote(args.capture_ethernet)
                               + ' --duration ' + str(args.duration + args.warmup + 8)
                               + ' --output ' + shlex.quote(capture_base))
            # All setup/ready polling is board-local. No streamed output or SFTP
            # transfers occur inside the measured sender interval.
            command = ('rm -f -- ' + shlex.quote(capture_base + '.ready') + '; '
                       + capture_command + ' >' + shlex.quote(capture_base + '.log')
                       + ' 2>&1 & wire_pid=$!; '
                       + 'for ready_attempt in $(seq 1 50); do test -f '
                       + shlex.quote(capture_base + '.ready')
                       + ' && break; kill -0 "$wire_pid" 2>/dev/null || exit 91; sleep 0.1; done; '
                       + 'test -f ' + shlex.quote(capture_base + '.ready') + ' || exit 92; '
                       + command + '; sender_code=$?; wait "$wire_pid" || exit 93; exit "$sender_code"')
        channel.exec_command(command)
        deadline = time.monotonic() + args.duration + args.warmup + 45
        while True:
            while channel.recv_ready():
                txlog.write(channel.recv(65536).decode('utf-8', errors='replace'))
                txlog.flush()
            if channel.exit_status_ready() and not channel.recv_ready():
                break
            if time.monotonic() > deadline:
                channel.close()
                raise RuntimeError('sender exceeded bounded trial timeout')
            time.sleep(.05)
        code = channel.recv_exit_status()
        clocks['after'] = sample_clocks(client)
        (directory / 'clock_sync.json').write_text(json.dumps(clocks, indent=2), encoding='utf-8')
        with client.open_sftp() as sftp:
            if args.capture_ethernet:
                for extension in ('json', 'jsonl', 'pcap', 'log'):
                    sftp.get(capture_base + '.' + extension,
                             str(directory / ('ethernet_wire.' + extension)))
            sftp.get(remote_log, str(directory / 'encoder.jsonl'))
            if args.board_local_sender_log:
                with sftp.open(local_sender_log, 'r') as saved:
                    txlog.write(saved.read().decode('utf-8', errors='replace'))
                    txlog.flush()
        return code, 0
    finally:
        client.close()


def candidates(stage: str, selected: list[dict] | None = None) -> list[dict]:
    if stage == 'A':
        return [dict(cap=120, width=320, height=180, fps=8, target=86,
                     gop_seconds=2, qp_min=28, qp_max=32, enhancer='x4', adaptive=False,
                     legacy=True)]
    if stage == 'B':
        result = []
        for cap, sizes, rates, targets in (
                (60, [(256, 144), (320, 180)], [6, 8], [40, 45, 50]),
                (120, [(256, 144), (320, 180)], [6, 8, 10], [85, 90, 95, 100]),
                (180, [(320, 180), (384, 216)], [8, 10, 12], [140, 145, 150, 155])):
            for (width, height), fps, target in itertools.product(sizes, rates, targets):
                if width == 384 and fps == 12:
                    continue
                result.append(dict(cap=cap, width=width, height=height, fps=fps,
                                   target=target, gop_seconds=2, qp_min=28, qp_max=32,
                                   enhancer='x4', adaptive=False))
        return result
    if not selected:
        raise ValueError('C/D/E/F require --selected containing measured Pareto candidates')
    result = []
    for item in selected:
        base = {key: item[key] for key in ('cap', 'width', 'height', 'fps', 'target',
                                          'gop_seconds', 'qp_min', 'qp_max', 'enhancer', 'adaptive')}
        base.update({key: item[key] for key in OPTIONAL_SELECTED_FIELDS if key in item})
        if stage == 'C':
            gops = item.get('gop_candidates', (1, 2, 4))
            qp_ranges = item.get('qp_candidates', ((28, 32), (28, 36), (24, 40)))
            if (not 1 <= len(gops) <= 8 or not 1 <= len(qp_ranges) <= 8 or
                    any(not isinstance(gop, int) or not 1 <= gop <= 8 for gop in gops) or
                    any(len(pair) != 2 or any(not isinstance(value, int) for value in pair)
                        or not 0 <= pair[0] <= pair[1] <= 51 for pair in qp_ranges)):
                raise ValueError('C requires 1..8 GOP values (1..8s) and 1..8 ordered QP ranges (0..51)')
            result.extend(dict(base, gop_seconds=gop, qp_min=lo, qp_max=hi)
                          for gop in gops for lo, hi in qp_ranges)
        elif stage == 'D':
            result.extend(dict(base, enhancer=enhancer) for enhancer in ('none', 'x2', 'x4'))
        elif stage == 'E':
            result.extend(dict(base, adaptive=adaptive) for adaptive in (False, True))
        else:
            result.append(base)
    return result


def commands(config: dict, args, directory: Path) -> tuple[list[str], list[str], str]:
    run_key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    remote_log = '/tmp/vq_' + run_key + '_' + directory.name + '.encoder.jsonl'
    profile = 'gan4' if config.get('legacy') else f"ganq{config['cap']}"
    sender = ['./rknn_yolov8_seg_cam', f'--rate-profile={profile}',
              '--model=model/yolov8_seg.rknn', '--audio=off', '--audio-reserve-bitrate=0',
              '--preview=off', '--profile-control=', '--camera-device=/dev/video-camera0',
              f'--udp-host={args.pc_host}', f'--udp-port={args.port}',
              f"--gan-fps={config['fps']}", f"--gan-video-bitrate-kbps={config['target']}",
              f"--gan-gop-seconds={config['gop_seconds']}",
              f"--gan-qp-min-p={config['qp_min']}", f"--gan-qp-max-p={config['qp_max']}",
              f"--gan-qp-min-i={config.get('qp_min_i', config['qp_min'])}", '--gan-inference-fps=0',
              f"--qp-max-i={config.get('qp_max_i', config['qp_max'])}",
              '--gan-debreath=off', '--gan-intra-refresh=off',
              f"--gan-super-frame={config.get('super_frame_policy', 'current')}",
              f"--adaptive-roi={'on' if config['adaptive'] else 'off'}",
              f'--encoder-debug-log={remote_log}',
              # Leave a source-frame tail beyond the measured interval. Capture
              # timestamps can jitter across the nominal last-frame boundary;
              # that tail must not shorten an otherwise complete measurement.
              f"--max-frames={int((args.duration + args.warmup + 1) * config['fps'])}"]
    if config.get('legacy'):
        # Stage A is the historical 320x180 gan4 reference.  Keep its opt-in
        # native-720p geometry after the gan4 default moves to rate120 input.
        sender.append('--gan-native-720p=on')
    sender += [f'--{flag}={config[key]}' for key, flag in OPTIONAL_ENCODER_OPTIONS.items()
               if key in config]
    if not config.get('legacy'):
        sender += [f"--encoder-width={config['width']}", f"--encoder-height={config['height']}"]
    input_video = args.input_video.format(fps=config['fps']) if args.input_video else None
    loop_input = bool(input_video and getattr(args, 'loop_input_video', False))
    fifo = '/tmp/vq_' + run_key + '.input.mkv'
    if input_video:
        sender.append('--input-video=' + (fifo if loop_input else input_video))
    remote = f'cd {shlex.quote(args.board_dir)} && export LD_LIBRARY_PATH="$PWD/lib" && exec ' + shlex.join(sender)
    if loop_input:
        cleanup = 'kill "$producer" 2>/dev/null || true; rm -f -- ' + shlex.quote(fifo)
        producer = shlex.join(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
                              '-stream_loop', '-1', '-i', input_video, '-c', 'copy', '-f', 'matroska', fifo])
        remote = (f'cd {shlex.quote(args.board_dir)} && export LD_LIBRARY_PATH="$PWD/lib" && '
                  f'mkfifo {shlex.quote(fifo)} && {{ {producer} >{shlex.quote(fifo + ".log")} 2>&1 & '
                  f'producer=$!; trap {shlex.quote(cleanup)} EXIT; ' + shlex.join(sender) + '; }')
    receiver = [sys.executable, '-u', str(ROOT / 'tools/live_h265_hud.py'),
                f'--udp-port={args.port}', '--rotate=none',
                f'--duration={args.duration + args.warmup + 15}',
                '--gan-enhancer=' + ('none' if config['enhancer'] == 'none' else 'esrgan'),
                '--ganq-scale=' + ('2' if config['enhancer'] == 'x2' else '4'),
                '--gan-execution-provider=' + ('auto' if config['enhancer'] == 'none' else args.provider),
                '--gan-esrgan-model=model/RealESRGAN_x2_dynamic.onnx',
                '--gan-esrgan-x4-model=model/RealESRGAN_x4plus_dynamic.onnx',
                f'--gan-debug-log={directory / "enhancer.jsonl"}',
                f'--quality-video-log={directory / "received.h265"}',
                f'--quality-packet-log={directory / "packets.jsonl"}']
    if not getattr(args, 'display', False):
        receiver.append('--headless')
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', args.board_host, remote], receiver, remote_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=list('ABCDEF'), default='B')
    parser.add_argument('--selected', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/video_quality_optimization/sweep')
    parser.add_argument('--board-host', default='root@192.168.0.101')
    parser.add_argument('--board-dir', default='/opt/atk/rknn_yolov8_seg_cam')
    parser.add_argument('--pc-host', default='192.168.0.100')
    parser.add_argument('--port', type=int, default=5004)
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--scene', choices=SCENES, default='static_person')
    parser.add_argument('--input-video', help='fixed source path on board; optional {fps} selects a time-aligned resample of one reference')
    parser.add_argument('--loop-input-video', action='store_true',
                        help='remux source repeatedly through a bounded FIFO for long trials, without duplicating large files')
    parser.add_argument('--provider', choices=('cuda', 'tensorrt-fp16'), default='tensorrt-fp16')
    parser.add_argument('--display', action='store_true',
                        help='show the actual receiver window; default benchmarks remain headless')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='skip recorded trials and archive interrupted trial directories')
    parser.add_argument('--ssh-password-env', help='environment variable containing SSH password; omitted uses SSH keys')
    parser.add_argument('--board-local-sender-log', action='store_true',
                        help='store sender stdout on board and retrieve after measurement to avoid SSH log traffic')
    parser.add_argument('--capture-ethernet', metavar='INTERFACE',
                        help='capture every outbound Ethernet protocol on the board; requires root and board-local logging')
    args = parser.parse_args()
    if args.board_local_sender_log and not args.ssh_password_env:
        parser.error('--board-local-sender-log currently requires --ssh-password-env')
    if args.capture_ethernet and not args.board_local_sender_log:
        parser.error('--capture-ethernet requires --board-local-sender-log')
    args.output_root = args.output_root.resolve()
    minimum = 1800 if args.stage == 'F' else 60
    if args.duration < minimum or args.warmup < 0:
        parser.error(f'stage {args.stage} requires at least {minimum}s measurement and nonnegative warmup')
    selected = json.loads(args.selected.read_text(encoding='utf-8')) if args.selected else None
    configs = candidates(args.stage, selected)
    args.output_root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff'], cwd=ROOT)
    source_hash = hashlib.sha256(diff)
    for path in sorted(list((ROOT / 'cpp').rglob('*.h')) + list((ROOT / 'cpp').rglob('*.cc')) +
                       list((ROOT / 'tools').glob('*.py')) + list((ROOT / 'tools').glob('*.sh')) +
                       list((ROOT / 'tools').glob('*.ps1'))):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    failures = []
    for config in configs:
        if args.display:
            config = dict(config, display=True)
        if args.board_local_sender_log:
            config = dict(config, board_local_sender_log=True)
        if args.capture_ethernet:
            config = dict(config, capture_ethernet=args.capture_ethernet,
                          ethernet_capture_version='kernel_timestamp_v1')
        tag = args.stage + '_' + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        directory = args.output_root / (args.scene + '_' + tag)
        if directory.exists() and args.resume:
            previous = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            if previous.get('status') == 'RECORDED':
                print(f'SKIP recorded {directory.name}', flush=True)
                continue
            previous['previous_status'] = previous.get('status')
            previous['status'] = 'INTERRUPTED'
            (directory / 'config.json').write_text(json.dumps(previous, indent=2), encoding='utf-8')
            directory.rename(directory.with_name(directory.name + '_interrupted_' + str(time.time_ns())))
        directory.mkdir(exist_ok=False)
        sender, receiver, remote_log = commands(config, args, directory)
        record = dict(config, stage=args.stage, scene=args.scene, duration_s=args.duration,
                      warmup_s=args.warmup, input_video=args.input_video, revision=revision,
                      display=args.display,
                      loop_input_video=args.loop_input_video,
                      source_sha256=source_hash.hexdigest(), sender=sender, receiver=receiver,
                      status='STARTING' if args.run else 'NOT RUN',
                      strict_same_source=bool(args.input_video))
        path = directory / 'config.json'
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f'{record["status"]} {directory.name}', flush=True)
        if not args.run:
            continue
        with (directory / 'receiver.log').open('w', encoding='utf-8') as rxlog, \
                (directory / 'sender.log').open('w', encoding='utf-8') as txlog:
            rx_started = time.monotonic()
            rx = subprocess.Popen(receiver, cwd=ROOT, stdout=rxlog, stderr=subprocess.STDOUT)
            record.update(status='RUNNING', receiver_pid=rx.pid, started_at_unix_s=time.time())
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(f'RUNNING {directory.name}', flush=True)
            memory_stop = threading.Event()
            memory_thread = threading.Thread(target=monitor_receiver,
                args=(rx.pid, memory_stop, directory / 'memory.jsonl'), daemon=True)
            memory_thread.start()
            try:
                time.sleep(2)
                if rx.poll() is not None:
                    raise RuntimeError('receiver exited before sender startup')
                if args.ssh_password_env:
                    tx_code, copy_code = run_password_sender(args, sender[-1], txlog,
                                                             remote_log, directory)
                else:
                    tx = subprocess.run(sender, stdout=txlog, stderr=subprocess.STDOUT,
                                        timeout=args.duration + args.warmup + 45)
                    copy = subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                       args.board_host + ':' + remote_log, str(directory / 'encoder.jsonl')],
                                      capture_output=True, timeout=20)
                    tx_code, copy_code = tx.returncode, copy.returncode
                record['returncodes'] = [tx_code, None, copy_code]
                if tx_code != 0 or copy_code != 0:
                    raise RuntimeError(f'sender exit={tx_code}; encoder log copy exit={copy_code}')
                remaining = args.duration + args.warmup + 25 - (time.monotonic() - rx_started)
                rx.wait(timeout=max(10, remaining))
                record['status'] = 'RECORDED' if tx_code == rx.returncode == copy_code == 0 else 'ERROR'
                record['returncodes'] = [tx_code, rx.returncode, copy_code]
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                record.update(status='ERROR', reason=str(error))
            finally:
                memory_stop.set()
                memory_thread.join(timeout=4)
                if rx.poll() is None:
                    rx.terminate()
                    try:
                        rx.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        rx.kill()
                        rx.wait()
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f'{record["status"]} {directory.name}', flush=True)
        if record['status'] == 'ERROR':
            failures.append(directory.name)
    print(f'{len(configs)} configurations saved in {args.output_root}')
    if failures:
        print('FAILED trials: ' + ', '.join(failures), file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

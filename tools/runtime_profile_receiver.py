"""Runtime mode router with bounded, generation-isolated receiver workers.

The public process owns the board UDP port. Workers only receive loopback UDP
once ready, starting at a complete IDR. No encoded FIFO is accumulated while a
model or Python process starts. Each mode replaces, rather than stacks, workers.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
import psutil


_worker_channel = None
_network_stats = None
_network_at = 0.0
NETWORK_FIELDS = ('rx_fps','p_fps','i_fps','p_frames','i_frames','rtp_kbps',
                  'wire_kbps','pps','packet_last_bytes','packet_avg_bytes',
                  'packet_max_bytes','lost','reordered','duplicates','packets','idr_age')


def runtime_network_stats():
    if time.monotonic()-_network_at < 1.5:
        return _network_stats
    return None


def runtime_worker_status(**fields):
    if _worker_channel is not None:
        control, address, token = _worker_channel
        try:
            control.sendto(json.dumps(dict(token=token, kind='status', fields=fields)).encode(), address)
        except OSError:
            pass


def notify_runtime_ready(stop):
    global _worker_channel
    port = os.environ.get("VC_RUNTIME_READY_PORT")
    if not port:
        return
    token = os.environ['VC_RUNTIME_TOKEN']
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control.bind(('127.0.0.1', 0))
    control.settimeout(.2)
    _worker_channel = (control, ('127.0.0.1', int(port)), token)
    control.sendto(json.dumps(dict(token=token, port=control.getsockname()[1])).encode(),
                   ('127.0.0.1', int(port)))
    def watch():
        global _network_stats, _network_at
        try:
            while not stop.is_set():
                try:
                    message, _ = control.recvfrom(8192)
                except socket.timeout:
                    continue
                if message == ('STOP:'+token).encode():
                    stop.set()
                elif message.startswith(('NET:'+token+':').encode()):
                    try:
                        _network_stats = json.loads(message.split(b':',2)[2])
                        _network_at = time.monotonic()
                    except ValueError:
                        pass
        finally:
            control.close()
    threading.Thread(target=watch, daemon=True, name='runtime-worker-control').start()


def worker_arguments(argv, port, output_fps):
    result = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        key = item.split('=', 1)[0]
        if key in ('--udp-port', '--video-port', '--fps', '--duration', '--quality-packet-log') or (output_fps and key == '--gan-display-fps'):
            skip = '=' not in item
            continue
        result.append(item)
    result += ['--udp-port='+str(port), '--duration=0']
    if output_fps:
        result.append('--fps='+str(output_fps))
    return result


def selected_fps(profile, default_fps):
    from live_h265_hud import is_gan_profile
    if not profile or not is_gan_profile(profile):
        return None
    # v3 zero explicitly disables interpolation, overriding a startup --fps.
    if 'output_fps' in profile:
        return profile['output_fps'] or None
    return default_fps


def run_runtime_receiver(args, listen_port):
    from live_h265_hud import ProfileSwitchGate, RtpStats, profile_switch_key
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    receiver.bind(('0.0.0.0', listen_port))
    receiver.settimeout(.05)
    ingress_stop = threading.Event()
    ingress_lock = threading.Lock()
    ingress = collections.deque(maxlen=256)
    ingress_overflow = 0
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control.bind(('127.0.0.1', 0)); control.setblocking(False)
    forward = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stats = RtpStats()
    for name in ('gan_debug_log', 'quality_packet_log', 'quality_video_log'):
        path = getattr(args, name, None)
        if path:
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'')
    packet_log = None
    if args.quality_packet_log:
        packet_log = Path(args.quality_packet_log).open('w', encoding='utf8', buffering=1)
    def receive_network():
        nonlocal ingress_overflow
        while not ingress_stop.is_set():
            try:
                packet, _ = receiver.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            at = time.perf_counter()
            stats.on_packet(packet)
            if packet_log is not None and len(packet) >= 12:
                offset = RtpStats._payload_offset(packet)
                packet_log.write(json.dumps(dict(at=at, udp_bytes=len(packet),
                    rtp_payload_bytes=max(0,len(packet)-(offset or len(packet))),
                    wire_bytes=RtpStats._wire_bytes(len(packet)),
                    rtp_timestamp=int.from_bytes(packet[4:8], 'big'), marker=bool(packet[1]&128),
                    scope='board-to-PC UDP ingress before worker routing'))+'\n')
            with ingress_lock:
                if len(ingress) == ingress.maxlen:
                    ingress_overflow += 1
                ingress.append((at, packet))
    network_thread = threading.Thread(target=receive_network, daemon=True, name='runtime-network-ingress')
    network_thread.start()
    last_network_update = 0.0
    log = None
    if args.gan_debug_log:
        path = Path(str(args.gan_debug_log)+'.control.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open('w', encoding='utf8', buffering=1)
    def event(kind, **fields):
        record = dict(type='RUNTIME_'+kind, at=time.perf_counter(), **fields)
        if log:
            log.write(json.dumps(record)+'\n')
        if kind not in ('STATUS', 'WORKER_STATUS'):
            print('RUNTIME '+kind+': '+str(fields), flush=True)
    event('START', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    process, worker_control, key, token = None, None, None, None
    ready, worker_port, generation = False, None, 0
    gate, gate_started = ProfileSwitchGate(), None
    retired = {}
    known_children = {}
    last_child_audit = time.perf_counter()
    start = time.perf_counter(); last_status = start; code = 0

    def stop_worker():
        nonlocal process
        if process is None:
            return
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        children = list({p.pid:p for p in children + list(known_children.values())}.values())
        if worker_control:
            control.sendto(('STOP:'+token).encode(), ('127.0.0.1', worker_control))
        try:
            process.wait(3)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(3)
        # Also reap children surviving a failed or abruptly stopped parent.
        for child in children:
            try:
                if child.is_running():
                    child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(children, timeout=2)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        event('WORKER_STOP', pid=process.pid, exit_code=process.returncode)
        for name in ('quality_video_log',):
            path = getattr(args, name, None)
            if path:
                for suffix in ('.runtime.json', '.hud.png'):
                    src = Path(path).with_suffix(suffix)
                    if src.exists():
                        dst = src.with_name(src.stem+'.g'+str(generation)+src.suffix)
                        dst.write_bytes(src.read_bytes())
        process = None
        known_children.clear()

    print(('INTERPOLATION LISTENING' if args.fps else 'RUNTIME LISTENING')+
          ': UDP '+str(listen_port)+'; waiting for sender profile', flush=True)
    try:
        while args.duration <= 0 or time.perf_counter()-start < args.duration:
            now = time.perf_counter()
            if process is not None and process.poll() is None and now-last_child_audit >= .5:
                known_children = {pid:p for pid,p in known_children.items() if p.is_running()}
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        known_children[child.pid] = child
                except psutil.Error:
                    pass
                last_child_audit = now
            if process is not None and process.poll() is not None:
                event('WORKER_EXIT', exit_code=process.returncode)
                code = process.returncode or 0
                break
            try:
                message, _ = control.recvfrom(1024)
                notice = json.loads(message)
                if notice.get('token') == token and notice.get('kind') == 'stop':
                    break
                if notice.get('token') == token and notice.get('kind') == 'status':
                    event('WORKER_STATUS', pid=process.pid, generation=generation, **notice['fields'])
                elif notice.get('token') == token:
                    worker_control = int(notice['port'])
                    # Drop socket backlog from the startup/old generation;
                    # clock calibration and decode must see fresh input.
                    with ingress_lock:
                        ingress.clear()
                    ready = True; gate = ProfileSwitchGate(); gate_started = None
                    event('WORKER_READY', pid=process.pid, generation=generation)
            except (BlockingIOError, ValueError, KeyError):
                pass
            if process is not None and not ready and now-worker_started > 20:
                raise RuntimeError('receiver worker did not bind its UDP port within 20s')
            if worker_control and now-last_network_update >= .5:
                network = stats.snapshot()
                control.sendto(('NET:'+token+':'+json.dumps({k:network[k] for k in NETWORK_FIELDS})).encode(),
                               ('127.0.0.1', worker_control))
                last_network_update = now
            with ingress_lock:
                entry = ingress.popleft() if ingress else None
            if entry is None:
                time.sleep(.001)
                continue
            arrival, packet = entry
            if now-arrival > .25:
                gate = ProfileSwitchGate(); gate_started = None
                continue
            if len(packet) < 12:
                continue
            profile = RtpStats._profile_extension(packet)
            extension = 12 + 4*(packet[0] & 15)
            if profile is None and packet[0] & 16 and packet[extension:extension+2] == b'RO':
                continue  # malformed project metadata is not a legacy stream
            next_key = profile_switch_key(profile)
            if next_key != key:
                if now-retired.get(next_key, -100) < 2:
                    continue
                if key is not None:
                    retired[key] = now
                retired = {k:t for k,t in retired.items() if now-t < 2}
                stop_worker()
                latest_profile = stats.snapshot()['profile']
                if latest_profile is not None:
                    profile = latest_profile
                key = profile_switch_key(profile); generation += 1
                with ingress_lock:
                    ingress.clear()
                ready = False; worker_control = None; token = secrets.token_hex(16)
                temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                temp.bind(('127.0.0.1', 0)); worker_port = temp.getsockname()[1]; temp.close()
                if args.quality_video_log:
                    Path(args.quality_video_log).with_suffix('.hud.png').unlink(missing_ok=True)
                fps = selected_fps(profile, args.fps)
                command = [sys.executable, '-u', str(Path(__file__).with_name('live_h265_hud.py')),
                           *worker_arguments(sys.argv[1:], worker_port, fps)]
                env = dict(os.environ, VC_RUNTIME_RECEIVER_WORKER='1',
                           VC_RUNTIME_APPEND_LOGS='1', VC_RUNTIME_TOKEN=token,
                           VC_RUNTIME_READY_PORT=str(control.getsockname()[1]))
                worker_started = time.perf_counter()
                process = subprocess.Popen(command, env=env, stdout=sys.stdout, stderr=sys.stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                event('WORKER_START', pid=process.pid, profile=profile, output_fps=fps,
                      generation=generation, waiting='fresh complete IDR; model warmup')
                continue
            if not ready:
                continue
            accepted, packets = gate.feed(packet, profile)
            if gate.pending_packets:
                gate_started = gate_started or now
                if len(gate.pending_packets) > 2048 or now-gate_started > 1:
                    gate = ProfileSwitchGate(); gate_started = None
                    event('IDR_RESET', reason='bounded startup access unit')
                    continue
            else:
                gate_started = None
            for item in packets:
                forward.sendto(item, ('127.0.0.1', worker_port))
            if now-last_status >= 5:
                event('STATUS', stats=stats.snapshot(), generation=generation)
                last_status = now
    except KeyboardInterrupt:
        pass
    except Exception as error:
        event('ERROR', error=str(error)); code = 1
    finally:
        stop_worker()
        event('FINISH', stats=stats.snapshot(), ingress_overflow=ingress_overflow)
        ingress_stop.set(); receiver.close(); network_thread.join(1)
        control.close(); forward.close()
        if packet_log:
            packet_log.close()
        if log:
            log.close()
    return code

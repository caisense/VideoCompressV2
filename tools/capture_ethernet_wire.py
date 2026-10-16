#!/usr/bin/env python3
"""Linux root-only, board-local outbound Ethernet capture without tcpdump.

AF_PACKET records exclude FCS. Physical accounting adds FCS, preamble and IFG
after minimum Ethernet padding. This is an independent observation, not shaping.
"""
import argparse
import json
import socket
import struct
import time
from pathlib import Path


def physical_bytes(frame_length):
    if frame_length < 14:
        raise ValueError('not a complete Ethernet header')
    return max(frame_length, 60) + 24


def counters(interface):
    root = Path('/sys/class/net') / interface / 'statistics'
    return {name: int((root / name).read_text())
            for name in ('tx_bytes', 'tx_packets', 'tx_dropped', 'tx_errors')}


def clock_pair():
    before = time.monotonic()
    wall = time.time()
    after = time.monotonic()
    return dict(monotonic_s=(before + after) / 2, wall_s=wall,
                uncertainty_s=(after - before) / 2)


def capture(interface, output, duration):
    # Keep every outbound protocol, including SSH, ARP and background traffic.
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    # Linux SO_TIMESTAMPNS: preserve kernel packet observation time rather
    # than assigning delayed userspace recv time to a physical-rate window.
    timestamp_ns = getattr(socket, 'SO_TIMESTAMPNS', 35)
    sock.setsockopt(socket.SOL_SOCKET, timestamp_ns, 1)
    sock.bind((interface, 0))
    sock.settimeout(.2)
    start = clock_pair()
    initial = counters(interface)
    count = wire = 0
    first = last = None
    max_receive_delay = 0.0
    max_wall_offset_change = 0.0
    with Path(str(output) + '.pcap').open('wb') as pcap, \
            Path(str(output) + '.jsonl').open('w') as records:
        pcap.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        Path(str(output) + '.ready').write_text(json.dumps(start))
        deadline = start['monotonic_s'] + duration
        while time.monotonic() < deadline:
            try:
                frame, ancillary, flags, address = sock.recvmsg(65535, 128)
            except socket.timeout:
                continue
            at = clock_pair()
            if address[2] != socket.PACKET_OUTGOING:
                continue
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise RuntimeError('truncated packet or timestamp ancillary data')
            timestamps = [struct.unpack('@ll', data[:struct.calcsize('@ll')])
                          for level, kind, data in ancillary
                          if level == socket.SOL_SOCKET and kind == timestamp_ns]
            if len(timestamps) != 1:
                raise RuntimeError('missing kernel packet timestamp')
            seconds, nanoseconds = timestamps[0]
            wall = seconds + nanoseconds / 1e9
            observed = at['monotonic_s'] + wall - at['wall_s']
            max_receive_delay = max(max_receive_delay, at['wall_s'] - wall)
            max_wall_offset_change = max(max_wall_offset_change, abs(
                (at['wall_s'] - at['monotonic_s'])
                - (start['wall_s'] - start['monotonic_s'])))
            sec = int(wall)
            pcap.write(struct.pack('<IIII', sec, int((wall-sec)*1e6), len(frame), len(frame)))
            pcap.write(frame)
            size = physical_bytes(len(frame))
            records.write(json.dumps(dict(at=observed, kernel_wall_s=wall,
                                          recv_monotonic_s=at['monotonic_s'], frame_bytes=len(frame),
                                          wire_bytes=size)) + '\n')
            count += 1
            wire += size
            first = at if first is None else first
            last = at
        # PACKET_STATISTICS resets on read, so obtain it exactly once at end.
        packets, drops = struct.unpack('II', sock.getsockopt(263, 6, 8))
    final = counters(interface)
    end = clock_pair()
    sock.close()
    result = dict(interface=interface, start=start, end=end, first=first, last=last,
                  outgoing_packets=count, wire_bytes=wire,
                  socket_packets_all_directions=packets, socket_drops=drops,
                  counters_before=initial, counters_after=final,
                  capture_complete=drops == 0,
                  max_receive_delay_s=max_receive_delay,
                  max_wall_offset_change_s=max_wall_offset_change,
                  timestamp_scope='SO_TIMESTAMPNS kernel AF_PACKET observation mapped through local wall/monotonic pairs; not a physical tap',
                  accounting='max(Ethernet length without FCS,60)+24; all outbound protocols')
    Path(str(output) + '.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='eth1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('duration must be positive')
    capture(args.interface, args.output, args.duration)

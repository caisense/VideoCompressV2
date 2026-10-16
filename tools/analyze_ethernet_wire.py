#!/usr/bin/env python3
"""Summarize independent board Ethernet observations in capture-PTS time.

Reports observed rates, never silently substitutes application RTP accounting.
Userspace receive timestamps do not prove instantaneous physical transmission.
"""
import argparse
import json
import struct
from pathlib import Path

from analyze_video_quality_sweep import read_jsonl


def pcap_protocols(path):
    """Classify retained Ethernet frames, keeping management traffic visible."""
    with path.open('rb') as source:
        header = source.read(24)
        if len(header) != 24 or header[:4] != b'\xd4\xc3\xb2\xa1':
            raise ValueError('expected little-endian microsecond Ethernet pcap')
        if struct.unpack('<I', header[20:24])[0] != 1:
            raise ValueError('pcap link type is not Ethernet')
        while True:
            packet = source.read(16)
            if not packet:
                return
            if len(packet) != 16:
                raise ValueError('truncated pcap record header')
            _, _, included, original = struct.unpack('<IIII', packet)
            frame = source.read(included)
            if len(frame) != included or included != original or included < 14:
                raise ValueError('truncated Ethernet capture')
            kind = 'other'
            if frame[12:14] == b'\x08\x00' and len(frame) >= 34:
                ihl = (frame[14] & 15) * 4
                protocol = frame[23]
                offset = 14 + ihl
                if ihl >= 20 and len(frame) >= offset + 4:
                    ports = struct.unpack('!HH', frame[offset:offset+4])
                    if protocol == 6:
                        kind = 'ssh' if 22 in ports else 'tcp_other'
                    elif protocol == 17:
                        kind = 'udp_' + str(ports[1])
            elif frame[12:14] == b'\x08\x06':
                kind = 'arp'
            yield dict(protocol=kind, frame_bytes=original,
                       physical_bytes=max(original, 60) + 24)


def summarize(directory):
    config = json.loads((directory / 'config.json').read_text())
    metadata = json.loads((directory / 'ethernet_wire.json').read_text())
    encoded = [r for r in read_jsonl(directory / 'encoder.jsonl')
               if r.get('TYPE') == 'ENC_FRAME']
    if not encoded:
        raise ValueError('no encoded frames')
    begin = encoded[0]['PTS'] / 1e6 + config['warmup_s']
    end = begin + config['duration_s']
    packets = read_jsonl(directory / 'ethernet_wire.jsonl')
    protocols = list(pcap_protocols(directory / 'ethernet_wire.pcap'))
    if len(protocols) != len(packets):
        raise ValueError('pcap/packet observation count mismatch')
    by_protocol = {}
    for packet, captured in zip(packets, protocols):
        if (packet['frame_bytes'] != captured['frame_bytes']
                or packet['wire_bytes'] != captured['physical_bytes']):
            raise ValueError('pcap/observation Ethernet length accounting mismatch')
        if begin <= packet['at'] < end:
            protocol = captured['protocol']
            by_protocol[protocol] = by_protocol.get(protocol, 0) + packet['wire_bytes']
    selected = [r for r in packets if begin <= r['at'] < end]
    duration = end - begin
    windows = {}
    # Search trailing windows ending at each observed packet, clipped only
    # after restricting endpoints to complete windows inside measurement.
    for window in (1, 5, 60):
        left = right = total = peak = 0
        endpoints = sorted({end, begin + window} | {
            r['at'] for r in selected if r['at'] >= begin + window})
        for endpoint in endpoints:
            while right < len(selected) and selected[right]['at'] <= endpoint:
                total += selected[right]['wire_bytes']
                right += 1
            while left < right and selected[left]['at'] < endpoint - window:
                total -= selected[left]['wire_bytes']
                left += 1
            peak = max(peak, total)
        windows[str(window)] = peak * 8 / window if duration >= window else None
    before, after = metadata['counters_before'], metadata['counters_after']
    result = dict(measurement_board_monotonic_s=[begin, end],
                  duration_s=duration, outgoing_packets=len(selected),
                  measured_wire_bytes_by_protocol=by_protocol,
                  whole_link_average_bps=sum(r['wire_bytes'] for r in selected)*8/duration,
                  whole_link_peak_trailing_window_bps=windows,
                  socket_drops=metadata['socket_drops'],
                  capture_covers_window=(metadata['start']['monotonic_s'] <= begin
                                         and metadata['end']['monotonic_s'] >= end),
                  nic_packet_delta=after['tx_packets']-before['tx_packets'],
                  captured_packet_total=metadata['outgoing_packets'],
                  nic_capture_boundary_packet_difference=(after['tx_packets']-before['tx_packets']
                                                          -metadata['outgoing_packets']),
                  cap_bps=config['cap']*1000,
                  scope='All outbound Ethernet protocols; physical length accounting',
                  timestamp_scope=metadata.get('timestamp_scope'),
                  max_receive_delay_s=metadata.get('max_receive_delay_s'),
                  max_wall_offset_change_s=metadata.get('max_wall_offset_change_s'),
                  final_acceptance=False,
                  acceptance_limitation='Validate NIC/capture boundary completeness and timestamp uncertainty before final cap verdict')
    result['accounting_health_verified'] = bool(
        result['capture_covers_window'] and result['socket_drops'] == 0
        and result['nic_capture_boundary_packet_difference'] == 0
        and 'SO_TIMESTAMPNS' in (result['timestamp_scope'] or '')
        and result['max_wall_offset_change_s'] is not None
        and result['max_wall_offset_change_s'] <= .001)
    result['whole_link_cap_screening_verified'] = bool(
        result['accounting_health_verified']
        and result['whole_link_average_bps'] <= result['cap_bps']
        and all(rate is not None and rate <= result['cap_bps'] + 1238*8/int(window)
                for window, rate in windows.items()))
    result['window_policy'] = 'same as RTP screen: cap plus one 1238-byte packet per window; average <= cap'
    (directory / 'ethernet_wire_summary.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))

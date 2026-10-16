import unittest
import json
import tempfile
import struct
from pathlib import Path

from capture_ethernet_wire import physical_bytes
from analyze_ethernet_wire import summarize


class EthernetAccountingTests(unittest.TestCase):
    def test_minimum_padding_and_frame_overheads(self):
        self.assertEqual(physical_bytes(42), 84)  # ARP, before NIC padding
        self.assertEqual(physical_bytes(60), 84)
        self.assertEqual(physical_bytes(1514), 1538)
        self.assertEqual(physical_bytes(1518), 1542)  # VLAN header retained

    def test_incomplete_header_is_rejected(self):
        with self.assertRaises(ValueError):
            physical_bytes(13)

    def test_full_duration_window_and_background_packets_are_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text(json.dumps(dict(warmup_s=0, duration_s=60, cap=60)))
            (root / 'encoder.jsonl').write_text(json.dumps(dict(TYPE='ENC_FRAME', PTS=1000000))+'\n')
            (root / 'ethernet_wire.json').write_text(json.dumps(dict(
                counters_before=dict(tx_packets=0), counters_after=dict(tx_packets=4),
                outgoing_packets=4, socket_drops=0,
                start=dict(monotonic_s=0), end=dict(monotonic_s=62))))
            packets = [dict(at=0, wire_bytes=900), dict(at=1, wire_bytes=84),
                       dict(at=60, wire_bytes=100), dict(at=61, wire_bytes=900)]
            for packet in packets:
                packet['frame_bytes'] = packet['wire_bytes'] - 24
            (root / 'ethernet_wire.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in packets))
            with (root / 'ethernet_wire.pcap').open('wb') as capture:
                capture.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
                for packet in packets:
                    frame = bytes(12) + b'\x08\x06' + bytes(packet['frame_bytes'] - 14)
                    capture.write(struct.pack('<IIII', 0, 0, len(frame), len(frame)))
                    capture.write(frame)
            result = summarize(root)
            self.assertEqual(result['outgoing_packets'], 2)
            self.assertAlmostEqual(result['whole_link_average_bps'], 184*8/60)
            self.assertAlmostEqual(result['whole_link_peak_trailing_window_bps']['60'], 184*8/60)
            self.assertFalse(result['final_acceptance'])
            self.assertEqual(result['measured_wire_bytes_by_protocol'], {'arp': 184})
            self.assertFalse(result['whole_link_cap_screening_verified'])
            metadata = json.loads((root / 'ethernet_wire.json').read_text())
            metadata.update(timestamp_scope='SO_TIMESTAMPNS kernel AF_PACKET',
                            max_wall_offset_change_s=.00001, max_receive_delay_s=.001)
            (root / 'ethernet_wire.json').write_text(json.dumps(metadata))
            self.assertTrue(summarize(root)['whole_link_cap_screening_verified'])
            metadata['socket_drops'] = 1
            (root / 'ethernet_wire.json').write_text(json.dumps(metadata))
            self.assertFalse(summarize(root)['whole_link_cap_screening_verified'])
            packets[1]['wire_bytes'] += 1
            (root / 'ethernet_wire.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in packets))
            with self.assertRaisesRegex(ValueError, 'length accounting mismatch'):
                summarize(root)
            packets[1]['wire_bytes'] -= 1
            (root / 'ethernet_wire.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in packets))
            metadata.update(socket_drops=0, max_wall_offset_change_s=.1)
            (root / 'ethernet_wire.json').write_text(json.dumps(metadata))
            self.assertFalse(summarize(root)['whole_link_cap_screening_verified'])


if __name__ == '__main__':
    unittest.main()

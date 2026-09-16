#!/usr/bin/env python3
"""Compare the authoritative PC parser against C++ on sender-generated RTP."""

import ast
import subprocess
import sys
from pathlib import Path


def load_authoritative_classes(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, ast.ClassDef) and
                node.name in ("RtpStats", "HevcRtpDepacketizer")]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["RtpStats"], namespace["HevcRtpDepacketizer"]


import __future__


def main():
    fixture, authoritative = sys.argv[1:3]
    stats, depacketizer_type = load_authoritative_classes(authoritative)
    pc = depacketizer_type()
    lines = subprocess.check_output([fixture, "--dump"], text=True).splitlines()
    assert lines
    for line in lines:
        packet_hex, offset, profile_csv, nal_type, annex_hex = line.split("|")
        packet = bytes.fromhex(packet_hex)
        profile = stats._profile_extension(packet)
        assert stats._payload_offset(packet) == int(offset)
        assert profile is not None
        expected = ",".join(str(value) for value in (
            1, profile["width"], profile["height"], profile["fps"],
            profile["generation"]))
        assert profile_csv == expected
        python_nal = stats._nal_type(packet)
        assert (255 if python_nal is None else python_nal) == int(nal_type)
        assert pc.feed(packet).hex() == annex_hex
    print(f"Python/C++ protocol parity: {len(lines)} sender datagrams")


if __name__ == "__main__":
    main()

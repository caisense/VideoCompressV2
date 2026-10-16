#!/usr/bin/env python3
"""Run the platform-independent native suite with g++ or a portable Zig compiler."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zig', type=Path)
    parser.add_argument('--only', help='run one named regression')
    parser.add_argument('--output', type=Path, default=Path('runs/video_quality_optimization/native'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    text = (root / 'cpp/tests/build_roi_tests.sh').read_text().replace('\\\n', ' ')
    results = []
    excluded = ('test_udp_sender', 'test_detection_event_sender', 'test_async_rtp_sender')
    for command, name in re.findall(r'g\+\+ (.*?) -o "\$out_dir/([^"]+)"', text):
        if args.only and name != args.only:
            continue
        sources = re.findall(r'"\$cpp_dir/([^\"]+\.cc)"', command)
        if name in excluded:
            results.append(dict(name=name, status='NOT RUN', reason='POSIX socket test requires Linux build_roi_tests.sh'))
            continue
        binary = (args.output / (name + '.exe')).resolve()
        source_paths = [str(root / 'cpp' / source) for source in sources]
        if name == 'test_h265_sdp':
            # Relocate only its fixed /tmp fixture filenames for Windows.
            original = Path(source_paths[-1])
            relocated = args.output / 'test_h265_sdp_portable.cc'
            relocated.write_text(original.read_text().replace('/tmp/', args.output.resolve().as_posix() + '/'), encoding='utf-8')
            source_paths[-1] = str(relocated.resolve())
        compile_command = ([str(args.zig.resolve()), 'c++'] if args.zig else ['g++']) + [
            '-std=c++11', '-Wall', '-Wextra', '-Werror', '-pthread', '-I' + str(root / 'cpp')
        ] + source_paths + ['-o', str(binary)]
        compiled = subprocess.run(compile_command, capture_output=True, text=True)
        (args.output / (name + '.build.log')).write_text(compiled.stdout + compiled.stderr, encoding='utf-8')
        result = dict(name=name, compile_command=compile_command, compile_returncode=compiled.returncode)
        if compiled.returncode:
            result['status'] = 'FAIL'
        else:
            test = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            result.update(status='PASS' if test.returncode == 0 else 'FAIL', test_returncode=test.returncode,
                          output=test.stdout + test.stderr)
        results.append(result)
        print(name, result['status'], flush=True)
    report = args.output / 'results.json'
    if args.only and report.exists():
        results = [item for item in json.loads(report.read_text()) if item['name'] != args.only] + results
    report.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return int(any(item['status'] == 'FAIL' for item in results))


if __name__ == '__main__':
    raise SystemExit(main())

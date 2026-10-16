#!/usr/bin/env python3
"""Keep feasible same-reference Pareto candidates, never rank missing quality."""
import argparse
import json
import math
from pathlib import Path

from run_video_quality_sweep import OPTIONAL_SELECTED_FIELDS


def pareto(records):
    candidates = [item for item in records if item['status'] == 'PASS' and item.get('strict_same_source')
                  and item.get('quality') and item['quality'].get('reference_id')
                  and item['quality'].get('input_checksum_verified') is not False
                  and all(isinstance(item['quality'].get(key), (int, float)) and
                          math.isfinite(item['quality'][key]) for key in ('ssim', 'temporal_variation'))]
    def dominates(left, right):
        if (left['quality']['reference_id'], left['scene'], left['cap']) != (
                right['quality']['reference_id'], right['scene'], right['cap']):
            return False
        l = (left['quality']['ssim'], -left['quality']['temporal_variation'],
             left['acceptance'].get('unique_presented_fps',
                                    left['acceptance']['unique_presented_gan_fps']))
        r = (right['quality']['ssim'], -right['quality']['temporal_variation'],
             right['acceptance'].get('unique_presented_fps',
                                     right['acceptance']['unique_presented_gan_fps']))
        # Preserve face/text-region and latency tradeoffs when measured. Missing
        # dimensions cannot be treated as worse than genuine measurements.
        for a, b, sign in ((left['quality'].get('roi_ssim'), right['quality'].get('roi_ssim'), 1),
                     (left['acceptance'].get('end_to_end_ms', {}).get('p95'),
                      right['acceptance'].get('end_to_end_ms', {}).get('p95'), -1)):
            valid_a = isinstance(a, (int, float)) and math.isfinite(a)
            valid_b = isinstance(b, (int, float)) and math.isfinite(b)
            if valid_a != valid_b:
                return False
            if valid_a:
                l += (sign * a,)
                r += (sign * b,)
        return all(a >= b for a, b in zip(l, r)) and any(a > b for a, b in zip(l, r))
    return [item for item in candidates if not any(dominates(other, item) for other in candidates)]


def shared_scene_pareto(records, scenes):
    """Require every requested scene and retain each quality/FPS tradeoff."""
    fields = ('cap', 'width', 'height', 'fps', 'target', 'gop_seconds',
              'qp_min', 'qp_max', 'enhancer', 'adaptive') + OPTIONAL_SELECTED_FIELDS
    groups = {}
    for item in records:
        scene = item.get('scene')
        quality, acceptance = item.get('quality') or {}, item.get('acceptance') or {}
        if (scene not in scenes or item.get('status') != 'PASS' or
                not item.get('strict_same_source') or not quality.get('reference_id') or
                quality.get('input_checksum_verified') is not True):
            continue
        values = (quality.get('ssim'), quality.get('roi_ssim'), quality.get('temporal_variation'),
                  acceptance.get('unique_presented_fps', acceptance.get('unique_presented_gan_fps')),
                  acceptance.get('end_to_end_ms', {}).get('p95'))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            continue
        config = {field: item[field] for field in fields if field in item}
        config.update(legacy=bool(item.get('legacy')), display=bool(item.get('display')))
        context = (json.dumps(acceptance.get('hardware'), sort_keys=True),
                   tuple(item.get('provider', [])), tuple(item.get('precision', [])))
        key = (json.dumps(config, sort_keys=True), context)
        group = groups.setdefault(key, dict(config, scene_results={}))
        if scene in group['scene_results']:
            raise ValueError('duplicate scene/configuration; separate repeated measurement sets explicitly')
        group['scene_results'][scene] = item
    complete = [group for group in groups.values() if set(group['scene_results']) == set(scenes)]
    for group in complete:
        group['joint_scenes'] = list(scenes)
        group['reference_ids'] = {scene: group['scene_results'][scene]['quality']['reference_id']
                                  for scene in scenes}
    def vector(group):
        values = []
        for scene in scenes:
            item = group['scene_results'][scene]
            quality, acceptance = item['quality'], item['acceptance']
            values.extend((quality['ssim'], quality['roi_ssim'], -quality['temporal_variation'],
                acceptance.get('unique_presented_fps', acceptance['unique_presented_gan_fps']),
                -acceptance['end_to_end_ms']['p95']))
        return values
    def dominates(left, right):
        if (left['cap'], left['reference_ids']) != (right['cap'], right['reference_ids']):
            return False
        # Different actual devices/providers/precision are separate comparisons.
        for scene in scenes:
            a, b = left['scene_results'][scene], right['scene_results'][scene]
            if (a['acceptance'].get('hardware'), a.get('provider'), a.get('precision')) != (
                    b['acceptance'].get('hardware'), b.get('provider'), b.get('precision')):
                return False
        a, b = vector(left), vector(right)
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
    return [group for group in complete if not any(dominates(other, group) for other in complete)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+',
                        help='require all listed scenes for each configuration; retain joint Pareto set')
    args = parser.parse_args()
    records = json.loads(args.summary.read_text(encoding='utf-8'))
    if args.scenes and (len(args.scenes) < 2 or len(set(args.scenes)) != len(args.scenes)):
        parser.error('--scenes needs at least two distinct scene names')
    chosen = shared_scene_pareto(records, args.scenes) if args.scenes else pareto(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding='utf-8')
    for cap in (60, 120, 180):
        count = sum(item['cap'] == cap for item in chosen)
        print(f'CAP{cap}: {count} measured Pareto candidates' if count else f'CAP{cap}: NOT RUN / no feasible measured quality candidate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

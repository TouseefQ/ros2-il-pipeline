"""
dataset_stats.py — print summary statistics for all episodes in a directory.

Usage:
    python tools/dataset_stats.py data/episodes/
    python tools/dataset_stats.py data/episodes/ --json

Output (text):
    Episodes    : 42  (valid: 42, corrupt: 0)
    Total steps : 8 413
    Avg / Min / Max : 200.3 / 87 / 498 steps
    Modes       : teleop=40  kinesthetic=2
    Images      : no
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def _read_meta(path: Path) -> dict:
    with h5py.File(path, 'r') as f:
        if 'meta' not in f or 'data' not in f:
            return {
                'path': str(path), 'episode_id': path.stem,
                'valid': False, 'error': 'missing meta or data group',
                'num_steps': 0, 'collection_mode': 'unknown', 'record_images': False,
            }

        meta = f['meta']

        def _s(key: str) -> str:
            v = meta[key][()]
            return v.decode() if isinstance(v, bytes) else str(v)

        num_steps = int(meta['num_steps'][()])
        duration = None
        if 'timestamp_start' in meta and 'timestamp_end' in meta:
            duration = float(meta['timestamp_end'][()] - meta['timestamp_start'][()])

        return {
            'path': str(path),
            'episode_id': _s('episode_id') if 'episode_id' in meta else path.stem,
            'num_steps': num_steps,
            'collection_mode': _s('collection_mode') if 'collection_mode' in meta else 'unknown',
            'record_images': bool(meta['record_images'][()]) if 'record_images' in meta else False,
            'duration_s': duration,
            'valid': num_steps > 0,
            'error': None if num_steps > 0 else 'empty episode (0 steps)',
        }


def scan(episodes_dir: Path) -> list[dict]:
    paths = sorted(episodes_dir.glob('*.hdf5'))
    results = []
    for p in paths:
        try:
            results.append(_read_meta(p))
        except Exception as exc:
            results.append({
                'path': str(p), 'episode_id': p.stem,
                'num_steps': 0, 'collection_mode': 'unknown',
                'record_images': False, 'valid': False, 'error': str(exc),
            })
    return results


def compute_stats(episodes: list[dict]) -> dict:
    valid = [e for e in episodes if e.get('valid', False)]
    corrupt = [e for e in episodes if not e.get('valid', False)]
    steps = [e['num_steps'] for e in valid]

    modes: dict[str, int] = {}
    for e in valid:
        m = e.get('collection_mode', 'unknown')
        modes[m] = modes.get(m, 0) + 1

    durations = [e['duration_s'] for e in valid if e.get('duration_s') is not None]

    return {
        'total_episodes': len(episodes),
        'valid_episodes': len(valid),
        'corrupt_episodes': len(corrupt),
        'total_steps': sum(steps),
        'avg_steps': round(sum(steps) / len(steps), 1) if steps else 0,
        'min_steps': min(steps) if steps else 0,
        'max_steps': max(steps) if steps else 0,
        'avg_duration_s': round(sum(durations) / len(durations), 1) if durations else None,
        'collection_modes': modes,
        'images_recorded': any(e.get('record_images', False) for e in valid),
        'corrupt': [{'path': e['path'], 'error': e.get('error')} for e in corrupt],
    }


def print_report(stats: dict) -> None:
    n, v, c = stats['total_episodes'], stats['valid_episodes'], stats['corrupt_episodes']
    ep_label = f"{n}  (valid: {v}, corrupt: {c})" if c else str(n)

    print(f"Episodes    : {ep_label}")
    print(f"Total steps : {stats['total_steps']:,}")

    if stats['valid_episodes']:
        print(f"Avg / Min / Max : {stats['avg_steps']} / {stats['min_steps']} / {stats['max_steps']} steps")
        if stats['avg_duration_s'] is not None:
            print(f"Avg duration : {stats['avg_duration_s']} s")

    modes_str = '  '.join(f"{k}={v}" for k, v in stats['collection_modes'].items())
    print(f"Modes       : {modes_str or 'n/a'}")
    print(f"Images      : {'yes' if stats['images_recorded'] else 'no'}")

    if stats['corrupt']:
        print('\nCorrupt / empty episodes:')
        for ep in stats['corrupt']:
            print(f"  {ep['path']}  —  {ep.get('error', 'unknown error')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Dataset statistics')
    p.add_argument('episodes_dir', help='Directory containing .hdf5 episode files')
    p.add_argument('--json', action='store_true', help='Output as JSON')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _H5PY_AVAILABLE:
        print('Error: h5py not installed — pip install h5py', file=sys.stderr)
        sys.exit(1)

    d = Path(args.episodes_dir)
    if not d.is_dir():
        print(f'Not a directory: {d}', file=sys.stderr)
        sys.exit(1)

    episodes = scan(d)
    if not episodes:
        print(f'No .hdf5 files found in {d}')
        return

    stats = compute_stats(episodes)

    if getattr(args, 'json'):
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)


if __name__ == '__main__':
    main()

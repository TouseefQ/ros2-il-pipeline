"""
export_to_lerobot.py — convert local HDF5 episodes to HuggingFace LeRobot v2 format.

LeRobot v2 uses Parquet-based episode files with companion JSON metadata.
The exported dataset can be loaded directly with the LeRobot training framework
or uploaded to the HuggingFace Hub.

Output layout:
    <output_dir>/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── episode_000001.parquet ...
    └── meta/
        ├── info.json         — dataset-level metadata and feature schema
        ├── episodes.jsonl    — one JSON object per episode (index, task, length)
        └── stats.json        — per-feature min/max/mean/std for normalisation

Parquet columns per episode:
    observation.state   list[float32]   joint positions  (length N_joints)
    observation.eef_pos list[float32]   end-effector xyz (length 3)
    action              list[float32]   joint_pos_abs    (length N_joints)
    timestamp           float32         seconds from episode start
    episode_index       int64
    frame_index         int64
    next.done           bool            True only on the last frame
    index               int64           global frame index across all episodes

Usage:
    python tools/export_to_lerobot.py \\
        --episodes_dir data/episodes/ \\
        --output_dir   data/lerobot_export/ \\
        --robot_type   panda \\
        --task         "pick and place" \\
        --fps          10.0 \\
        --joint_names  panda_joint1 panda_joint2 panda_joint3 \\
                       panda_joint4 panda_joint5 panda_joint6 panda_joint7
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

try:
    import numpy as np
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

CHUNKS_SIZE = 1000  # episodes per chunk folder


# ── HDF5 reading ──────────────────────────────────────────────────────────────

def _decode(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def read_episode(path: Path) -> dict | None:
    try:
        with h5py.File(path, 'r') as f:
            if 'meta' not in f or 'data' not in f:
                return None
            meta = f['meta']
            obs = f['data/observation']
            act = f['data/action']
            return {
                'episode_id': _decode(meta['episode_id'][()]) if 'episode_id' in meta else path.stem,
                'collection_mode': _decode(meta['collection_mode'][()]) if 'collection_mode' in meta else 'unknown',
                'num_steps': int(meta['num_steps'][()]),
                'timestamps': obs['timestamps'][()],
                'joint_pos': obs['joint_pos'][()],    # (T, N)  float32
                'eef_pos': obs['eef_pos'][()],         # (T, 3)  float32
                'gripper_state': obs['gripper_state'][()].squeeze(),  # (T,)
                'joint_pos_abs': act['joint_pos_abs'][()],  # (T, N)  float32
            }
    except Exception as exc:
        print(f'  Warning: skipping {path.name} — {exc}', file=sys.stderr)
        return None


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _feature_stats(arrays: list) -> dict:
    """Aggregate min/max/mean/std across all episodes for one feature."""
    combined = np.concatenate(arrays, axis=0)  # (total_T, N) or (total_T,)
    if combined.ndim == 1:
        combined = combined[:, None]
    return {
        'min': combined.min(axis=0).tolist(),
        'max': combined.max(axis=0).tolist(),
        'mean': combined.mean(axis=0).tolist(),
        'std': combined.std(axis=0).tolist(),
    }


# ── Export ────────────────────────────────────────────────────────────────────

def export(
    episodes_dir: Path,
    output_dir: Path,
    robot_type: str,
    task: str,
    fps: float,
    joint_names: list[str] | None,
) -> None:
    paths = sorted(episodes_dir.glob('*.hdf5'))
    if not paths:
        print(f'No .hdf5 files found in {episodes_dir}')
        return

    print(f'Found {len(paths)} file(s). Reading...')
    episodes = [ep for p in paths if (ep := read_episode(p)) and ep['num_steps'] > 0]

    if not episodes:
        print('No valid episodes to export.')
        return

    n_joints = episodes[0]['joint_pos'].shape[1]
    if joint_names is None:
        joint_names = [f'joint{i+1}' for i in range(n_joints)]
    elif len(joint_names) != n_joints:
        print(
            f'Warning: {len(joint_names)} joint names given but episodes have {n_joints} joints. '
            'Using auto-generated names.',
            file=sys.stderr,
        )
        joint_names = [f'joint{i+1}' for i in range(n_joints)]

    total_frames = sum(e['num_steps'] for e in episodes)
    n_chunks = max(1, (len(episodes) + CHUNKS_SIZE - 1) // CHUNKS_SIZE)

    print(f'Exporting {len(episodes)} episode(s) → {output_dir} ...')

    ep_meta_lines: list[dict] = []
    global_frame = 0

    for ep_idx, ep in enumerate(episodes):
        chunk_idx = ep_idx // CHUNKS_SIZE
        chunk_dir = output_dir / 'data' / f'chunk-{chunk_idx:03d}'
        chunk_dir.mkdir(parents=True, exist_ok=True)

        T = ep['num_steps']
        t_rel = (ep['timestamps'] - ep['timestamps'][0]).astype(np.float32)

        df = pd.DataFrame({
            'observation.state': [ep['joint_pos'][i].tolist() for i in range(T)],
            'observation.eef_pos': [ep['eef_pos'][i].tolist() for i in range(T)],
            'action': [ep['joint_pos_abs'][i].tolist() for i in range(T)],
            'timestamp': t_rel.tolist(),
            'episode_index': np.full(T, ep_idx, dtype=np.int64).tolist(),
            'frame_index': list(range(T)),
            'next.done': [False] * (T - 1) + [True],
            'index': list(range(global_frame, global_frame + T)),
        })

        parquet_path = chunk_dir / f'episode_{ep_idx:06d}.parquet'
        df.to_parquet(parquet_path, index=False)

        ep_meta_lines.append({
            'episode_index': ep_idx,
            'tasks': [task],
            'length': T,
        })
        global_frame += T

        print(f'  [{ep_idx+1}/{len(episodes)}] {ep["episode_id"]:30s} '
              f'→ {parquet_path.relative_to(output_dir)}  ({T} frames)')

    # ── meta/info.json ────────────────────────────────────────────────────────
    meta_dir = output_dir / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        'codebase_version': 'v2.0',
        'robot_type': robot_type,
        'total_episodes': len(episodes),
        'total_frames': total_frames,
        'total_tasks': 1,
        'total_videos': 0,
        'total_chunks': n_chunks,
        'chunks_size': CHUNKS_SIZE,
        'fps': fps,
        'splits': {'train': f'0:{len(episodes)}'},
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'features': {
            'observation.state': {
                'dtype': 'float32', 'shape': [n_joints], 'names': joint_names,
            },
            'observation.eef_pos': {
                'dtype': 'float32', 'shape': [3], 'names': ['x', 'y', 'z'],
            },
            'action': {
                'dtype': 'float32', 'shape': [n_joints], 'names': joint_names,
            },
            'timestamp':     {'dtype': 'float32', 'shape': [1], 'names': None},
            'episode_index': {'dtype': 'int64',   'shape': [1], 'names': None},
            'frame_index':   {'dtype': 'int64',   'shape': [1], 'names': None},
            'next.done':     {'dtype': 'bool',    'shape': [1], 'names': None},
            'index':         {'dtype': 'int64',   'shape': [1], 'names': None},
        },
    }
    (meta_dir / 'info.json').write_text(json.dumps(info, indent=2))

    # ── meta/episodes.jsonl ───────────────────────────────────────────────────
    with open(meta_dir / 'episodes.jsonl', 'w') as fh:
        for line in ep_meta_lines:
            fh.write(json.dumps(line) + '\n')

    # ── meta/stats.json ───────────────────────────────────────────────────────
    stats = {
        'observation.state': _feature_stats([e['joint_pos'] for e in episodes]),
        'observation.eef_pos': _feature_stats([e['eef_pos'] for e in episodes]),
        'action': _feature_stats([e['joint_pos_abs'] for e in episodes]),
    }
    (meta_dir / 'stats.json').write_text(json.dumps(stats, indent=2))

    print('\nDone.')
    print(f'  episodes : {len(episodes)}')
    print(f'  frames   : {total_frames:,}')
    print(f'  chunks   : {n_chunks}')
    print(f'  output   : {output_dir.resolve()}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Export HDF5 episodes to LeRobot v2 format')
    p.add_argument('--episodes_dir', required=True,
                   help='Directory containing .hdf5 episode files')
    p.add_argument('--output_dir', required=True,
                   help='Output directory for the LeRobot dataset')
    p.add_argument('--robot_type', default='panda',
                   help='Robot type string written to meta/info.json (default: panda)')
    p.add_argument('--task', default='manipulation',
                   help='Task description written to meta/episodes.jsonl (default: manipulation)')
    p.add_argument('--fps', type=float, default=10.0,
                   help='Recording FPS used during collection (default: 10.0)')
    p.add_argument('--joint_names', nargs='+',
                   help='Joint names in order (default: joint1...jointN)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _H5PY_AVAILABLE:
        print('Error: h5py not installed — pip install h5py', file=sys.stderr)
        sys.exit(1)
    if not _PANDAS_AVAILABLE:
        print('Error: pandas/numpy not installed — pip install pandas pyarrow numpy',
              file=sys.stderr)
        sys.exit(1)

    episodes_dir = Path(args.episodes_dir)
    output_dir = Path(args.output_dir)

    if not episodes_dir.is_dir():
        print(f'Not a directory: {episodes_dir}', file=sys.stderr)
        sys.exit(1)

    export(
        episodes_dir=episodes_dir,
        output_dir=output_dir,
        robot_type=args.robot_type,
        task=args.task,
        fps=args.fps,
        joint_names=args.joint_names,
    )


if __name__ == '__main__':
    main()

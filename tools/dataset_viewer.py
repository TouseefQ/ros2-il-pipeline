"""
dataset_viewer.py — visualize collected episodes from the HDF5 store.

Usage:
    python tools/dataset_viewer.py data/episodes/ep_000.hdf5
    python tools/dataset_viewer.py data/episodes/ep_000.hdf5 --show_images

Plots joint trajectories, gripper state, end-effector path, and
optionally plays back camera frames side by side.
"""

import argparse
import sys
from pathlib import Path

# TODO Stage 4: implement with h5py + matplotlib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize an episode HDF5 file')
    p.add_argument('episode', help='Path to .hdf5 episode file')
    p.add_argument('--show_images', action='store_true',
                   help='Play back camera frames (requires record_images=true)')
    p.add_argument('--fps', type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.episode)
    if not path.exists():
        print(f'File not found: {path}', file=sys.stderr)
        sys.exit(1)
    # TODO Stage 4: open HDF5, read data/, plot joint_pos, eef trajectory,
    #               optionally render images as video
    raise NotImplementedError('dataset_viewer.py — implemented in Stage 4')


if __name__ == '__main__':
    main()

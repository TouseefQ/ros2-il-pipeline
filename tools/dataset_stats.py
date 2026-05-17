"""
dataset_stats.py — print summary statistics for all episodes in a directory.

Usage:
    python tools/dataset_stats.py data/episodes/
    python tools/dataset_stats.py data/episodes/ --json

Output (text):
    Episodes : 42
    Total steps : 8 413
    Avg length  : 200.3 steps
    Min / Max   : 87 / 498 steps
    Collection modes: teleop=40, kinesthetic=2
    Images recorded : no
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Dataset statistics')
    p.add_argument('episodes_dir', help='Directory containing .hdf5 episode files')
    p.add_argument('--json', action='store_true', help='Output as JSON')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO Stage 4: scan dir, open each HDF5, aggregate stats, print/dump
    raise NotImplementedError('dataset_stats.py — implemented in Stage 4')


if __name__ == '__main__':
    main()

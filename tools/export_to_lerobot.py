"""
export_to_lerobot.py — convert local HDF5 episodes to HuggingFace LeRobot format.

LeRobot uses a Parquet-based format with a companion meta/info.json.
This tool enables uploading datasets to the HuggingFace Hub for
sharing, benchmarking, and use with the LeRobot training framework.

Usage:
    python tools/export_to_lerobot.py \\
        --episodes_dir data/episodes/ \\
        --output_dir data/lerobot_export/ \\
        --robot_type panda \\
        --task "pick and place"
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Export to LeRobot format')
    p.add_argument('--episodes_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--robot_type', default='panda')
    p.add_argument('--task', default='manipulation')
    p.add_argument('--fps', type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO Stage 4: read HDF5 episodes, write Parquet shards + meta/info.json
    #               matching LeRobot v2 spec
    raise NotImplementedError('export_to_lerobot.py — implemented in Stage 4')


if __name__ == '__main__':
    main()

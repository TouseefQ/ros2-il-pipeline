"""
train_bc.py — Behavior Cloning training entry point.

Usage:
    python train_bc.py                              # uses configs/bc_config.yaml
    python train_bc.py --config configs/bc_config.yaml
    python train_bc.py --config configs/bc_config.yaml --episodes_dir /data/episodes
"""

import argparse
from pathlib import Path

# TODO Stage 5: import DemonstrationDataset, MLPPolicy, training loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train BC policy')
    p.add_argument('--config', default='configs/bc_config.yaml')
    p.add_argument('--episodes_dir', default=None,
                   help='Override episodes_dir from config')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Override checkpoint_dir from config')
    p.add_argument('--device', default='auto',
                   help='cpu | cuda | auto (auto picks cuda if available)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO Stage 5: load config, build dataset/dataloaders, build model,
    #               run training loop, save checkpoints
    raise NotImplementedError('train_bc.py — implemented in Stage 5')


if __name__ == '__main__':
    main()

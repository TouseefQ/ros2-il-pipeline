"""
train_act.py — ACT (Action Chunking Transformer) training entry point.

Usage:
    python train_act.py                               # uses configs/act_config.yaml
    python train_act.py --config configs/act_config.yaml
"""

import argparse
from pathlib import Path

# TODO Stage 7: import DemonstrationDataset, ACTPolicy, training loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train ACT policy')
    p.add_argument('--config', default='configs/act_config.yaml')
    p.add_argument('--episodes_dir', default=None)
    p.add_argument('--checkpoint_dir', default=None)
    p.add_argument('--device', default='auto')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO Stage 7: load config, build dataset/dataloaders, build ACTPolicy,
    #               run training loop with KL + reconstruction loss, save checkpoints
    raise NotImplementedError('train_act.py — implemented in Stage 7')


if __name__ == '__main__':
    main()

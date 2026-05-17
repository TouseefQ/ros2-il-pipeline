"""
evaluate.py — evaluate a trained policy checkpoint on held-out episodes.

Usage:
    python evaluate.py --checkpoint ../data/models/bc_v1.pt --algorithm bc
    python evaluate.py --checkpoint ../data/models/act_v1.pt --algorithm act \\
                       --episodes_dir ../data/episodes/eval
"""

import argparse

# TODO Stage 6+: import PolicyLoader, DemonstrationDataset, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate IL policy')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--algorithm', default='bc', choices=['bc', 'act', 'diffusion'])
    p.add_argument('--episodes_dir', default='../data/episodes')
    p.add_argument('--device', default='auto')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO Stage 6+: load policy, run on each eval episode,
    #                report MSE, action error, per-joint breakdown
    raise NotImplementedError('evaluate.py — implemented in Stage 6+')


if __name__ == '__main__':
    main()

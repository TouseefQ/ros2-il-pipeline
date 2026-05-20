"""
PolicyLoader — loads a trained BC or ACT checkpoint and exposes a
predict() interface compatible with the inference node.

Usage:
    loader = PolicyLoader.from_checkpoint(
        '/data/models/bc_v1.pt',
        training_src_dir='/path/to/ros2-il-pipeline/training',
    )
    action = loader.predict(obs_dict)   # obs_dict keys: joint_pos, joint_vel, eef_pos, eef_quat
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class PolicyConfig:
    algorithm: str           # "bc" | "act"
    num_joints: int
    uses_images: bool
    action_chunk_size: int
    checkpoint_path: str


def _build_bc_model(config: dict):
    """Reconstruct a BC model architecture from the training config saved in the checkpoint."""
    from models.bc_policy import MLPPolicy, CNNMLPPolicy  # noqa: PLC0415

    mcfg = config['model']
    num_joints  = mcfg.get('joint_input_dim', 7)
    hidden_dims = mcfg.get('hidden_dims', [256, 256, 256])
    dropout     = mcfg.get('dropout', 0.1)
    model_type  = mcfg.get('type', 'mlp')

    if model_type == 'mlp':
        return MLPPolicy(num_joints=num_joints, hidden_dims=hidden_dims, dropout=dropout)
    elif model_type == 'cnn_mlp':
        num_cameras = 1 + int(config.get('data', {}).get('record_images', False))
        return CNNMLPPolicy(
            num_joints=num_joints,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_cameras=num_cameras,
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')


class PolicyLoader:
    """Abstract base — subclassed by BCPolicyLoader and ACTPolicyLoader."""

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        algorithm: str = '',
        training_src_dir: str = '',
    ) -> 'PolicyLoader':
        """
        Load a checkpoint and return the appropriate PolicyLoader subclass.

        Args:
            path: Path to .pt checkpoint produced by train_bc.py or train_act.py.
            algorithm: Override algorithm detection (leave empty to read from checkpoint).
            training_src_dir: Absolute path to the training/ directory so that
                              models.bc_policy and datasets.dataset_loader can be imported.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch is required — pip install torch')

        if training_src_dir:
            p = str(Path(training_src_dir).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        algo = algorithm or ckpt.get('algorithm', '')

        if not algo:
            raise ValueError(
                f'Cannot determine algorithm from checkpoint (no "algorithm" key): {path}'
            )

        if algo == 'bc':
            return BCPolicyLoader._from_ckpt(ckpt, path)
        elif algo == 'act':
            return ACTPolicyLoader._from_ckpt(ckpt, path)
        else:
            raise ValueError(f'Unsupported algorithm: {algo!r}')

    def predict(self, obs: dict) -> np.ndarray:
        """
        Args:
            obs: dict with keys matching training observation format.
                 Required keys: joint_pos, joint_vel, eef_pos, eef_quat (all np.ndarray).
                 Optional keys: camera_wrist, camera_top (np.ndarray uint8 H×W×3).
        Returns:
            action: np.ndarray of shape (num_joints,)
        """
        raise NotImplementedError

    @property
    def config(self) -> PolicyConfig:
        return self._config


class BCPolicyLoader(PolicyLoader):
    """Wraps a trained MLPPolicy or CNNMLPPolicy checkpoint."""

    def __init__(self, config: PolicyConfig, model, normalizer) -> None:
        super().__init__(config)
        self._model = model
        self._normalizer = normalizer

    @classmethod
    def _from_ckpt(cls, ckpt: dict, path: str) -> 'BCPolicyLoader':
        from datasets.dataset_loader import Normalizer  # noqa: PLC0415

        config     = ckpt['config']
        num_joints = ckpt.get('num_joints', config['model'].get('joint_input_dim', 7))
        model_type = config['model'].get('type', 'mlp')

        normalizer = Normalizer.from_state_dict(ckpt['normalizer'])
        model = _build_bc_model(config)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        pcfg = PolicyConfig(
            algorithm='bc',
            num_joints=num_joints,
            uses_images=(model_type == 'cnn_mlp'),
            action_chunk_size=1,
            checkpoint_path=str(path),
        )
        return cls(pcfg, model, normalizer)

    def predict(self, obs: dict) -> np.ndarray:
        return self._model.predict(obs, self._normalizer)


class ACTPolicyLoader(PolicyLoader):
    """Placeholder — ACT inference implemented in Stage 7."""

    @classmethod
    def _from_ckpt(cls, ckpt: dict, path: str) -> 'ACTPolicyLoader':
        raise NotImplementedError('ACT inference — Stage 7')

    def predict(self, obs: dict) -> np.ndarray:
        raise NotImplementedError('ACT inference — Stage 7')

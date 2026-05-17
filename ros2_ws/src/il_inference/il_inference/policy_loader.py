"""
PolicyLoader — loads a trained BC or ACT checkpoint and exposes a
predict() interface compatible with the inference node.

Usage (Stage 6):
    loader = PolicyLoader.from_checkpoint('/data/models/bc_v1.pt', algorithm='bc')
    action = loader.predict(obs_dict)
"""

from __future__ import annotations

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
    algorithm: str           # "bc" | "act" | "diffusion"
    num_joints: int
    uses_images: bool
    action_chunk_size: int   # relevant for ACT
    checkpoint_path: str


class PolicyLoader:
    """Abstract base — subclassed by BCPolicyLoader and ACTPolicyLoader."""

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config
        self._model = None

    @classmethod
    def from_checkpoint(cls, path: str, algorithm: str) -> 'PolicyLoader':
        # TODO Stage 6: detect algorithm from checkpoint metadata,
        #               instantiate correct subclass, load weights
        raise NotImplementedError('PolicyLoader.from_checkpoint — Stage 6')

    def predict(self, obs: dict) -> np.ndarray:
        """
        Args:
            obs: dict with keys matching training observation format
                 e.g. {'joint_pos': ndarray, 'eef_pos': ndarray, ...}
        Returns:
            action: ndarray of shape (num_joints,) or (chunk, num_joints)
        """
        # TODO Stage 6: implement forward pass
        raise NotImplementedError('PolicyLoader.predict — Stage 6')

    @property
    def config(self) -> PolicyConfig:
        return self._config

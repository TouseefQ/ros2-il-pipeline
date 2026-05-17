"""
DemonstrationDataset — PyTorch Dataset that reads HDF5 episode files.

Supports both state-only and state+image observations.
Produces (observation, action) pairs for BC training,
and (observation_sequence, action_chunk) pairs for ACT training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


class DemonstrationDataset:
    """
    Loads all .hdf5 episode files from episodes_dir.

    Each sample is a dict:
      {
        'joint_pos':    (N,)      float32
        'joint_vel':    (N,)      float32
        'eef_pos':      (3,)      float32
        'eef_quat':     (4,)      float32
        'gripper':      (1,)      float32
        'action':       (N,)      float32   joint_pos_delta (BC) or abs (ACT)
        'camera_wrist': (H,W,3)   uint8     optional
        'camera_top':   (H,W,3)   uint8     optional
      }
    """

    def __init__(
        self,
        episodes_dir: str,
        record_images: bool = False,
        chunk_size: int = 1,          # 1 = BC, >1 = ACT
        action_key: str = 'joint_pos_delta',
        transform: Callable | None = None,
    ) -> None:
        self._episodes_dir = Path(episodes_dir)
        self._record_images = record_images
        self._chunk_size = chunk_size
        self._action_key = action_key
        self._transform = transform

        self._index: list[tuple[str, int]] = []  # (hdf5_path, step_idx)
        self._build_index()

    def _build_index(self) -> None:
        # TODO Stage 5: scan episodes_dir, open each HDF5, collect valid step indices
        raise NotImplementedError('DemonstrationDataset._build_index — Stage 5')

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        # TODO Stage 5: read step from HDF5, return obs+action dict
        raise NotImplementedError('DemonstrationDataset.__getitem__ — Stage 5')

    @property
    def num_episodes(self) -> int:
        # TODO Stage 5
        return 0

    @property
    def total_steps(self) -> int:
        return len(self._index)


def make_dataloaders(
    episodes_dir: str,
    val_split: float = 0.1,
    batch_size: int = 32,
    record_images: bool = False,
    chunk_size: int = 1,
    action_key: str = 'joint_pos_delta',
    num_workers: int = 4,
):
    """Split dataset into train/val and return DataLoader pair."""
    # TODO Stage 5: implement split + DataLoader creation
    raise NotImplementedError('make_dataloaders — Stage 5')

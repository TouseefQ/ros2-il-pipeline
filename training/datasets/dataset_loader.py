"""
DemonstrationDataset — PyTorch Dataset that reads HDF5 episode files.

Supports both state-only and state+image observations.
Produces (observation, action) pairs for BC training,
and (observation_sequence, action_chunk) pairs for ACT training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """Per-feature z-score normalizer (mean/std).  Handles zero-std gracefully."""

    def __init__(self, stats: dict[str, dict]) -> None:
        # stats: { key: {'mean': np.array, 'std': np.array} }
        self._mean: dict[str, np.ndarray] = {}
        self._std:  dict[str, np.ndarray] = {}
        for key, s in stats.items():
            self._mean[key] = np.asarray(s['mean'], dtype=np.float32)
            self._std[key]  = np.asarray(s['std'],  dtype=np.float32)
            # avoid division by zero for constant features
            self._std[key]  = np.where(self._std[key] < 1e-8, 1.0, self._std[key])

    def normalize(self, key: str, x: np.ndarray) -> np.ndarray:
        return (x - self._mean[key]) / self._std[key]

    def denormalize(self, key: str, x: np.ndarray) -> np.ndarray:
        return x * self._std[key] + self._mean[key]

    def state_dict(self) -> dict:
        return {
            key: {
                'mean': self._mean[key].tolist(),
                'std':  self._std[key].tolist(),
            }
            for key in self._mean
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> 'Normalizer':
        return cls(d)

    @classmethod
    def from_episodes(cls, episodes: list[dict], keys: list[str]) -> 'Normalizer':
        """Compute mean/std from a list of in-memory episode dicts."""
        stats: dict[str, dict] = {}
        for key in keys:
            arrays = [ep[key] for ep in episodes if key in ep and ep[key] is not None]
            if not arrays:
                stats[key] = {'mean': [0.0], 'std': [1.0]}
                continue
            combined = np.concatenate(arrays, axis=0)  # (total_T, D) or (total_T,)
            if combined.ndim == 1:
                combined = combined[:, None]
            stats[key] = {
                'mean': combined.mean(axis=0).tolist(),
                'std':  combined.std(axis=0).tolist(),
            }
        return cls(stats)

    def as_transform(self) -> Callable[[dict], dict]:
        """Returns a callable that normalizes a numpy sample dict and converts to tensors."""
        def _transform(sample: dict) -> dict:
            out = {}
            for key, val in sample.items():
                if key in self._mean and isinstance(val, np.ndarray):
                    out[key] = torch.from_numpy(self.normalize(key, val))
                elif isinstance(val, np.ndarray):
                    out[key] = torch.from_numpy(val)
                else:
                    out[key] = val
            return out
        return _transform


# ── Episode loading ───────────────────────────────────────────────────────────

def _load_episode(path: Path, record_images: bool, action_key: str) -> dict | None:
    """Load all steps of one episode into memory as numpy arrays."""
    try:
        with h5py.File(path, 'r') as f:
            if 'meta' not in f or 'data' not in f:
                return None
            meta = f['meta']
            obs  = f['data/observation']
            act  = f['data/action']

            num_steps = int(meta['num_steps'][()])
            if num_steps == 0:
                return None

            ep: dict = {
                'path':      str(path),
                'num_steps': num_steps,
                'joint_pos': obs['joint_pos'][()].astype(np.float32),   # (T, N)
                'joint_vel': obs['joint_vel'][()].astype(np.float32),   # (T, N)
                'eef_pos':   obs['eef_pos'][()].astype(np.float32),     # (T, 3)
                'eef_quat':  obs['eef_quat'][()].astype(np.float32),    # (T, 4)
                'gripper':   obs['gripper_state'][()].astype(np.float32).reshape(-1, 1),  # (T, 1)
                'action':    act[action_key][()].astype(np.float32),    # (T, N)
                'camera_wrist': None,
                'camera_top':   None,
            }

            if record_images and 'images' in obs:
                imgs = obs['images']
                if 'camera_wrist' in imgs:
                    ep['camera_wrist'] = imgs['camera_wrist'][()]  # (T, H, W, 3) uint8
                if 'camera_top' in imgs:
                    ep['camera_top'] = imgs['camera_top'][()]

            return ep
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class DemonstrationDataset(Dataset if _TORCH_AVAILABLE else object):
    """
    Loads all .hdf5 episode files from episodes_dir into memory.

    Each sample is a dict with torch tensors:
      joint_pos    (N,)      float32
      joint_vel    (N,)      float32
      eef_pos      (3,)      float32
      eef_quat     (4,)      float32
      gripper      (1,)      float32
      action       (N,)      float32   (chunk_size=1) or (chunk_size, N) (ACT)
      camera_wrist (H,W,3)   uint8     optional
      camera_top   (H,W,3)   uint8     optional
    """

    def __init__(
        self,
        episodes_dir: str,
        record_images: bool = False,
        chunk_size: int = 1,
        action_key: str = 'joint_pos_abs',
        transform: Callable | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch is required — pip install torch')
        if not _H5PY_AVAILABLE:
            raise RuntimeError('h5py is required — pip install h5py')

        self._episodes_dir = Path(episodes_dir)
        self._record_images = record_images
        self._chunk_size = chunk_size
        self._action_key = action_key
        self._transform = transform

        self._episodes: list[dict] = []
        self._index: list[tuple[int, int]] = []  # (episode_idx, step_idx)
        self._build_index()

    def _build_index(self) -> None:
        paths = sorted(self._episodes_dir.glob('*.hdf5'))
        for path in paths:
            ep = _load_episode(path, self._record_images, self._action_key)
            if ep is None:
                continue
            ep_idx = len(self._episodes)
            self._episodes.append(ep)
            # For each valid anchor step (accounting for chunk window)
            T = ep['num_steps']
            for step in range(T):
                self._index.append((ep_idx, step))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, step = self._index[idx]
        ep = self._episodes[ep_idx]
        T = ep['num_steps']

        sample: dict = {
            'joint_pos':  ep['joint_pos'][step].copy(),
            'joint_vel':  ep['joint_vel'][step].copy(),
            'eef_pos':    ep['eef_pos'][step].copy(),
            'eef_quat':   ep['eef_quat'][step].copy(),
            'gripper':    ep['gripper'][step].copy(),
        }

        if self._chunk_size == 1:
            sample['action'] = ep['action'][step].copy()
        else:
            # Collect chunk; pad last steps with final action
            end = min(step + self._chunk_size, T)
            chunk = ep['action'][step:end].copy()
            if chunk.shape[0] < self._chunk_size:
                pad = np.repeat(chunk[-1:], self._chunk_size - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            sample['action'] = chunk  # (chunk_size, N)

        if self._record_images:
            if ep['camera_wrist'] is not None:
                sample['camera_wrist'] = ep['camera_wrist'][step].copy()
            if ep['camera_top'] is not None:
                sample['camera_top'] = ep['camera_top'][step].copy()

        if self._transform is not None:
            return self._transform(sample)

        # Default: convert numpy arrays to tensors
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in sample.items()}

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        return len(self._index)

    def get_episodes(self) -> list[dict]:
        return self._episodes


# ── Dataloaders ───────────────────────────────────────────────────────────────

def make_dataloaders(
    episodes_dir: str,
    val_split: float = 0.1,
    batch_size: int = 32,
    record_images: bool = False,
    chunk_size: int = 1,
    action_key: str = 'joint_pos_abs',
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Episode-level train/val split → (train_loader, val_loader, normalizer).

    Normalizer is fitted on training episodes only and stored with the
    checkpoint so the inference node can denormalize predictions.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError('PyTorch is required — pip install torch')

    # Load all episodes first (no transform yet — we need raw numpy for normalizer)
    full_ds = DemonstrationDataset(
        episodes_dir=episodes_dir,
        record_images=record_images,
        chunk_size=chunk_size,
        action_key=action_key,
        transform=None,
    )

    n_ep = full_ds.num_episodes
    if n_ep == 0:
        raise ValueError(f'No valid episodes found in {episodes_dir}')

    rng = random.Random(seed)
    ep_indices = list(range(n_ep))
    rng.shuffle(ep_indices)

    n_val = max(1, round(n_ep * val_split)) if n_ep > 1 else 0
    val_ep_set  = set(ep_indices[:n_val])
    train_ep_set = set(ep_indices[n_val:])

    # Build index subsets
    train_indices = [i for i, (ep_i, _) in enumerate(full_ds._index) if ep_i in train_ep_set]
    val_indices   = [i for i, (ep_i, _) in enumerate(full_ds._index) if ep_i in val_ep_set]

    # Compute normalizer from training episodes only
    train_episodes = [full_ds.get_episodes()[i] for i in train_ep_set]
    normalizer = Normalizer.from_episodes(
        train_episodes,
        keys=['joint_pos', 'joint_vel', 'eef_pos', 'eef_quat', 'gripper', 'action'],
    )

    # Attach transform (normalize + convert to tensor)
    transform = normalizer.as_transform()
    full_ds._transform = transform

    train_subset = torch.utils.data.Subset(full_ds, train_indices)
    val_subset   = torch.utils.data.Subset(full_ds, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_indices) >= batch_size,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, normalizer

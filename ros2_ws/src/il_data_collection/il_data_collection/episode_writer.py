"""
EpisodeWriter — writes a single demonstration episode to HDF5.

HDF5 schema (per episode file):
  meta/
    episode_id          str
    robot_description   str
    collection_mode     str
    timestamp_start     float64
    timestamp_end       float64
    num_steps           int
    record_images       bool
  data/
    observation/
      joint_pos         (T, N_joints) float32
      joint_vel         (T, N_joints) float32
      eef_pos           (T, 3)        float32   xyz
      eef_quat          (T, 4)        float32   xyzw
      gripper_state     (T, 1)        float32   0.0–1.0
      images/
        camera_wrist    (T, H, W, 3)  uint8     optional
        camera_top      (T, H, W, 3)  uint8     optional
    action/
      joint_pos_delta   (T, N_joints) float32   for BC
      joint_pos_abs     (T, N_joints) float32   for ACT
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

# h5py imported lazily so the package still imports without it
try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


@dataclass
class ObsStep:
    joint_pos: np.ndarray        # (N,)
    joint_vel: np.ndarray        # (N,)
    eef_pos: np.ndarray          # (3,)
    eef_quat: np.ndarray         # (4,)  xyzw
    gripper_state: float
    camera_wrist: np.ndarray | None = None   # (H, W, 3) uint8
    camera_top: np.ndarray | None = None     # (H, W, 3) uint8


@dataclass
class ActionStep:
    joint_pos_abs: np.ndarray    # (N,)  absolute target
    joint_pos_delta: np.ndarray  # (N,)  relative delta


@dataclass
class Step:
    timestamp: float
    obs: ObsStep
    action: ActionStep


class EpisodeWriter:
    """Accumulates steps in memory, then flushes to HDF5 on save()."""

    def __init__(
        self,
        output_dir: str,
        episode_id: str,
        robot_description: str,
        collection_mode: str,
        num_joints: int,
        record_images: bool = False,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self._output_dir = output_dir
        self._episode_id = episode_id
        self._robot_description = robot_description
        self._collection_mode = collection_mode
        self._num_joints = num_joints
        self._record_images = record_images
        self._image_size = image_size
        self._steps: list[Step] = []
        self._t_start = time.time()

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    def add_step(self, step: Step) -> None:
        self._steps.append(step)

    def save(self) -> str:
        """Flush all buffered steps to disk. Returns path of written file."""
        # TODO Stage 2: implement HDF5 write
        raise NotImplementedError('EpisodeWriter.save() — implemented in Stage 2')

    def discard(self) -> None:
        self._steps.clear()

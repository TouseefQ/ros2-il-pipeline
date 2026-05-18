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
      timestamps        (T,)          float64   unix seconds
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
        """Flush all buffered steps to disk. Returns absolute path of written file."""
        if not _H5PY_AVAILABLE:
            raise RuntimeError('h5py is not installed — pip install h5py')
        if not self._steps:
            raise ValueError('No steps recorded — cannot save empty episode')

        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, '%s.hdf5' % self._episode_id)

        T = len(self._steps)
        H, W = self._image_size

        timestamps = np.array([s.timestamp for s in self._steps], dtype=np.float64)
        joint_pos = np.stack([s.obs.joint_pos for s in self._steps]).astype(np.float32)
        joint_vel = np.stack([s.obs.joint_vel for s in self._steps]).astype(np.float32)
        eef_pos = np.stack([s.obs.eef_pos for s in self._steps]).astype(np.float32)
        eef_quat = np.stack([s.obs.eef_quat for s in self._steps]).astype(np.float32)
        gripper_state = np.array(
            [[s.obs.gripper_state] for s in self._steps], dtype=np.float32
        )
        joint_pos_delta = np.stack(
            [s.action.joint_pos_delta for s in self._steps]
        ).astype(np.float32)
        joint_pos_abs = np.stack(
            [s.action.joint_pos_abs for s in self._steps]
        ).astype(np.float32)

        with h5py.File(path, 'w') as f:
            meta = f.create_group('meta')
            meta.create_dataset('episode_id', data=self._episode_id)
            meta.create_dataset('robot_description', data=self._robot_description)
            meta.create_dataset('collection_mode', data=self._collection_mode)
            meta.create_dataset('timestamp_start', data=self._t_start)
            meta.create_dataset('timestamp_end', data=time.time())
            meta.create_dataset('num_steps', data=T)
            meta.create_dataset('record_images', data=self._record_images)

            data = f.create_group('data')
            obs_grp = data.create_group('observation')
            obs_grp.create_dataset('timestamps', data=timestamps)
            obs_grp.create_dataset('joint_pos', data=joint_pos)
            obs_grp.create_dataset('joint_vel', data=joint_vel)
            obs_grp.create_dataset('eef_pos', data=eef_pos)
            obs_grp.create_dataset('eef_quat', data=eef_quat)
            obs_grp.create_dataset('gripper_state', data=gripper_state)

            if self._record_images:
                imgs_grp = obs_grp.create_group('images')
                wrist_frames = [s.obs.camera_wrist for s in self._steps]
                top_frames = [s.obs.camera_top for s in self._steps]
                blank = np.zeros((H, W, 3), dtype=np.uint8)
                if any(f is not None for f in wrist_frames):
                    arr = np.stack([
                        f if f is not None else blank for f in wrist_frames
                    ])
                    imgs_grp.create_dataset(
                        'camera_wrist', data=arr,
                        compression='gzip', compression_opts=4,
                    )
                if any(f is not None for f in top_frames):
                    arr = np.stack([
                        f if f is not None else blank for f in top_frames
                    ])
                    imgs_grp.create_dataset(
                        'camera_top', data=arr,
                        compression='gzip', compression_opts=4,
                    )

            act_grp = data.create_group('action')
            act_grp.create_dataset('joint_pos_delta', data=joint_pos_delta)
            act_grp.create_dataset('joint_pos_abs', data=joint_pos_abs)

        return path

    def discard(self) -> None:
        self._steps.clear()

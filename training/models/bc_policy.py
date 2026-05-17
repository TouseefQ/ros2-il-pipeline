"""
BCPolicy — Behavior Cloning policy networks.

Two variants:
  MLPPolicy     — state-only, lightweight, fast to train
  CNNMLPPolicy  — image + state, uses ResNet-18 visual encoder
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """
    Fully-connected policy: (joint_pos, joint_vel, eef_pos, eef_quat) → joint_delta.

    Input  dim = num_joints*2 + 3 + 4 = num_joints*2 + 7
    Output dim = num_joints  (joint position delta)
    """

    def __init__(
        self,
        num_joints: int = 7,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 256]
        input_dim = num_joints * 2 + 7  # pos + vel + eef_pos + eef_quat
        output_dim = num_joints

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO Stage 5: concatenate obs fields, pass through net
        raise NotImplementedError('MLPPolicy.forward — Stage 5')


class CNNMLPPolicy(nn.Module):
    """
    Visual policy: (image, joint_pos, joint_vel, eef) → joint_delta.
    Uses a frozen ResNet-18 as image encoder.
    """

    def __init__(
        self,
        num_joints: int = 7,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 256]
        # TODO Stage 5: build ResNet-18 encoder + MLP head
        raise NotImplementedError('CNNMLPPolicy.__init__ — Stage 5')

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('CNNMLPPolicy.forward — Stage 5')

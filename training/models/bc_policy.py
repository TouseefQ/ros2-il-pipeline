"""
BCPolicy — Behavior Cloning policy networks.

Two variants:
  MLPPolicy     — state-only, lightweight, fast to train
  CNNMLPPolicy  — image + state, uses ResNet-18 visual encoder
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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
        self.num_joints = num_joints

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            [obs['joint_pos'], obs['joint_vel'], obs['eef_pos'], obs['eef_quat']],
            dim=-1,
        )
        return self.net(x)

    def predict(self, obs: dict[str, np.ndarray], normalizer=None) -> np.ndarray:
        """Numpy in → numpy out (single-step inference)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            tensor_obs = {}
            for key in ('joint_pos', 'joint_vel', 'eef_pos', 'eef_quat'):
                arr = obs[key].astype(np.float32)
                if normalizer is not None:
                    arr = normalizer.normalize(key, arr)
                tensor_obs[key] = torch.from_numpy(arr).unsqueeze(0).to(device)
            action = self.forward(tensor_obs).squeeze(0).cpu().numpy()
        if normalizer is not None:
            action = normalizer.denormalize('action', action)
        return action


class CNNMLPPolicy(nn.Module):
    """
    Visual policy: (image, joint_pos, joint_vel, eef) → joint_delta.
    Uses a pretrained ResNet-18 as image encoder (frozen backbone).
    """

    # ImageNet normalisation constants
    _IMGNET_MEAN = [0.485, 0.456, 0.406]
    _IMGNET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        num_joints: int = 7,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 256]
        self.num_joints = num_joints
        self.num_cameras = num_cameras

        # ResNet-18 visual encoder (pretrained, backbone frozen)
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=True)  # fallback for older torchvision
        backbone.fc = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        visual_dim = 512 * num_cameras  # 512 features per camera

        # State embedding
        state_dim = num_joints * 2 + 7  # pos + vel + eef_pos + eef_quat
        self.state_embed = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())

        # Fusion MLP
        fusion_input = visual_dim + 128
        layers: list[nn.Module] = []
        prev = fusion_input
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_joints))
        self.fusion = nn.Sequential(*layers)

        # ImageNet mean/std as buffers so they move with the model's device
        self.register_buffer(
            'imgnet_mean',
            torch.tensor(self._IMGNET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            'imgnet_std',
            torch.tensor(self._IMGNET_STD,  dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, H, W, 3) uint8 → (B, 512) float feature."""
        x = img.float() / 255.0                    # (B, H, W, 3)
        x = x.permute(0, 3, 1, 2)                  # (B, 3, H, W)
        x = (x - self.imgnet_mean) / self.imgnet_std
        return self.backbone(x)                     # (B, 512)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        visual_feats = []
        for cam_key in ('camera_wrist', 'camera_top'):
            if cam_key in obs and obs[cam_key] is not None:
                visual_feats.append(self._encode_image(obs[cam_key]))
        if not visual_feats:
            raise ValueError('CNNMLPPolicy.forward: no camera images in obs')
        visual = torch.cat(visual_feats, dim=-1)

        state = torch.cat(
            [obs['joint_pos'], obs['joint_vel'], obs['eef_pos'], obs['eef_quat']],
            dim=-1,
        )
        state_feat = self.state_embed(state)
        fused = torch.cat([visual, state_feat], dim=-1)
        return self.fusion(fused)

    def predict(self, obs: dict[str, np.ndarray], normalizer=None) -> np.ndarray:
        """Numpy in → numpy out (single-step inference)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            tensor_obs = {}
            for key in ('joint_pos', 'joint_vel', 'eef_pos', 'eef_quat'):
                arr = obs[key].astype(np.float32)
                if normalizer is not None:
                    arr = normalizer.normalize(key, arr)
                tensor_obs[key] = torch.from_numpy(arr).unsqueeze(0).to(device)
            for cam_key in ('camera_wrist', 'camera_top'):
                if cam_key in obs and obs[cam_key] is not None:
                    tensor_obs[cam_key] = torch.from_numpy(
                        obs[cam_key].astype(np.uint8)
                    ).unsqueeze(0).to(device)
            action = self.forward(tensor_obs).squeeze(0).cpu().numpy()
        if normalizer is not None:
            action = normalizer.denormalize('action', action)
        return action

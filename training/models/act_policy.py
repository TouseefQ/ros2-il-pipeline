"""
ACTPolicy — Action Chunking Transformer (Zhao et al., 2023).

Architecture:
  - During training: CVAE encoder compresses action chunk → latent z
  - Transformer decoder: cross-attends over (obs_tokens, z) to predict
    a chunk of T_pred future joint positions
  - At inference: z = 0 (or sampled), decoder predicts action chunk

Reference: https://arxiv.org/abs/2304.13705
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTPolicy(nn.Module):
    """
    Full ACT model (encoder + decoder).

    Inputs (training):
      obs_dict: joint_pos, joint_vel, eef_pos, eef_quat, [images]
      action_chunk: (B, T_pred, num_joints)  target absolute joint positions

    Inputs (inference):
      obs_dict only — encoder is bypassed (z = 0)

    Output:
      predicted_chunk: (B, T_pred, num_joints)
      kl_loss: scalar (only during training)
    """

    def __init__(
        self,
        num_joints: int = 7,
        latent_dim: int = 32,
        chunk_size: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_images: bool = False,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.use_images = use_images

        # TODO Stage 7: implement CVAE encoder, transformer decoder, embeddings
        raise NotImplementedError('ACTPolicy.__init__ — Stage 7')

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        action_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (predicted_chunk, kl_loss)
            kl_loss is 0.0 at inference time.
        """
        raise NotImplementedError('ACTPolicy.forward — Stage 7')

    @torch.no_grad()
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference-only: returns (T_pred, num_joints) action chunk."""
        raise NotImplementedError('ACTPolicy.predict — Stage 7')

"""
ACTPolicy — Action Chunking Transformer (Zhao et al., 2023).

Architecture:
  - During training: CVAE encoder compresses action chunk → latent z
  - Transformer decoder: cross-attends over (obs_tokens, z) to predict
    a chunk of T_pred future joint positions
  - At inference: z = 0 (encoder bypassed), decoder predicts action chunk

Reference: https://arxiv.org/abs/2304.13705
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Returns (1, seq_len, d_model) sinusoidal positional encoding."""
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, seq_len, d_model, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe


class ACTPolicy(nn.Module):
    """
    Full ACT model (CVAE encoder + Transformer decoder).

    Inputs (training):
      obs:          dict with joint_pos, joint_vel, eef_pos, eef_quat (all batched tensors)
      action_chunk: (B, chunk_size, num_joints)  target absolute joint positions

    Inputs (inference):
      obs only — encoder bypassed, z = 0

    Outputs:
      predicted_chunk: (B, chunk_size, num_joints)
      kl_loss: scalar tensor (0.0 at inference)
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

        # joint_pos + joint_vel + eef_pos(3) + eef_quat(4)
        state_dim = num_joints * 2 + 3 + 4

        # ── CVAE Encoder ──────────────────────────────────────────────────────
        # Input: [CLS, state_token, action_tokens] → CLS output → (μ, logvar)
        self._enc_state_emb  = nn.Linear(state_dim, d_model)
        self._enc_action_emb = nn.Linear(num_joints, d_model)
        self._enc_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self._encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self._enc_mu     = nn.Linear(d_model, latent_dim)
        self._enc_logvar = nn.Linear(d_model, latent_dim)

        # ── Transformer Decoder ───────────────────────────────────────────────
        # Memory: [state_token, z_token]; Queries: chunk_size learned embeddings
        self._dec_state_emb = nn.Linear(state_dim, d_model)
        self._dec_z_emb     = nn.Linear(latent_dim, d_model)
        self._query_embed   = nn.Embedding(chunk_size, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self._decoder     = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self._action_head = nn.Linear(d_model, num_joints)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self._enc_cls, std=0.02)

    def _encode_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate obs vectors → (B, state_dim)."""
        return torch.cat(
            [obs['joint_pos'], obs['joint_vel'], obs['eef_pos'], obs['eef_quat']],
            dim=-1,
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        action_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:          dict of normalized batched tensors
            action_chunk: (B, chunk_size, num_joints) or None for inference

        Returns:
            (pred_chunk (B, chunk_size, num_joints), kl_loss scalar)
        """
        B      = next(iter(obs.values())).shape[0]
        device = next(iter(obs.values())).device
        state  = self._encode_state(obs)   # (B, state_dim)

        if action_chunk is not None:
            # CVAE encoder: [CLS, state_tok, action_toks] → μ, logvar → z
            state_tok   = self._enc_state_emb(state).unsqueeze(1)   # (B, 1, d)
            action_toks = self._enc_action_emb(action_chunk)         # (B, T, d)
            cls_tok     = self._enc_cls.expand(B, -1, -1)            # (B, 1, d)

            tokens = torch.cat([cls_tok, state_tok, action_toks], dim=1)  # (B, 2+T, d)
            tokens = tokens + _sinusoidal_pe(tokens.shape[1], tokens.shape[2], device)

            enc_out = self._encoder(tokens)   # (B, 2+T, d)
            cls_out = enc_out[:, 0]           # (B, d)

            mu     = self._enc_mu(cls_out)      # (B, latent_dim)
            logvar = self._enc_logvar(cls_out)  # (B, latent_dim)
            z      = self._reparameterize(mu, logvar)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z       = torch.zeros(B, self.latent_dim, device=device, dtype=state.dtype)
            kl_loss = torch.zeros((), device=device, dtype=state.dtype)

        # Transformer decoder: queries cross-attend over [state_mem, z_mem]
        state_mem = self._dec_state_emb(state).unsqueeze(1)   # (B, 1, d)
        z_mem     = self._dec_z_emb(z).unsqueeze(1)           # (B, 1, d)
        memory    = torch.cat([state_mem, z_mem], dim=1)      # (B, 2, d)

        q_idx   = torch.arange(self.chunk_size, device=device)
        queries = self._query_embed(q_idx).unsqueeze(0).expand(B, -1, -1)  # (B, T, d)

        dec_out    = self._decoder(queries, memory)   # (B, chunk_size, d)
        pred_chunk = self._action_head(dec_out)       # (B, chunk_size, num_joints)

        return pred_chunk, kl_loss

    @torch.no_grad()
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference-only: z=0, returns (chunk_size, num_joints)."""
        pred_chunk, _ = self.forward(obs, action_chunk=None)
        return pred_chunk[0]  # (chunk_size, num_joints)

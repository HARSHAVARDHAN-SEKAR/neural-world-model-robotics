"""
Transformer world model (PyTorch reference implementation).

Sequence-to-sequence world model in the spirit of transformer-based
world models: consume the past context of (latent state, action) tokens
and autoregressively predict future latent states.

    Input : past  100 steps (10 s @ 10 Hz)  of [z_t, a_t]
    Output: future 50 steps (5 s @ 10 Hz)   of ẑ_{t+1..t+50}

The runnable CPU pipeline in src/nwm uses MLP world models with the same
rollout interface; swap this class in when training at scale on GPU.

Requires: torch >= 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerWorldModel(nn.Module):
    def __init__(self, latent_dim: int = 8, action_dim: int = 2,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 context: int = 100):
        super().__init__()
        self.context = context
        self.token = nn.Linear(latent_dim + action_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads,
                                           dim_feedforward=4 * d_model,
                                           batch_first=True,
                                           norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, latent_dim)
        mask = torch.triu(torch.full((context, context), float("-inf")), 1)
        self.register_buffer("causal_mask", mask)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z (B,T,latent), a (B,T,action) -> next-latent prediction (B,T,latent)."""
        T = z.shape[1]
        h = self.token(torch.cat([z, a], dim=-1)) + self.pos[:, :T]
        h = self.backbone(h, mask=self.causal_mask[:T, :T])
        return self.head(h)

    @torch.no_grad()
    def imagine(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor,
                a_future: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive imagination.
        z_ctx (B,Tc,latent), a_ctx (B,Tc,action), a_future (B,H,action)
        -> imagined latents (B,H,latent)
        """
        z, a = z_ctx.clone(), a_ctx.clone()
        outs = []
        for h in range(a_future.shape[1]):
            z_next = self.forward(z, a)[:, -1:]
            outs.append(z_next)
            z = torch.cat([z, z_next], dim=1)[:, -self.context:]
            a = torch.cat([a, a_future[:, h:h + 1]], dim=1)[:, -self.context:]
        return torch.cat(outs, dim=1)


if __name__ == "__main__":
    wm = TransformerWorldModel()
    z = torch.randn(4, 100, 8)
    a = torch.randn(4, 100, 2)
    af = torch.randn(4, 50, 2)
    print("imagined:", wm.imagine(z, a, af).shape)  # (4, 50, 8)

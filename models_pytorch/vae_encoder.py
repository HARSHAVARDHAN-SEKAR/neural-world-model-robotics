"""
Variational Autoencoder sensor encoder (PyTorch reference implementation).

Compresses raw sensor observations (e.g. a 24-beam LiDAR scan, or an
image feature vector) into a compact latent state z used by the world
model. This file is a GPU-scale counterpart to the PCA/feature encoder
used in the runnable CPU pipeline; the interfaces match.

Usage:
    enc = SensorVAE(obs_dim=24, latent_dim=8)
    z, mu, logvar = enc(obs)
    loss = vae_loss(recon, obs, mu, logvar)

Requires: torch >= 2.0  (pip install torch)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorVAE(nn.Module):
    def __init__(self, obs_dim: int = 24, latent_dim: int = 8,
                 hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, z, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1e-3) -> torch.Tensor:
    rec = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld


if __name__ == "__main__":
    vae = SensorVAE()
    x = torch.rand(32, 24) * 6.0
    recon, z, mu, logvar = vae(x)
    print("latent z:", z.shape, "loss:", float(vae_loss(recon, x, mu, logvar)))

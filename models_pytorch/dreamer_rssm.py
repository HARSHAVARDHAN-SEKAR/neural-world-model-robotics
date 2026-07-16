"""
Dreamer-style Recurrent State-Space Model (PyTorch reference).

Minimal RSSM in the spirit of DreamerV3:

  deterministic path : h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
  stochastic prior   : ẑ_t ~ p(z | h_t)
  posterior          : z_t ~ q(z | h_t, embed(o_t))
  decoder            : ô_t = dec(h_t, z_t)

Train with reconstruction + KL(posterior || prior); imagine futures by
rolling the prior forward under candidate action sequences, exactly the
role the MLP world model plays in the runnable CPU pipeline.

Requires: torch >= 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, obs_dim: int = 24, action_dim: int = 2,
                 stoch: int = 16, deter: int = 128, hidden: int = 128):
        super().__init__()
        self.stoch, self.deter = stoch, deter
        self.embed = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU())
        self.gru = nn.GRUCell(stoch + action_dim, deter)
        self.prior_net = nn.Sequential(nn.Linear(deter, hidden), nn.SiLU(),
                                       nn.Linear(hidden, 2 * stoch))
        self.post_net = nn.Sequential(nn.Linear(deter + hidden, hidden),
                                      nn.SiLU(), nn.Linear(hidden, 2 * stoch))
        self.decoder = nn.Sequential(nn.Linear(deter + stoch, hidden),
                                     nn.SiLU(), nn.Linear(hidden, obs_dim))

    @staticmethod
    def _dist(stats: torch.Tensor):
        mu, logstd = stats.chunk(2, dim=-1)
        std = F.softplus(logstd) + 1e-3
        return mu, std

    def initial(self, batch: int, device=None):
        return (torch.zeros(batch, self.deter, device=device),
                torch.zeros(batch, self.stoch, device=device))

    def step_prior(self, h, z, a):
        h = self.gru(torch.cat([z, a], -1), h)
        mu, std = self._dist(self.prior_net(h))
        z_next = mu + std * torch.randn_like(std)
        return h, z_next, (mu, std)

    def step_posterior(self, h, obs):
        e = self.embed(obs)
        mu, std = self._dist(self.post_net(torch.cat([h, e], -1)))
        z = mu + std * torch.randn_like(std)
        return z, (mu, std)

    def observe(self, obs_seq, act_seq):
        """Teacher-forced pass. obs (B,T,obs), act (B,T,act). Returns losses."""
        B, T, _ = obs_seq.shape
        h, z = self.initial(B, obs_seq.device)
        rec_loss = kl_loss = 0.0
        for t in range(T):
            h, _, (pmu, pstd) = self.step_prior(h, z, act_seq[:, t])
            z, (qmu, qstd) = self.step_posterior(h, obs_seq[:, t])
            recon = self.decoder(torch.cat([h, z], -1))
            rec_loss = rec_loss + F.mse_loss(recon, obs_seq[:, t])
            kl = (torch.log(pstd / qstd)
                  + (qstd ** 2 + (qmu - pmu) ** 2) / (2 * pstd ** 2) - 0.5)
            kl_loss = kl_loss + kl.sum(-1).mean()
        return rec_loss / T, kl_loss / T

    @torch.no_grad()
    def imagine(self, h, z, actions):
        """Roll the prior forward. actions (B,H,act) -> decoded obs (B,H,obs)."""
        outs = []
        for t in range(actions.shape[1]):
            h, z, _ = self.step_prior(h, z, actions[:, t])
            outs.append(self.decoder(torch.cat([h, z], -1)))
        return torch.stack(outs, dim=1)


if __name__ == "__main__":
    m = RSSM()
    obs = torch.rand(4, 30, 24)
    act = torch.rand(4, 30, 2)
    rec, kl = m.observe(obs, act)
    h, z = m.initial(4)
    fut = m.imagine(h, z, torch.rand(4, 50, 2))
    print("rec", float(rec), "kl", float(kl), "imagined", fut.shape)

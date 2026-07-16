"""
Ensemble world model — upgrade #2 (uncertainty-aware prediction).

Instead of one deterministic obstacle predictor, train M bootstrapped
members. At planning time:

  * each member imagines its own future  -> M plausible futures
  * the spread (disagreement) between members is an *epistemic
    uncertainty* estimate: where members disagree, the model does not
    know — and the planner should hedge.

This is the standard probabilistic-ensembles technique from model-based
RL (PETS, Chua et al. 2018) applied to obstacle motion. It keeps the
whole pipeline CPU-runnable while providing the "multiple futures with
probabilities" capability of research-grade systems.
"""

from __future__ import annotations

import numpy as np

from nwm.models.world_model import ObstacleMotionModel


class EnsembleObstacleModel:
    """Bag of bootstrapped ObstacleMotionModels sharing one interface."""

    def __init__(self, n_members: int = 5, seed: int = 0):
        self.members = [ObstacleMotionModel(seed=seed + 31 * i)
                        for i in range(n_members)]
        self.n_members = n_members
        self.seed = seed

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Each member trains on its own bootstrap resample of the data."""
        rng = np.random.default_rng(self.seed)
        mses = []
        n = len(X)
        for m in self.members:
            idx = rng.integers(0, n, size=n)
            mses.append(m.fit(X[idx], Y[idx]))
        return float(np.mean(mses))

    # ------------------------------------------------------------------ #
    def rollout_all(self, pos_hist: np.ndarray, horizon: int) -> np.ndarray:
        """All imagined futures. Returns (M, horizon, K, 2)."""
        return np.stack([m.rollout(pos_hist, horizon) for m in self.members])

    def rollout(self, pos_hist: np.ndarray, horizon: int) -> np.ndarray:
        """Ensemble-mean future (drop-in replacement for a single model)."""
        return self.rollout_all(pos_hist, horizon).mean(axis=0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def disagreement(futures: np.ndarray) -> np.ndarray:
        """
        futures: (M, H, K, 2) -> per-step, per-obstacle epistemic
        uncertainty (H, K): mean distance of members from the ensemble mean.
        """
        mean = futures.mean(axis=0, keepdims=True)
        return np.linalg.norm(futures - mean, axis=-1).mean(axis=0)

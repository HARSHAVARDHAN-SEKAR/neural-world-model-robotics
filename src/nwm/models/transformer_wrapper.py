"""
Transformer world-model wrapper — Stage 2.

Loads a checkpoint trained by scripts/train_world_transformer.py and
exposes the SAME `.rollout(pos_hist, horizon) -> (horizon, K, 2)`
interface as nwm.models.world_model.ObstacleMotionModel, so it drops
straight into NeuralMPCPlanner / OccupancyNeuralMPC with no other change.

Scope note (kept honest in the benchmark): the transformer predicts in
the same privileged-state latent the MLP ensemble uses — robot state
plus flattened obstacle coordinates, latent layout [robot(3), obst(2K)].
Test 7 therefore measures the *architecture* (Transformer vs MLP
ensemble), not raw-sensor perception; the VAE-over-lidar variant is a
later roadmap stage.

torch is imported lazily so the rest of the repo (and CI) runs without
torch installed. If torch is missing, constructing this class raises a
clear ImportError.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models_pytorch"


class TransformerWorldModelWrapper:
    """Adapts a trained TransformerWorldModel to the obstacle-model API."""

    def __init__(self, checkpoint_path, n_obstacles: int = 6,
                 robot_state_dim: int = 3, device: str = "cpu"):
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "TransformerWorldModelWrapper requires torch. "
                "Install with `pip install torch`.") from e
        import torch
        if str(MODELS_DIR) not in sys.path:
            sys.path.insert(0, str(MODELS_DIR))
        from world_transformer import TransformerWorldModel

        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        self.latent_dim = ckpt["latent_dim"]
        self.action_dim = ckpt["action_dim"]
        self.context = ckpt["context"]
        self.robot_state_dim = robot_state_dim
        self.n_obstacles = n_obstacles
        # obstacle coords occupy latent dims [robot_state_dim : ]
        self.obst_slice = slice(robot_state_dim, self.latent_dim)

        self.model = TransformerWorldModel(
            latent_dim=self.latent_dim, action_dim=self.action_dim,
            context=self.context).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # rolling latent/action context, seeded lazily on first rollout
        self._z_ctx = None
        self._a_ctx = None

    # ------------------------------------------------------------------ #
    def _obst_to_latent(self, obst_pos: np.ndarray,
                        robot_state: np.ndarray | None = None) -> np.ndarray:
        """Assemble a latent vector [robot(3), obst_flat(2K)]."""
        robot = (robot_state if robot_state is not None
                 else np.zeros(self.robot_state_dim))
        return np.concatenate([robot[:self.robot_state_dim],
                               obst_pos.reshape(-1)]).astype(np.float32)

    def rollout(self, pos_hist: np.ndarray, horizon: int,
                robot_hist: np.ndarray | None = None,
                action_hist: np.ndarray | None = None) -> np.ndarray:
        """
        pos_hist: (HISTORY+1, K, 2) recent obstacle positions.
        Returns predicted future obstacle positions (horizon, K, 2).

        The transformer needs a (state, action) context. We build a
        context from the obstacle history (robot dims + actions default
        to zeros when not supplied — obstacle dynamics here are
        independent of the robot, so this is exact for the obstacle
        channel). Missing context is left-padded by repeating the oldest
        available step up to `context` length.
        """
        import torch

        T_hist, K, _ = pos_hist.shape
        # build latent history from obstacle positions
        z_hist = np.stack([self._obst_to_latent(pos_hist[t]) for t in range(T_hist)])
        # left-pad to context length by repeating the first frame
        if T_hist < self.context:
            pad = np.repeat(z_hist[:1], self.context - T_hist, axis=0)
            z_ctx = np.concatenate([pad, z_hist], axis=0)
        else:
            z_ctx = z_hist[-self.context:]

        a_ctx = np.zeros((self.context, self.action_dim), dtype=np.float32)
        a_future = np.zeros((horizon, self.action_dim), dtype=np.float32)

        z_ctx_t = torch.tensor(z_ctx[None], dtype=torch.float32, device=self.device)
        a_ctx_t = torch.tensor(a_ctx[None], dtype=torch.float32, device=self.device)
        a_fut_t = torch.tensor(a_future[None], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            latent_future = self.model.imagine(z_ctx_t, a_ctx_t, a_fut_t)[0]  # (H, latent)
        latent_future = latent_future.cpu().numpy()
        obst = latent_future[:, self.obst_slice].reshape(horizon, K, 2)
        return obst

    # ------------------------------------------------------------------ #
    def rollout_all(self, pos_hist: np.ndarray, horizon: int) -> np.ndarray:
        """
        Single-model wrapper exposes one deterministic future; return it
        with a leading singleton member axis so it is drop-in compatible
        anywhere EnsembleObstacleModel.rollout_all is expected (M=1).
        For multi-hypothesis, the Dreamer RSSM (stochastic latents) is the
        roadmap path; a deterministic transformer has a single rollout.
        """
        return self.rollout(pos_hist, horizon)[None]

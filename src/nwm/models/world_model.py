"""
Neural World Model (runnable, dependency-light version).

Two learned components:

1. RobotDynamicsModel
   MLP that maps (v_cmd, w_cmd) -> body-frame displacement
   (dx_body, dy_body, dtheta). Learned purely from experience, then used
   inside the imagination planner instead of the analytic model.

2. ObstacleMotionModel
   MLP that maps a short history window of an obstacle's motion
   (H past velocity vectors) -> next-step velocity. Rolled out
   autoregressively to imagine obstacle futures. Because simulator
   obstacles turn, this beats the classical constant-velocity (CV)
   predictor — this gap is exactly what the ADE/FDE benchmark measures.

A PyTorch VAE + Transformer / Dreamer-style implementation of the same
interfaces lives in `models_pytorch/` for GPU-scale training on real
sensor data; the sklearn MLPs here keep the full research pipeline
executable on any CPU in minutes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

from nwm.env.simulator import DT

HISTORY = 6  # past steps fed to the obstacle motion model


# ====================================================================== #
class RobotDynamicsModel:
    """Learns body-frame robot motion from (action -> displacement) pairs."""

    def __init__(self, seed: int = 0):
        self.net = MLPRegressor(hidden_layer_sizes=(64, 64),
                                activation="tanh",
                                max_iter=800,
                                random_state=seed,
                                early_stopping=True,
                                n_iter_no_change=20)
        self.trained = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def make_dataset(states, actions, next_states):
        """Convert global transitions into body-frame supervised pairs."""
        X, Y = [], []
        for s, a, sn in zip(states, actions, next_states):
            th = s[2]
            dx, dy = sn[0] - s[0], sn[1] - s[1]
            # rotate world displacement into the body frame
            bx = np.cos(-th) * dx - np.sin(-th) * dy
            by = np.sin(-th) * dx + np.cos(-th) * dy
            dth = np.arctan2(np.sin(sn[2] - s[2]), np.cos(sn[2] - s[2]))
            X.append(a)
            Y.append([bx, by, dth])
        return np.asarray(X), np.asarray(Y)

    def fit(self, states, actions, next_states):
        X, Y = self.make_dataset(states, actions, next_states)
        self.net.fit(X, Y)
        self.trained = True
        pred = self.net.predict(X)
        return float(np.mean((pred - Y) ** 2))

    # ------------------------------------------------------------------ #
    def step(self, robot: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One imagined step. robot=(x,y,th), action=(v,w)."""
        d = self.net.predict(np.asarray(action, dtype=float)[None, :])[0]
        x, y, th = robot
        gx = x + np.cos(th) * d[0] - np.sin(th) * d[1]
        gy = y + np.sin(th) * d[0] + np.cos(th) * d[1]
        gth = np.arctan2(np.sin(th + d[2]), np.cos(th + d[2]))
        return np.array([gx, gy, gth])

    def step_batch(self, robots: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Vectorised imagined step for N rollouts. robots (N,3), actions (N,2)."""
        d = self.net.predict(np.asarray(actions, dtype=float))
        th = robots[:, 2]
        gx = robots[:, 0] + np.cos(th) * d[:, 0] - np.sin(th) * d[:, 1]
        gy = robots[:, 1] + np.sin(th) * d[:, 0] + np.cos(th) * d[:, 1]
        gth = np.arctan2(np.sin(th + d[:, 2]), np.cos(th + d[:, 2]))
        return np.stack([gx, gy, gth], axis=1)


# ====================================================================== #
def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class ObstacleMotionModel:
    """
    Rotation-invariant obstacle motion predictor.

    Features : the last HISTORY velocity vectors of an obstacle, rotated
               into the frame of its most recent heading. This makes the
               model equivariant to heading and lets it generalize to
               unseen speeds/turn-rates (verified in Test 3).
    Target   : the next-step velocity, expressed in the same local frame.
    Rollout  : autoregressive — each predicted velocity is appended to
               the history window, imagining several seconds ahead.
    Training : wall-bounce transitions (large instantaneous heading
               jumps) are filtered out so the model learns smooth
               dynamics rather than discontinuities.
    """

    BOUNCE_ANGLE = 0.5  # rad; larger heading jumps are treated as bounces

    def __init__(self, seed: int = 0):
        self.net = MLPRegressor(hidden_layer_sizes=(96, 96),
                                activation="tanh",
                                max_iter=600,
                                random_state=seed,
                                early_stopping=True,
                                n_iter_no_change=15)
        self.trained = False

    # ------------------------------------------------------------------ #
    @classmethod
    def make_dataset(cls, obst_tracks: np.ndarray):
        """
        obst_tracks: (T, K, 2) obstacle positions over one episode.
        Returns rotation-invariant (X, Y) pairs, bounce-filtered.
        """
        vel = np.diff(obst_tracks, axis=0) / DT          # (T-1, K, 2)
        X, Y = [], []
        T, K, _ = vel.shape
        for k in range(K):
            for t in range(HISTORY, T):
                win = vel[t - HISTORY:t, k]              # (HISTORY, 2)
                ang = np.arctan2(win[:, 1], win[:, 0])
                jump = np.abs(np.arctan2(np.sin(np.diff(ang)),
                                         np.cos(np.diff(ang))))
                if jump.size and jump.max() > cls.BOUNCE_ANGLE:
                    continue                             # crosses a bounce
                R = _rot(-ang[-1])
                X.append((win @ R.T).ravel())
                Y.append(vel[t, k] @ R.T)
        return np.asarray(X), np.asarray(Y)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.net.fit(X, Y)
        self.trained = True
        pred = self.net.predict(X)
        return float(np.mean((pred - Y) ** 2))

    # ------------------------------------------------------------------ #
    def rollout(self, pos_hist: np.ndarray, horizon: int) -> np.ndarray:
        """
        pos_hist: (HISTORY+1, K, 2) most recent obstacle positions.
        Returns imagined future positions (horizon, K, 2).
        """
        vel = np.diff(pos_hist, axis=0) / DT             # (HISTORY, K, 2)
        pos = pos_hist[-1].copy()
        K = pos.shape[0]
        out = np.empty((horizon, K, 2))
        for h in range(horizon):
            Xb = np.empty((K, 2 * HISTORY))
            angs = np.empty(K)
            for k in range(K):
                win = vel[:, k]
                a = np.arctan2(win[-1, 1], win[-1, 0])
                Xb[k] = (win @ _rot(-a).T).ravel()
                angs[k] = a
            v_loc = self.net.predict(Xb)                 # (K, 2)
            v_next = np.stack([v_loc[k] @ _rot(angs[k]).T
                               for k in range(K)])
            pos = pos + v_next * DT
            out[h] = pos
            vel = np.concatenate([vel[1:], v_next[None]], axis=0)
        return out


# ====================================================================== #
def constant_velocity_rollout(pos_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Classical CV baseline: extrapolate the last observed velocity."""
    v = (pos_hist[-1] - pos_hist[-2]) / DT
    steps = np.arange(1, horizon + 1)[:, None, None]
    return pos_hist[-1][None] + v[None] * steps * DT


# ====================================================================== #
def save_models(path: str | Path, robot_model, obst_model):
    with open(path, "wb") as f:
        pickle.dump({"robot": robot_model, "obst": obst_model}, f)


def load_models(path: str | Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["robot"], d["obst"]

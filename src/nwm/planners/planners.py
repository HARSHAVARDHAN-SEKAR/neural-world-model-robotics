"""
Planners benchmarked in this project.

Reactive      : go-to-goal, hard stop when an obstacle is close ahead.
DWA           : Dynamic Window Approach — constant (v, w) rollouts,
                physics model + constant-velocity obstacle prediction.
MPPI          : sampling-based MPC using the *analytic* physics model
                and constant-velocity obstacle prediction.
Neural-MPC    : the SAME MPPI optimizer, but every rollout runs inside
                the *learned* world model (learned robot dynamics +
                learned obstacle motion predictor). This is the
                "imagine futures before acting" planner.

All prediction-based planners share cost terms so the comparison
isolates the effect of the world model, not the cost design.
"""

from __future__ import annotations

import numpy as np

from nwm.env.simulator import (ARENA, DT, OBST_RADIUS, ROBOT_RADIUS,
                               V_MAX, W_MAX, DynamicWorld)
from nwm.models.world_model import (HISTORY, constant_velocity_rollout)

SAFE = ROBOT_RADIUS + OBST_RADIUS          # hard collision distance
SOFT = SAFE + 0.35                         # start penalising here


# ---------------------------------------------------------------------- #
def _traj_cost(traj, obst_pred, goal, actions):
    """
    Shared cost for prediction-based planners.
    traj      : (H, N, 3) imagined robot states
    obst_pred : (H, K, 2) imagined obstacle positions
    goal      : (2,)
    actions   : (N, H, 2)
    returns   : (N,) cost per rollout
    """
    H, N = traj.shape[0], traj.shape[1]
    pos = traj[:, :, :2]                                    # (H, N, 2)
    d_obs = np.linalg.norm(pos[:, :, None, :] -
                           obst_pred[:, None, :, :], axis=-1)  # (H, N, K)
    min_d = d_obs.min(axis=2)                               # (H, N)

    collision = 1e4 * (min_d < SAFE).any(axis=0)
    proximity = np.clip(SOFT - min_d, 0.0, None).sum(axis=0) * 40.0

    goal_term = 3.0 * np.linalg.norm(pos[-1] - goal[None], axis=-1)
    progress = 0.4 * np.linalg.norm(pos - goal[None, None], axis=-1).mean(axis=0)

    effort = 0.02 * (actions[:, :, 0] ** 2).sum(axis=1)
    smooth = 0.05 * (np.diff(actions, axis=1) ** 2).sum(axis=(1, 2))

    wall = 5.0 * ((pos < 0.4) | (pos > ARENA - 0.4)).any(axis=(0, 2))
    return collision + proximity + goal_term + progress + effort + smooth + wall


def _physics_rollout(robot, actions):
    """Vectorised analytic unicycle rollout. actions (N, H, 2) -> (H, N, 3)."""
    N, H, _ = actions.shape
    st = np.tile(robot, (N, 1))
    out = np.empty((H, N, 3))
    for h in range(H):
        v = np.clip(actions[:, h, 0], 0.0, V_MAX)
        w = np.clip(actions[:, h, 1], -W_MAX, W_MAX)
        st = st.copy()
        st[:, 0] += v * np.cos(st[:, 2]) * DT
        st[:, 1] += v * np.sin(st[:, 2]) * DT
        st[:, 2] = np.arctan2(np.sin(st[:, 2] + w * DT),
                              np.cos(st[:, 2] + w * DT))
        out[h] = st
    return out


# ====================================================================== #
class ReactivePlanner:
    """Naive baseline: head to goal, freeze when something is close ahead."""

    name = "Reactive"

    def __init__(self, **_):
        pass

    def reset(self):
        pass

    def act(self, obs):
        robot, goal = obs["robot"], obs["goal"]
        to_goal = goal - robot[:2]
        heading = np.arctan2(to_goal[1], to_goal[0])
        err = np.arctan2(np.sin(heading - robot[2]),
                         np.cos(heading - robot[2]))
        rel = obs["obst_pos"] - robot[:2]
        d = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - robot[2]
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        danger = np.any((d < 1.4) & (np.abs(ang) < np.pi / 3))
        v = 0.0 if danger else min(V_MAX, 0.9 * np.linalg.norm(to_goal))
        return np.array([v, np.clip(2.0 * err, -W_MAX, W_MAX)])


# ====================================================================== #
class DWAPlanner:
    """Dynamic Window Approach with CV obstacle prediction."""

    name = "DWA"

    def __init__(self, horizon: int = 15, n_v: int = 7, n_w: int = 11, **_):
        self.h = horizon
        vs = np.linspace(0.0, V_MAX, n_v)
        ws = np.linspace(-W_MAX, W_MAX, n_w)
        grid = np.array([[v, w] for v in vs for w in ws])
        self.actions = np.repeat(grid[:, None, :], horizon, axis=1)  # (N,H,2)

    def reset(self):
        pass

    def act(self, obs):
        traj = _physics_rollout(obs["robot"], self.actions)
        hist = np.stack([obs["obst_pos"] - obs["obst_vel"] * DT,
                         obs["obst_pos"]])
        obst_pred = constant_velocity_rollout(hist, self.h)
        cost = _traj_cost(traj, obst_pred, obs["goal"], self.actions)
        return self.actions[int(np.argmin(cost)), 0].copy()


# ====================================================================== #
class MPPIPlanner:
    """Sampling MPC with analytic dynamics + CV obstacle prediction."""

    name = "MPPI"

    def __init__(self, horizon: int = 18, samples: int = 220,
                 lam: float = 1.0, seed: int = 0, **_):
        self.h, self.n, self.lam = horizon, samples, lam
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.mean = np.zeros((self.h, 2))
        self.mean[:, 0] = 0.6 * V_MAX

    def _sample_actions(self):
        noise = self.rng.normal(0.0, [0.35, 0.7],
                                size=(self.n, self.h, 2))
        acts = self.mean[None] + noise
        acts[:, :, 0] = np.clip(acts[:, :, 0], 0.0, V_MAX)
        acts[:, :, 1] = np.clip(acts[:, :, 1], -W_MAX, W_MAX)
        return acts

    def _predict_obstacles(self, obs):
        hist = np.stack([obs["obst_pos"] - obs["obst_vel"] * DT,
                         obs["obst_pos"]])
        return constant_velocity_rollout(hist, self.h)

    def _rollout(self, robot, acts):
        return _physics_rollout(robot, acts)

    def act(self, obs):
        acts = self._sample_actions()
        traj = self._rollout(obs["robot"], acts)
        obst_pred = self._predict_obstacles(obs)
        cost = _traj_cost(traj, obst_pred, obs["goal"], acts)
        w = np.exp(-(cost - cost.min()) / self.lam)
        w /= w.sum()
        self.mean = np.tensordot(w, acts, axes=1)
        a = self.mean[0].copy()
        self.mean = np.roll(self.mean, -1, axis=0)
        self.mean[-1] = self.mean[-2]
        return a


# ====================================================================== #
class NeuralMPCPlanner(MPPIPlanner):
    """
    Imagination planner: identical MPPI optimizer, but futures are
    imagined by the LEARNED world model:
      - learned robot dynamics (RobotDynamicsModel)
      - learned obstacle motion predictor (ObstacleMotionModel)
    Maintains a short observation history so the obstacle model can
    infer curvature online.
    """

    name = "Neural-MPC"

    def __init__(self, robot_model, obst_model, **kw):
        super().__init__(**kw)
        self.robot_model = robot_model
        self.obst_model = obst_model
        self.pos_hist: list[np.ndarray] = []

    def reset(self):
        super().reset()
        self.pos_hist = []

    def _predict_obstacles(self, obs):
        self.pos_hist.append(obs["obst_pos"].copy())
        if len(self.pos_hist) > HISTORY + 1:
            self.pos_hist = self.pos_hist[-(HISTORY + 1):]
        if len(self.pos_hist) < HISTORY + 1:
            hist = np.stack([obs["obst_pos"] - obs["obst_vel"] * DT,
                             obs["obst_pos"]])
            return constant_velocity_rollout(hist, self.h)
        return self.obst_model.rollout(np.stack(self.pos_hist), self.h)

    def _rollout(self, robot, acts):
        N, H, _ = acts.shape
        st = np.tile(robot, (N, 1))
        out = np.empty((H, N, 3))
        for h in range(H):
            st = self.robot_model.step_batch(st, acts[:, h])
            out[h] = st
        return out


# ====================================================================== #
def run_episode(env: DynamicWorld, planner, max_steps: int = 350):
    """Roll one episode, return metrics + trajectories for plotting."""
    obs = env.observe()
    planner.reset()
    robot_traj = [obs["robot"][:2].copy()]
    obst_traj = [obs["obst_pos"].copy()]
    actions = []
    success = collided = False
    for _ in range(max_steps):
        a = planner.act(obs)
        actions.append(a)
        obs, hit, reached = env.step(a)
        robot_traj.append(obs["robot"][:2].copy())
        obst_traj.append(obs["obst_pos"].copy())
        if hit:
            collided = True
            break
        if reached:
            success = True
            break
    actions = np.asarray(actions) if actions else np.zeros((1, 2))
    robot_traj = np.asarray(robot_traj)
    path_len = float(np.linalg.norm(np.diff(robot_traj, axis=0),
                                    axis=1).sum())
    return {
        "success": success,
        "collision": collided,
        "steps": len(actions),
        "path_length": path_len,
        "smoothness": float((np.diff(actions, axis=0) ** 2).sum())
        if len(actions) > 1 else 0.0,
        "energy": float((actions[:, 0] ** 2).sum() * DT),
        "robot_traj": robot_traj,
        "obst_traj": np.asarray(obst_traj),
        "goal": obs["goal"],
    }


# ====================================================================== #
class RiskAwareNeuralMPC(NeuralMPCPlanner):
    """
    Upgrade #2: risk-aware imagination planner.

    Uses an EnsembleObstacleModel. Every candidate action sequence is
    evaluated against ALL M imagined futures; the rollout's cost is the
    CVaR (Conditional Value-at-Risk) over members — the average of the
    worst `risk_q` fraction of futures. The planner therefore avoids
    actions that are safe *on average* but catastrophic in plausible
    futures the model is uncertain about.

        cost(rollout) = CVaR_q over m of cost(rollout | future_m)
    """

    name = "Risk-MPC"

    def __init__(self, robot_model, ensemble_model, risk_q: float = 0.4,
                 **kw):
        super().__init__(robot_model, ensemble_model, **kw)
        self.risk_q = risk_q

    def _predict_obstacles_all(self, obs):
        """(M, H, K, 2) imagined futures; CV fallback until history fills."""
        self.pos_hist.append(obs["obst_pos"].copy())
        if len(self.pos_hist) > HISTORY + 1:
            self.pos_hist = self.pos_hist[-(HISTORY + 1):]
        if len(self.pos_hist) < HISTORY + 1:
            hist = np.stack([obs["obst_pos"] - obs["obst_vel"] * DT,
                             obs["obst_pos"]])
            return constant_velocity_rollout(hist, self.h)[None]
        return self.obst_model.rollout_all(np.stack(self.pos_hist), self.h)

    def act(self, obs):
        acts = self._sample_actions()
        traj = self._rollout(obs["robot"], acts)
        futures = self._predict_obstacles_all(obs)          # (M, H, K, 2)
        costs = np.stack([_traj_cost(traj, f, obs["goal"], acts)
                          for f in futures])                # (M, N)
        M = costs.shape[0]
        k = max(1, int(np.ceil(self.risk_q * M)))
        cvar = np.sort(costs, axis=0)[-k:].mean(axis=0)     # worst-q mean
        w = np.exp(-(cvar - cvar.min()) / self.lam)
        w /= w.sum()
        self.mean = np.tensordot(w, acts, axes=1)
        a = self.mean[0].copy()
        self.mean = np.roll(self.mean, -1, axis=0)
        self.mean[-1] = self.mean[-2]
        return a


# ====================================================================== #
class OccupancyNeuralMPC(NeuralMPCPlanner):
    """
    Stage 1 upgrade: plans against a PREDICTED OCCUPANCY GRID instead of
    (or alongside) per-obstacle point distances.

    At each planning step, the ensemble's imagined obstacle positions at
    every horizon step are rasterized into an occupancy grid (member
    disagreement naturally widens/softens the grid — see
    nwm.env.occupancy.rasterize_ensemble). The MPPI cost then reads
    predicted occupancy directly under each candidate trajectory, in
    addition to the existing point-based cost, so the planner reasons
    about "how much of this cell will be blocked" rather than only
    "how far is the nearest point estimate".
    """

    name = "Occupancy-MPC"

    def __init__(self, robot_model, ensemble_model, occ_weight: float = 55.0,
                 resolution: float = 0.5, **kw):
        super().__init__(robot_model, ensemble_model, **kw)
        self.occ_weight = occ_weight
        self.resolution = resolution

    def act(self, obs):
        from nwm.env.occupancy import rasterize_ensemble, trajectory_occupancy_cost

        acts = self._sample_actions()
        traj = self._rollout(obs["robot"], acts)             # (H, N, 3)

        self.pos_hist.append(obs["obst_pos"].copy())
        if len(self.pos_hist) > HISTORY + 1:
            self.pos_hist = self.pos_hist[-(HISTORY + 1):]
        if len(self.pos_hist) < HISTORY + 1:
            hist = np.stack([obs["obst_pos"] - obs["obst_vel"] * DT,
                             obs["obst_pos"]])
            fut = constant_velocity_rollout(hist, self.h)
            occ_cost = _traj_cost(traj, fut, obs["goal"], acts)
        else:
            all_futures = self.obst_model.rollout_all(
                np.stack(self.pos_hist), self.h)               # (M, H, K, 2)
            point_fut = all_futures.mean(axis=0)                # (H, K, 2)
            base_cost = _traj_cost(traj, point_fut, obs["goal"], acts)

            # occupancy term: rasterize each horizon step's ensemble,
            # accumulate cost of the imagined trajectory passing through it
            occ_extra = np.zeros(acts.shape[0])
            for h in range(self.h):
                occ = rasterize_ensemble(all_futures[:, h], self.resolution)
                occ_extra += trajectory_occupancy_cost(
                    traj[h:h + 1, :, :2], occ, self.resolution,
                    weight=self.occ_weight)
            occ_cost = base_cost + occ_extra

        w = np.exp(-(occ_cost - occ_cost.min()) / self.lam)
        w /= w.sum()
        self.mean = np.tensordot(w, acts, axes=1)
        a = self.mean[0].copy()
        self.mean = np.roll(self.mean, -1, axis=0)
        self.mean[-1] = self.mean[-2]
        return a

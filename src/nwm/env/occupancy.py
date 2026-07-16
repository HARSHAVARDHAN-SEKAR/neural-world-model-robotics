"""
Occupancy grid representation — Stage 1 upgrade.

Motivation (see docs/ROADMAP.md Stage 1): predicting per-obstacle (x, y)
points throws away information about *shape* and *uncertainty*. A future
occupancy grid instead answers "what fraction of this cell will be
blocked, and how confident are we?" — which is what the planner actually
needs, and what modern autonomous-driving stacks predict (occupancy flow
/ occupancy networks) rather than per-agent trajectories alone.

This module provides:
  - `world_to_grid` / `grid_to_world`      : coordinate transforms
  - `rasterize_obstacles`                  : ground-truth occupancy from
                                              true obstacle positions
  - `rasterize_predicted`                  : occupancy from a *single*
                                              imagined future (soft disks)
  - `rasterize_ensemble`                   : occupancy from an ensemble
                                              of imagined futures — cells
                                              are "occupied" if occupied
                                              in ANY member, giving a
                                              probability-like map that
                                              naturally widens under
                                              uncertainty (no separate
                                              disagreement term needed)
  - `occupancy_iou` / `occupancy_soft_iou` : evaluation metric for
                                              Test 6 (occupancy prediction
                                              quality vs point ADE/FDE)
  - `trajectory_occupancy_cost`            : planner cost term that reads
                                              predicted occupancy under
                                              a batch of imagined robot
                                              trajectories

Grid convention: grid[row, col] with row = y-axis, col = x-axis, origin
at world (0, 0) in the bottom-left, matching imshow(..., origin="lower").
"""

from __future__ import annotations

import numpy as np

from nwm.env.simulator import ARENA, OBST_RADIUS, ROBOT_RADIUS

DEFAULT_RES = 0.5          # metres per cell
DEFAULT_SOFT_SIGMA = 0.35  # metres, Gaussian splat std-dev for soft occupancy


def grid_size(resolution: float = DEFAULT_RES) -> int:
    return int(round(ARENA / resolution))


def world_to_grid(xy: np.ndarray, resolution: float = DEFAULT_RES) -> np.ndarray:
    """(..., 2) world coords -> (..., 2) integer [row, col] grid indices."""
    n = grid_size(resolution)
    col = np.clip((xy[..., 0] / resolution).astype(int), 0, n - 1)
    row = np.clip((xy[..., 1] / resolution).astype(int), 0, n - 1)
    return np.stack([row, col], axis=-1)


def grid_to_world(row: np.ndarray, col: np.ndarray,
                  resolution: float = DEFAULT_RES) -> np.ndarray:
    """Cell centers in world coordinates."""
    x = (col + 0.5) * resolution
    y = (row + 0.5) * resolution
    return np.stack([x, y], axis=-1)


def _cell_centers(resolution: float):
    n = grid_size(resolution)
    idx = np.arange(n)
    xs = (idx + 0.5) * resolution
    ys = (idx + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)          # gx varies along columns
    return gx, gy                          # each (n, n)


def _splat_soft(centers_xy: np.ndarray, radius: float, resolution: float,
                sigma: float = DEFAULT_SOFT_SIGMA) -> np.ndarray:
    """
    Soft occupancy from K disk centers (radius + Gaussian falloff).
    centers_xy: (K, 2). Returns (n, n) grid in [0, 1], max-combined.
    """
    n = grid_size(resolution)
    if len(centers_xy) == 0:
        return np.zeros((n, n))
    gx, gy = _cell_centers(resolution)                     # (n, n)
    grid = np.zeros((n, n))
    for cx, cy in centers_xy:
        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) - radius
        soft = np.exp(-np.clip(d, 0, None) ** 2 / (2 * sigma ** 2))
        soft[d <= 0] = 1.0
        grid = np.maximum(grid, soft)
    return grid


def rasterize_obstacles(obst_pos: np.ndarray,
                        resolution: float = DEFAULT_RES) -> np.ndarray:
    """Ground-truth binary-ish occupancy grid from true obstacle centers."""
    return _splat_soft(obst_pos, OBST_RADIUS, resolution, sigma=0.15)


def rasterize_predicted(pred_pos: np.ndarray,
                        resolution: float = DEFAULT_RES) -> np.ndarray:
    """Occupancy grid from ONE imagined future's obstacle positions (K, 2)."""
    return _splat_soft(pred_pos, OBST_RADIUS, resolution)


def rasterize_ensemble(pred_pos_members: np.ndarray,
                       resolution: float = DEFAULT_RES,
                       member_sigma: float = 0.15) -> np.ndarray:
    """
    Occupancy grid from an ENSEMBLE of imagined futures.
    pred_pos_members: (M, K, 2) — M members, K obstacles, one future step.
    Cells get higher occupancy where more members agree an obstacle is
    present; disagreement between members naturally spreads/softens the
    grid — but each member is splatted with the SAME tight sigma used
    for the ground-truth grid (member_sigma, default matches
    rasterize_obstacles). This matters: giving individual members a
    wider blur than the ground-truth disk would inflate every predicted
    footprint regardless of accuracy, penalizing IoU even when the
    ensemble is well localized. With matched sigma, a confident
    (agreeing) ensemble looks like a single tight disk — exactly like
    ground truth — and grids only widen where members genuinely
    disagree, which is the uncertainty signal we actually want.
    """
    n = grid_size(resolution)
    acc = np.zeros((n, n))
    for members_at_t in pred_pos_members:
        acc += _splat_soft(members_at_t, OBST_RADIUS, resolution,
                           sigma=member_sigma)
    return np.clip(acc / len(pred_pos_members), 0.0, 1.0)


# ====================================================================== #
def occupancy_iou(pred: np.ndarray, truth: np.ndarray,
                  thresh: float = 0.5) -> float:
    """Hard IoU after thresholding both grids."""
    p = pred >= thresh
    t = truth >= thresh
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)


def occupancy_soft_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Soft (fuzzy) IoU: sum(min)/sum(max), no thresholding — the metric
    used for Test 6, since it credits *calibrated* partial occupancy
    rather than forcing a hard decision boundary."""
    inter = np.minimum(pred, truth).sum()
    union = np.maximum(pred, truth).sum()
    if union < 1e-9:
        return 1.0
    return float(inter / union)


# ====================================================================== #
def trajectory_occupancy_cost(traj_xy: np.ndarray, occ_grid: np.ndarray,
                              resolution: float = DEFAULT_RES,
                              weight: float = 60.0) -> np.ndarray:
    """
    Cost contribution from reading predicted occupancy under imagined
    robot positions — a drop-in additional term for the MPPI cost.

    traj_xy  : (H, N, 2) imagined robot (x, y) positions
    occ_grid : (n, n) predicted occupancy for this horizon step (or a
               single grid reused across H if called per-step)
    returns  : (N,) cost contribution, summed over H if traj_xy has H>1
    """
    idx = world_to_grid(traj_xy, resolution)          # (H, N, 2) -> row,col
    row, col = idx[..., 0], idx[..., 1]
    occ = occ_grid[row, col]                          # (H, N)
    # small extra penalty for grazing the robot's own footprint radius
    return weight * occ.sum(axis=0)

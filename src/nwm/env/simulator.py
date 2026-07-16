"""
2D dynamic-world robot simulator.

Unicycle robot navigating an arena with moving circular obstacles.
Obstacles follow noisy curved trajectories (constant speed + per-obstacle
turn rate), which makes a *learned* motion predictor meaningfully better
than a constant-velocity baseline.

State conventions
-----------------
robot : [x, y, theta]
action: [v, w]  (linear velocity, angular velocity)
obstacle i: position p_i (2,), velocity v_i (2,), turn rate k_i (scalar)
"""

from __future__ import annotations

import numpy as np

ARENA = 20.0          # arena is [0, ARENA] x [0, ARENA]
DT = 0.1              # simulation timestep [s]
ROBOT_RADIUS = 0.30
OBST_RADIUS = 0.40
V_MAX, W_MAX = 1.2, 1.5
GOAL_TOL = 0.5


class DynamicWorld:
    """Simple, fully deterministic-given-seed dynamic environment."""

    def __init__(self,
                 n_obstacles: int = 6,
                 obst_speed_range: tuple = (0.30, 0.60),
                 obst_turn_range: tuple = (-0.35, 0.35),
                 process_noise: float = 0.01,
                 seed: int | None = None):
        self.n_obstacles = n_obstacles
        self.obst_speed_range = obst_speed_range
        self.obst_turn_range = obst_turn_range
        self.process_noise = process_noise
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        rng = self.rng
        # robot starts bottom-left region, goal top-right region
        self.robot = np.array([
            rng.uniform(1.5, 4.0),
            rng.uniform(1.5, 4.0),
            rng.uniform(-np.pi, np.pi),
        ])
        self.goal = np.array([
            rng.uniform(ARENA - 4.0, ARENA - 1.5),
            rng.uniform(ARENA - 4.0, ARENA - 1.5),
        ])

        # obstacles: keep initial positions away from robot and goal
        pos, vel, turn = [], [], []
        while len(pos) < self.n_obstacles:
            p = rng.uniform(2.0, ARENA - 2.0, size=2)
            if (np.linalg.norm(p - self.robot[:2]) < 2.5
                    or np.linalg.norm(p - self.goal) < 2.0):
                continue
            speed = rng.uniform(*self.obst_speed_range)
            ang = rng.uniform(-np.pi, np.pi)
            pos.append(p)
            vel.append(speed * np.array([np.cos(ang), np.sin(ang)]))
            turn.append(rng.uniform(*self.obst_turn_range))
        self.obst_pos = np.array(pos)          # (K, 2)
        self.obst_vel = np.array(vel)          # (K, 2)
        self.obst_turn = np.array(turn)        # (K,)
        self.t = 0
        return self.observe()

    # ------------------------------------------------------------------ #
    @staticmethod
    def robot_step(robot: np.ndarray, action: np.ndarray, dt: float = DT):
        """Ground-truth unicycle dynamics (also used by physics planners)."""
        v = float(np.clip(action[0], 0.0, V_MAX))
        w = float(np.clip(action[1], -W_MAX, W_MAX))
        x, y, th = robot
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = np.arctan2(np.sin(th + w * dt), np.cos(th + w * dt))
        return np.array([x, y, th])

    def _obstacles_step(self):
        """Curved obstacle motion with wall bounce + small process noise."""
        c, s = np.cos(self.obst_turn * DT), np.sin(self.obst_turn * DT)
        vx, vy = self.obst_vel[:, 0].copy(), self.obst_vel[:, 1].copy()
        self.obst_vel[:, 0] = c * vx - s * vy
        self.obst_vel[:, 1] = s * vx + c * vy
        self.obst_vel += self.rng.normal(0.0, self.process_noise,
                                         self.obst_vel.shape)
        self.obst_pos += self.obst_vel * DT
        # bounce off arena walls
        for d in range(2):
            low = self.obst_pos[:, d] < OBST_RADIUS
            high = self.obst_pos[:, d] > ARENA - OBST_RADIUS
            self.obst_vel[low | high, d] *= -1.0
            self.obst_pos[:, d] = np.clip(self.obst_pos[:, d],
                                          OBST_RADIUS, ARENA - OBST_RADIUS)

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray):
        self.robot = self.robot_step(self.robot, action)
        self.robot[0] = np.clip(self.robot[0], 0.0, ARENA)
        self.robot[1] = np.clip(self.robot[1], 0.0, ARENA)
        self._obstacles_step()
        self.t += 1

        dists = np.linalg.norm(self.obst_pos - self.robot[:2], axis=1)
        collided = bool(np.any(dists < ROBOT_RADIUS + OBST_RADIUS))
        reached = bool(np.linalg.norm(self.goal - self.robot[:2]) < GOAL_TOL)
        return self.observe(), collided, reached

    # ------------------------------------------------------------------ #
    def observe(self) -> dict:
        return {
            "robot": self.robot.copy(),
            "goal": self.goal.copy(),
            "obst_pos": self.obst_pos.copy(),
            "obst_vel": self.obst_vel.copy(),
            "t": self.t,
        }

    # ------------------------------------------------------------------ #
    def lidar(self, n_beams: int = 24, max_range: float = 6.0) -> np.ndarray:
        """Simple ray-cast range sensor against circular obstacles + walls."""
        angles = self.robot[2] + np.linspace(-np.pi, np.pi, n_beams,
                                             endpoint=False)
        ranges = np.full(n_beams, max_range)
        o = self.robot[:2]
        for i, a in enumerate(angles):
            d = np.array([np.cos(a), np.sin(a)])
            # circles
            for p in self.obst_pos:
                oc = p - o
                proj = oc @ d
                if proj <= 0:
                    continue
                perp2 = oc @ oc - proj * proj
                r2 = OBST_RADIUS ** 2
                if perp2 < r2:
                    hit = proj - np.sqrt(r2 - perp2)
                    if 0 < hit < ranges[i]:
                        ranges[i] = hit
            # walls
            for axis, bound in ((0, 0.0), (0, ARENA), (1, 0.0), (1, ARENA)):
                if abs(d[axis]) > 1e-9:
                    tt = (bound - o[axis]) / d[axis]
                    if 0 < tt < ranges[i]:
                        ranges[i] = tt
        return ranges

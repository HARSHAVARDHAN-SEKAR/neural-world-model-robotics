"""
Stage 2 — collect episode-indexed sequence data for Transformer training.

The CPU pipeline's `datasets/robot_experience.npz` is flat (concatenated
transitions, no episode boundaries) because the MLP models don't need
sequence structure. The Transformer does — it needs full per-episode
(robot, action, obstacle) sequences to build context/future windows.
This script re-runs the same exploration policy as
`scripts/run_pipeline.py::collect_experience` but preserves episode
structure and saves it as ragged (object-array) .npz.

Run: python scripts/collect_sequence_data.py [--episodes 60] [--steps 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nwm.env.simulator import DynamicWorld, V_MAX, W_MAX  # noqa: E402

TRAIN_ENV = dict(n_obstacles=6, obst_speed_range=(0.30, 0.60),
                 obst_turn_range=(-0.50, 0.50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "datasets" / "sequences.npz"))
    args = ap.parse_args()

    rng = np.random.default_rng(123)
    robot_seqs, action_seqs, obst_seqs = [], [], []
    for ep in range(args.episodes):
        env = DynamicWorld(seed=200 + ep, **TRAIN_ENV)
        obs = env.observe()
        robot, actions, obst = [obs["robot"].copy()], [], [obs["obst_pos"].copy()]
        a = np.array([0.6, 0.0])
        for _ in range(args.steps):
            if rng.random() < 0.25:
                a = np.array([rng.uniform(0.0, V_MAX), rng.uniform(-W_MAX, W_MAX)])
            obs, _, _ = env.step(a)
            robot.append(obs["robot"].copy())
            actions.append(a.copy())
            obst.append(obs["obst_pos"].copy())
        robot_seqs.append(np.asarray(robot[:-1]))     # align lengths: T states
        action_seqs.append(np.asarray(actions))       # T actions
        obst_seqs.append(np.asarray(obst))             # T+1 obstacle snapshots
        if (ep + 1) % 10 == 0:
            print(f"[collect_sequence_data] {ep+1}/{args.episodes} episodes")

    np.savez_compressed(
        args.out,
        robot=np.array(robot_seqs, dtype=object),
        actions=np.array(action_seqs, dtype=object),
        obst=np.array(obst_seqs, dtype=object),
    )
    print(f"[collect_sequence_data] wrote {args.out} "
         f"({args.episodes} episodes x {args.steps} steps)")


if __name__ == "__main__":
    main()

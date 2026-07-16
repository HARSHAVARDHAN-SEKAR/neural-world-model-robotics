"""
Full research pipeline (single command, CPU-only, ~2-4 min):

  1. Collect robot experience in the dynamic simulator
  2. Train the neural world model (robot dynamics + obstacle motion)
  3. Test 1 — dynamic obstacle prediction (ADE / FDE vs constant-velocity)
  4. Test 2 — planning benchmark (Reactive vs DWA vs MPPI vs Neural-MPC)
  5. Test 3 — generalization to an unseen, harder environment
  6. Save all figures to assets/ and metrics to results/

Run:  python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nwm.env.simulator import DT, ARENA, DynamicWorld           # noqa: E402
from nwm.models.world_model import (HISTORY, ObstacleMotionModel,  # noqa: E402
                                    RobotDynamicsModel,
                                    constant_velocity_rollout,
                                    save_models)
from nwm.planners.planners import (DWAPlanner, MPPIPlanner,      # noqa: E402
                                   NeuralMPCPlanner,
                                   ReactivePlanner, run_episode)

ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
DATASETS = ROOT / "datasets"
for p in (ASSETS, RESULTS, DATASETS):
    p.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

TRAIN_ENV = dict(n_obstacles=6, obst_speed_range=(0.30, 0.60),
                 obst_turn_range=(-0.50, 0.50))
OOD_ENV = dict(n_obstacles=9, obst_speed_range=(0.60, 0.95),
               obst_turn_range=(-0.70, 0.70))


# ====================================================================== #
def collect_experience(n_episodes=25, steps=220, seed0=100):
    """Random-ish exploration policy gathering (s, a, s') + obstacle tracks."""
    S, A, SN, tracks = [], [], [], []
    rng = np.random.default_rng(7)
    for ep in range(n_episodes):
        env = DynamicWorld(seed=seed0 + ep, **TRAIN_ENV)
        obs = env.observe()
        track = [obs["obst_pos"].copy()]
        a = np.array([0.6, 0.0])
        for _ in range(steps):
            if rng.random() < 0.25:                      # resample action
                a = np.array([rng.uniform(0.0, 1.2),
                              rng.uniform(-1.5, 1.5)])
            s = obs["robot"].copy()
            obs, _, _ = env.step(a)
            S.append(s); A.append(a.copy()); SN.append(obs["robot"].copy())
            track.append(obs["obst_pos"].copy())
        tracks.append(np.asarray(track))
    S, A, SN = map(np.asarray, (S, A, SN))
    np.savez_compressed(DATASETS / "robot_experience.npz",
                        states=S, actions=A, next_states=SN)
    print(f"[data] {len(S)} transitions, {len(tracks)} episodes")
    return S, A, SN, tracks


# ====================================================================== #
def train_world_model(S, A, SN, tracks):
    t0 = time.time()
    robot_model = RobotDynamicsModel(seed=0)
    mse_r = robot_model.fit(S, A, SN)

    Xs, Ys = [], []
    for tr in tracks:
        X, Y = ObstacleMotionModel.make_dataset(tr)
        Xs.append(X); Ys.append(Y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    obst_model = ObstacleMotionModel(seed=0)
    mse_o = obst_model.fit(X, Y)

    save_models(DATASETS / "world_model.pkl", robot_model, obst_model)
    print(f"[train] robot-dyn MSE {mse_r:.2e} | obstacle MSE {mse_o:.2e} "
          f"| {time.time()-t0:.1f}s | {len(X)} obstacle samples")
    return robot_model, obst_model


# ====================================================================== #
def test1_prediction(obst_model, horizon=30, n_eval=40, env_cfg=TRAIN_ENV,
                     tag="in-distribution"):
    """ADE/FDE of learned predictor vs constant-velocity, |horizon| steps."""
    ade_nn, fde_nn, ade_cv, fde_cv = [], [], [], []
    sample = None
    for i in range(n_eval):
        env = DynamicWorld(seed=5000 + i, **env_cfg)
        track = [env.observe()["obst_pos"].copy()]
        for _ in range(HISTORY + horizon):
            obs, _, _ = env.step(np.array([0.0, 0.0]))
            track.append(obs["obst_pos"].copy())
        track = np.asarray(track)                        # (T, K, 2)
        hist, future = track[:HISTORY + 1], track[HISTORY + 1:]
        p_nn = obst_model.rollout(hist, horizon)
        p_cv = constant_velocity_rollout(hist, horizon)
        e_nn = np.linalg.norm(p_nn - future, axis=-1)    # (H, K)
        e_cv = np.linalg.norm(p_cv - future, axis=-1)
        ade_nn.append(e_nn.mean()); fde_nn.append(e_nn[-1].mean())
        ade_cv.append(e_cv.mean()); fde_cv.append(e_cv[-1].mean())
        if i == 0:
            sample = (hist, future, p_nn, p_cv)
    out = {"ADE_neural": float(np.mean(ade_nn)),
           "FDE_neural": float(np.mean(fde_nn)),
           "ADE_const_vel": float(np.mean(ade_cv)),
           "FDE_const_vel": float(np.mean(fde_cv)),
           "horizon_s": horizon * DT, "episodes": n_eval, "env": tag}
    print(f"[test1:{tag}] ADE nn {out['ADE_neural']:.3f} vs cv "
          f"{out['ADE_const_vel']:.3f} | FDE nn {out['FDE_neural']:.3f} "
          f"vs cv {out['FDE_const_vel']:.3f}  ({horizon*DT:.0f}s horizon)")
    return out, sample


def plot_prediction(sample):
    hist, future, p_nn, p_cv = sample
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    K = future.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for k in range(K):
        c = colors[k]
        ax.plot(hist[:, k, 0], hist[:, k, 1], "-", color=c, lw=1.2, alpha=0.5)
        ax.plot(future[:, k, 0], future[:, k, 1], "-", color=c, lw=2,
                label="ground truth" if k == 0 else None)
        ax.plot(p_nn[:, k, 0], p_nn[:, k, 1], "--", color=c, lw=1.6,
                label="neural world model" if k == 0 else None)
        ax.plot(p_cv[:, k, 0], p_cv[:, k, 1], ":", color=c, lw=1.6,
                label="constant velocity" if k == 0 else None)
        ax.plot(*hist[-1, k], "o", color=c, ms=5)
    ax.set(title="Test 1 — Imagined obstacle futures (3 s horizon)",
           xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "prediction_vs_groundtruth.png")
    plt.close(fig)


# ====================================================================== #
def make_planners(robot_model, obst_model, seed=0):
    return [ReactivePlanner(),
            DWAPlanner(),
            MPPIPlanner(seed=seed),
            NeuralMPCPlanner(robot_model, obst_model, seed=seed)]


def test2_planning(robot_model, obst_model, n_eps=30, env_cfg=TRAIN_ENV,
                   tag="in-distribution", seed0=9000):
    rows = {}
    demo_traj = {}
    for planner in make_planners(robot_model, obst_model):
        m = {k: [] for k in ("success", "collision", "steps",
                             "path_length", "smoothness", "energy")}
        t0 = time.time()
        for i in range(n_eps):
            env = DynamicWorld(seed=seed0 + i, **env_cfg)
            r = run_episode(env, planner)
            for k in m:
                m[k].append(float(r[k]))
            if i == 2:
                demo_traj[planner.name] = r
        rows[planner.name] = {
            "success_rate": float(np.mean(m["success"])),
            "collision_rate": float(np.mean(m["collision"])),
            "avg_steps": float(np.mean(m["steps"])),
            "avg_path_length": float(np.mean(m["path_length"])),
            "avg_smoothness": float(np.mean(m["smoothness"])),
            "avg_energy": float(np.mean(m["energy"])),
        }
        print(f"[test2:{tag}] {planner.name:<11} "
              f"success {rows[planner.name]['success_rate']*100:5.1f}%  "
              f"collision {rows[planner.name]['collision_rate']*100:5.1f}%  "
              f"({time.time()-t0:.0f}s)")
    return rows, demo_traj


def plot_planning(rows, tag, fname):
    names = list(rows)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    metrics = [("success_rate", "Success rate", 100, "%"),
               ("collision_rate", "Collision rate", 100, "%"),
               ("avg_smoothness", "Control smoothness cost", 1, ""),
               ("avg_energy", "Energy [a.u.]", 1, "")]
    palette = ["#9aa0a6", "#7aa6c2", "#e2a24b", "#4caf7d"]
    for ax, (key, title, scale, unit) in zip(axes, metrics):
        vals = [rows[n][key] * scale for n in names]
        bars = ax.bar(names, vals, color=palette)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}{unit}",
                    ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.18)
    fig.suptitle(f"Test 2 — Planner benchmark ({tag})", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / fname, bbox_inches="tight")
    plt.close(fig)


def plot_episode(demo_traj, fname):
    fig, axes = plt.subplots(1, len(demo_traj), figsize=(3.3 * len(demo_traj), 3.6))
    for ax, (name, r) in zip(np.atleast_1d(axes), demo_traj.items()):
        ot = r["obst_traj"]
        for k in range(ot.shape[1]):
            ax.plot(ot[:, k, 0], ot[:, k, 1], "-", color="tomato",
                    alpha=0.35, lw=1)
            ax.plot(*ot[-1, k], "o", color="tomato", ms=7)
        rt = r["robot_traj"]
        ax.plot(rt[:, 0], rt[:, 1], "-", color="#1565c0", lw=2)
        ax.plot(*rt[0], "s", color="#1565c0", ms=7)
        ax.plot(*r["goal"], "*", color="green", ms=13)
        status = "success" if r["success"] else ("collision" if r["collision"]
                                                 else "timeout")
        ax.set(title=f"{name} — {status}", xlim=(0, ARENA), ylim=(0, ARENA),
               aspect="equal", xticks=[], yticks=[])
    fig.suptitle("Same scenario, four planners (blue robot, red moving obstacles)")
    fig.tight_layout()
    fig.savefig(ASSETS / fname, bbox_inches="tight")
    plt.close(fig)


# ====================================================================== #
def plot_generalization(pred_in, pred_ood, plan_in, plan_ood):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w / 2, [pred_in["ADE_neural"], pred_ood["ADE_neural"]],
                w, label="Neural world model", color="#4caf7d")
    axes[0].bar(x + w / 2, [pred_in["ADE_const_vel"], pred_ood["ADE_const_vel"]],
                w, label="Constant velocity", color="#9aa0a6")
    axes[0].set_xticks(x, ["train env", "unseen harder env"])
    axes[0].set_title("Prediction ADE [m] (lower = better)")
    axes[0].legend(fontsize=8)

    names = list(plan_in)
    x = np.arange(len(names))
    axes[1].bar(x - w / 2, [plan_in[n]["success_rate"] * 100 for n in names],
                w, label="train env", color="#7aa6c2")
    axes[1].bar(x + w / 2, [plan_ood[n]["success_rate"] * 100 for n in names],
                w, label="unseen harder env", color="#e2a24b")
    axes[1].set_xticks(x, names, rotation=15)
    axes[1].set_title("Planner success rate [%]")
    axes[1].legend(fontsize=8)
    fig.suptitle("Test 3 — Generalization to an unseen environment", y=1.03)
    fig.tight_layout()
    fig.savefig(ASSETS / "generalization.png", bbox_inches="tight")
    plt.close(fig)


# ====================================================================== #
def main():
    t0 = time.time()
    print("=" * 66)
    print("Neural World Model & Predictive Robot Intelligence — pipeline")
    print("=" * 66)

    S, A, SN, tracks = collect_experience()
    robot_model, obst_model = train_world_model(S, A, SN, tracks)

    pred_in, sample = test1_prediction(obst_model)
    plot_prediction(sample)

    plan_in, demo = test2_planning(robot_model, obst_model)
    plot_planning(plan_in, "training environment", "benchmark_planning.png")
    plot_episode(demo, "episode_trajectories.png")

    pred_ood, _ = test1_prediction(obst_model, env_cfg=OOD_ENV,
                                   tag="unseen-harder")
    plan_ood, _ = test2_planning(robot_model, obst_model, n_eps=30,
                                 env_cfg=OOD_ENV, tag="unseen-harder",
                                 seed0=42000)
    plot_planning(plan_ood, "unseen harder environment",
                  "benchmark_planning_ood.png")
    plot_generalization(pred_in, pred_ood, plan_in, plan_ood)

    results = {"test1_prediction": {"in_distribution": pred_in,
                                    "out_of_distribution": pred_ood},
               "test2_planning": {"in_distribution": plan_in,
                                  "out_of_distribution": plan_ood},
               "runtime_s": round(time.time() - t0, 1)}
    with open(RESULTS / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] total {results['runtime_s']}s — figures in assets/, "
          f"metrics in results/benchmark_results.json")


if __name__ == "__main__":
    main()

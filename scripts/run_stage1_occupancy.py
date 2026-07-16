"""
Stage 1 pipeline — occupancy-grid prediction & planning (Test 6).

  Test 6a: occupancy-IoU of the ensemble's predicted grid vs the
           true future occupancy grid, over a 1.5 s horizon, compared
           against a constant-velocity occupancy baseline.
  Test 6b: does planning against occupancy (Occupancy-MPC) change
           outcomes vs point-based Neural-MPC / Risk-MPC, in the same
           stress environment used for Tests 4-5?

Run: python scripts/run_stage1_occupancy.py <1|2|3>
  1 -> Test 6a (occupancy IoU) + qualitative grid figure
  2 -> Test 6b (planner benchmark, stress env)
  3 -> figures + merge into results/benchmark_results.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nwm.env.simulator import DT, DynamicWorld                       # noqa: E402
from nwm.env.occupancy import (DEFAULT_RES, occupancy_iou,           # noqa: E402
                               occupancy_soft_iou, rasterize_ensemble,
                               rasterize_obstacles)
from nwm.models.world_model import HISTORY, load_models              # noqa: E402
from nwm.planners.planners import (NeuralMPCPlanner, OccupancyNeuralMPC,  # noqa: E402
                                   RiskAwareNeuralMPC, run_episode)

ASSETS, RESULTS, DATASETS = ROOT / "assets", ROOT / "results", ROOT / "datasets"
STATE = RESULTS / "_stage1_state.pkl"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

STRESS_ENV = dict(n_obstacles=12, obst_speed_range=(0.85, 1.25),
                  obst_turn_range=(-0.90, 0.90), process_noise=0.03)
HORIZON = 15  # 1.5 s @ 10 Hz — occupancy grids are coarser/costlier than points


# ====================================================================== #
def stage1():
    """Test 6a: occupancy prediction quality."""
    t0 = time.time()
    with open(DATASETS / "ensemble_model.pkl", "rb") as f:
        ens = pickle.load(f)

    ious, soft_ious, ious_cv = [], [], []
    fan = None
    for i in range(30):
        env = DynamicWorld(seed=31000 + i, **STRESS_ENV)
        tr = [env.observe()["obst_pos"].copy()]
        for _ in range(HISTORY + HORIZON):
            obs, _, _ = env.step(np.array([0.0, 0.0]))
            tr.append(obs["obst_pos"].copy())
        tr = np.asarray(tr)
        hist, future = tr[:HISTORY + 1], tr[HISTORY + 1:]

        all_futures = ens.rollout_all(hist, HORIZON)          # (M, H, K, 2)
        # constant-velocity occupancy baseline (single point per step)
        v = (hist[-1] - hist[-2]) / DT
        cv_future = hist[-1][None] + v[None] * np.arange(1, HORIZON + 1)[:, None, None] * DT

        for h in (4, 9, 14):  # 0.5s, 1.0s, 1.5s checkpoints
            occ_pred = rasterize_ensemble(all_futures[:, h])
            occ_true = rasterize_obstacles(future[h])
            occ_cv = rasterize_obstacles(cv_future[h])
            ious.append(occupancy_iou(occ_pred, occ_true))
            soft_ious.append(occupancy_soft_iou(occ_pred, occ_true))
            ious_cv.append(occupancy_iou(occ_cv, occ_true))
        if i == 0:
            fan = (hist, future, all_futures, cv_future)

    out = {"IoU_neural_ensemble": float(np.mean(ious)),
           "soft_IoU_neural_ensemble": float(np.mean(soft_ious)),
           "IoU_const_velocity": float(np.mean(ious_cv)),
           "checkpoints_s": [0.5, 1.0, 1.5], "episodes": 30}
    print(f"[test6a] occupancy IoU: neural {out['IoU_neural_ensemble']:.3f} "
          f"vs const-vel {out['IoU_const_velocity']:.3f}  "
          f"(soft-IoU {out['soft_IoU_neural_ensemble']:.3f})  "
          f"[{time.time()-t0:.0f}s]")

    with open(STATE, "wb") as f:
        pickle.dump({"test6a": out, "fan": fan}, f)
    return out


def plot_occupancy_grids(fan):
    hist, future, all_futures, cv_future = fan
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    steps = [(4, "0.5 s"), (9, "1.0 s"), (14, "1.5 s")]
    for ax, (h, label) in zip(axes[:3], steps):
        occ_pred = rasterize_ensemble(all_futures[:, h])
        occ_true = rasterize_obstacles(future[h])
        ax.imshow(occ_pred, origin="lower", cmap="Reds", vmin=0, vmax=1,
                  extent=(0, 20, 0, 20), alpha=0.85)
        ys, xs = np.where(occ_true > 0.5)
        ax.scatter((xs + 0.5) * DEFAULT_RES, (ys + 0.5) * DEFAULT_RES,
                   s=6, c="blue", marker="x", label="ground truth"
                   if h == 4 else None)
        ax.set_title(f"Predicted occupancy @ {label}\n(blue x = true obstacle cells)",
                    fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[3].axis("off")
    axes[3].text(0, 0.5, "Red intensity = ensemble-predicted\noccupancy "
                "probability.\nWider/softer red = more member\n"
                "disagreement = higher uncertainty.\n\nThis IS the "
                "uncertainty map —\nno separate disagreement\nscalar "
                "needed.", fontsize=8.5, va="center")
    fig.suptitle("Test 6a — Predicted occupancy grid vs ground truth "
                 "(stress environment)", y=1.03)
    fig.tight_layout()
    fig.savefig(ASSETS / "occupancy_prediction.png", bbox_inches="tight")
    plt.close(fig)


# ====================================================================== #
def stage2(n_eps=20):
    """Test 6b: occupancy-aware planning vs point-based planning."""
    robot_model, _ = load_models(DATASETS / "world_model.pkl")
    with open(DATASETS / "ensemble_model.pkl", "rb") as f:
        ens = pickle.load(f)

    rows = {}
    planners = [
        NeuralMPCPlanner(robot_model, ens, horizon=HORIZON, seed=0),
        RiskAwareNeuralMPC(robot_model, ens, horizon=HORIZON, risk_q=0.4, seed=0),
        OccupancyNeuralMPC(robot_model, ens, horizon=HORIZON, seed=0),
    ]
    for planner in planners:
        m = {k: [] for k in ("success", "collision", "energy")}
        t0 = time.time()
        for i in range(n_eps):
            env = DynamicWorld(seed=51000 + i, **STRESS_ENV)
            r = run_episode(env, planner)
            for k in m:
                m[k].append(float(r[k]))
        rows[planner.name] = {k + "_rate" if k != "energy" else k:
                              float(np.mean(v)) for k, v in m.items()}
        print(f"[test6b:stress] {planner.name:<14} "
              f"success {rows[planner.name]['success_rate']*100:5.1f}%  "
              f"collision {rows[planner.name]['collision_rate']*100:5.1f}%  "
              f"({time.time()-t0:.0f}s)")

    with open(STATE, "rb") as f:
        st = pickle.load(f)
    st["test6b"] = rows
    with open(STATE, "wb") as f:
        pickle.dump(st, f)
    return rows


# ====================================================================== #
def stage3():
    with open(STATE, "rb") as f:
        st = pickle.load(f)
    plot_occupancy_grids(st["fan"])

    raw = st.get("test6b_raw") or st.get("test6b")
    if "test6b_raw" in st:
        rows = {}
        for name, eps in raw.items():
            rows[name] = {
                "success_rate": float(np.mean([d["success"] for d in eps.values()])),
                "collision_rate": float(np.mean([d["collision"] for d in eps.values()])),
                "energy": float(np.mean([d["energy"] for d in eps.values()])),
                "episodes": len(eps)}
    else:
        rows = raw
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, key, title in zip(axes, ("success_rate", "collision_rate"),
                              ("Success rate", "Collision rate")):
        vals = [rows[n][key] * 100 for n in names]
        bars = ax.bar(names, vals, color=["#9aa0a6", "#7aa6c2", "#4caf7d"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=10)
        ax.margins(y=0.2)
    fig.suptitle("Test 6b — Point vs occupancy-grid planning (stress env)", y=1.04)
    fig.tight_layout()
    fig.savefig(ASSETS / "occupancy_planning.png", bbox_inches="tight")
    plt.close(fig)

    with open(RESULTS / "benchmark_results.json") as f:
        results = json.load(f)
    results["test6_occupancy"] = {"prediction": st["test6a"],
                                  "planning_stress": rows}
    with open(RESULTS / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    STATE.unlink(missing_ok=True)
    print("[done] Stage 1 occupancy results written")


if __name__ == "__main__":
    {1: stage1, 2: stage2, 3: stage3}[int(sys.argv[1])]()

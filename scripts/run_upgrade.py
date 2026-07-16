"""
Upgrade #2 pipeline — uncertainty-aware world model + risk-aware planning.

Stages (resumable):
  1  train 5-member bootstrapped ensemble on the existing experience;
     Test 4: is disagreement a *calibrated* uncertainty signal?
     (does high ensemble disagreement predict high actual error?)
  2  Test 5: STRESS benchmark — 12 fast, noisy, sharp-turning obstacles.
     Deterministic Neural-MPC vs Risk-MPC (CVaR over 5 imagined futures).
  3  figures + results JSON merge.

Run:  python scripts/run_upgrade.py <1|2|3>
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

from nwm.env.simulator import DT, DynamicWorld                      # noqa: E402
from nwm.models.ensemble import EnsembleObstacleModel               # noqa: E402
from nwm.models.world_model import HISTORY, ObstacleMotionModel, load_models  # noqa: E402
from nwm.planners.planners import (NeuralMPCPlanner,                # noqa: E402
                                   RiskAwareNeuralMPC, run_episode)

ASSETS, RESULTS, DATASETS = ROOT / "assets", ROOT / "results", ROOT / "datasets"
STATE = RESULTS / "_upgrade_state.pkl"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

TRAIN_ENV = dict(n_obstacles=6, obst_speed_range=(0.30, 0.60),
                 obst_turn_range=(-0.50, 0.50))
STRESS_ENV = dict(n_obstacles=12, obst_speed_range=(0.85, 1.25),
                  obst_turn_range=(-0.90, 0.90), process_noise=0.03)


# ====================================================================== #
def stage1():
    t0 = time.time()
    # rebuild the same training tracks used for the base model
    tracks = []
    for ep in range(25):
        env = DynamicWorld(seed=100 + ep, **TRAIN_ENV)
        tr = [env.observe()["obst_pos"].copy()]
        for _ in range(220):
            obs, _, _ = env.step(np.array([0.0, 0.0]))
            tr.append(obs["obst_pos"].copy())
        tracks.append(np.asarray(tr))
    Xs, Ys = [], []
    for tr in tracks:
        X, Y = ObstacleMotionModel.make_dataset(tr)
        Xs.append(X); Ys.append(Y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)

    ens = EnsembleObstacleModel(n_members=5, seed=0)
    mse = ens.fit(X, Y)
    with open(DATASETS / "ensemble_model.pkl", "wb") as f:
        pickle.dump(ens, f)
    print(f"[upgrade] ensemble trained (5 members, bootstrap) "
          f"MSE {mse:.2e} in {time.time()-t0:.0f}s")

    # ---- Test 4: uncertainty calibration -------------------------------
    horizon = 30
    dis, err = [], []
    fan_sample = None
    for i in range(60):
        env = DynamicWorld(seed=20000 + i, **STRESS_ENV)
        tr = [env.observe()["obst_pos"].copy()]
        for _ in range(HISTORY + horizon):
            obs, _, _ = env.step(np.array([0.0, 0.0]))
            tr.append(obs["obst_pos"].copy())
        tr = np.asarray(tr)
        hist, fut = tr[:HISTORY + 1], tr[HISTORY + 1:]
        futures = ens.rollout_all(hist, horizon)            # (M, H, K, 2)
        mean_pred = futures.mean(axis=0)
        d = EnsembleObstacleModel.disagreement(futures)     # (H, K)
        e = np.linalg.norm(mean_pred - fut, axis=-1)        # (H, K)
        dis.extend(d.mean(axis=0)); err.extend(e.mean(axis=0))
        if i == 0:
            fan_sample = (hist, fut, futures)
    dis, err = np.asarray(dis), np.asarray(err)
    corr = float(np.corrcoef(dis, err)[0, 1])
    print(f"[test4] disagreement↔error correlation r = {corr:.2f} "
          f"({len(dis)} obstacle tracks, stress env)")

    with open(STATE, "wb") as f:
        pickle.dump({"corr": corr, "dis": dis, "err": err,
                     "fan": fan_sample}, f)


# ====================================================================== #
def stage2(n_eps=25):
    robot_model, _ = load_models(DATASETS / "world_model.pkl")
    with open(DATASETS / "ensemble_model.pkl", "rb") as f:
        ens = pickle.load(f)
    with open(STATE, "rb") as f:
        st = pickle.load(f)

    rows = {}
    for planner in (NeuralMPCPlanner(robot_model, ens, seed=0),
                    RiskAwareNeuralMPC(robot_model, ens, risk_q=0.4, seed=0)):
        m = {k: [] for k in ("success", "collision", "steps", "energy")}
        t0 = time.time()
        for i in range(n_eps):
            env = DynamicWorld(seed=77000 + i, **STRESS_ENV)
            r = run_episode(env, planner)
            for k in m:
                m[k].append(float(r[k]))
        rows[planner.name] = {k: float(np.mean(v)) for k, v in m.items()}
        rows[planner.name]["success_rate"] = rows[planner.name].pop("success")
        rows[planner.name]["collision_rate"] = rows[planner.name].pop("collision")
        print(f"[test5:stress] {planner.name:<11} "
              f"success {rows[planner.name]['success_rate']*100:5.1f}%  "
              f"collision {rows[planner.name]['collision_rate']*100:5.1f}%  "
              f"({time.time()-t0:.0f}s)")
    st["stress"] = rows
    with open(STATE, "wb") as f:
        pickle.dump(st, f)


# ====================================================================== #
def stage3():
    with open(STATE, "rb") as f:
        st = pickle.load(f)

    # ---- figure: imagined futures fan + calibration --------------------
    hist, fut, futures = st["fan"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    K = fut.shape[1]
    show = np.argsort(np.linalg.norm(fut[-1] - hist[-1], axis=-1))[-4:]
    colors = plt.cm.tab10(np.linspace(0, 1, len(show)))
    for c, k in zip(colors, show):
        for m in range(futures.shape[0]):
            ax.plot(futures[m, :, k, 0], futures[m, :, k, 1], "-",
                    color=c, alpha=0.35, lw=1.1,
                    label="5 imagined futures (ensemble)"
                    if m == 0 and k == show[0] else None)
        ax.plot(fut[:, k, 0], fut[:, k, 1], "-", color=c, lw=2.4,
                label="ground truth" if k == show[0] else None)
        ax.plot(*hist[-1, k], "o", color=c, ms=6)
    ax.set(title="Multiple imagined futures per obstacle (stress env)",
           xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    dis, err = st["dis"], st["err"]
    ax.scatter(dis, err, s=14, alpha=0.5, color="#5c6bc0")
    b = np.polyfit(dis, err, 1)
    xs = np.linspace(dis.min(), dis.max(), 50)
    ax.plot(xs, np.polyval(b, xs), "-", color="#c62828", lw=2,
            label=f"fit (r = {st['corr']:.2f})")
    ax.set(title="Uncertainty is calibrated:\nensemble disagreement predicts actual error",
           xlabel="ensemble disagreement [m]",
           ylabel="actual prediction error [m]")
    ax.legend(fontsize=8)
    fig.suptitle("Test 4 — Epistemic uncertainty from the ensemble world model",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "uncertainty_ensemble.png", bbox_inches="tight")
    plt.close(fig)

    # ---- figure: stress benchmark --------------------------------------
    import numpy as _np
    order = [("Single-MPC", "Single model\n(1 future)"),
             ("Neural-MPC", "Ensemble mean\n(5 futures)"),
             ("Risk-MPC", "Ensemble CVaR\n(risk-aware)")]
    rows = {}
    for key, label in order:
        raw = st["stress_raw"][key]
        rows[label] = {
            "success_rate": float(_np.mean([d["success"] for d in raw.values()])),
            "collision_rate": float(_np.mean([d["collision"] for d in raw.values()])),
            "energy": float(_np.mean([d["energy"] for d in raw.values()])),
            "episodes": len(raw)}
    names = list(rows)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, (key, title, scale, unit) in zip(
            axes, [("success_rate", "Success rate", 100, "%"),
                   ("collision_rate", "Collision rate", 100, "%"),
                   ("energy", "Energy [a.u.]", 1, "")]):
        vals = [rows[n][key] * scale for n in names]
        bars = ax.bar(names, vals, color=["#9aa0a6", "#4caf7d", "#7aa6c2"])
        ax.tick_params(axis="x", labelsize=7)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.1f}{unit}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.margins(y=0.2)
    fig.suptitle("Test 5 — Stress environment (12 fast, noisy obstacles): "
                 "deterministic vs risk-aware imagination", y=1.04)
    fig.tight_layout()
    fig.savefig(ASSETS / "risk_aware_stress.png", bbox_inches="tight")
    plt.close(fig)

    # ---- merge results --------------------------------------------------
    with open(RESULTS / "benchmark_results.json") as f:
        results = json.load(f)
    results["test4_uncertainty"] = {
        "disagreement_error_correlation": st["corr"],
        "n_members": 5, "env": "stress"}
    results["test5_risk_aware_stress"] = rows
    with open(RESULTS / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    STATE.unlink(missing_ok=True)
    print("[done] upgrade figures + results written")


if __name__ == "__main__":
    {1: stage1, 2: stage2, 3: stage3}[int(sys.argv[1])]()

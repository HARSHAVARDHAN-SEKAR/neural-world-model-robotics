"""
Stage 2 — Test 7: Transformer world model vs MLP ensemble.

Head-to-head on the SAME episodes and metrics used elsewhere:
  7a  prediction ADE/FDE (3 s horizon) — transformer vs MLP ensemble
      vs constant-velocity
  7b  planning success/collision in the stress environment —
      Neural-MPC backed by the transformer vs backed by the MLP ensemble

Requires torch + a trained checkpoint (datasets/world_transformer.pt):
    python scripts/collect_sequence_data.py
    python scripts/train_world_transformer.py --epochs 60
    python scripts/run_stage2_benchmark.py 1     # Test 7a + figure
    python scripts/run_stage2_benchmark.py 2     # Test 7b (chunk via --i0/--i1)
    python scripts/run_stage2_benchmark.py 3     # merge + figure

Not run in CI (needs torch/GPU-scale deps).
"""

from __future__ import annotations

import argparse
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
from nwm.models.world_model import (HISTORY, constant_velocity_rollout,  # noqa: E402
                                    load_models)
from nwm.models.transformer_wrapper import TransformerWorldModelWrapper  # noqa: E402
from nwm.planners.planners import NeuralMPCPlanner, run_episode      # noqa: E402

ASSETS, RESULTS, DATASETS = ROOT / "assets", ROOT / "results", ROOT / "datasets"
STATE = RESULTS / "_stage2_state.pkl"
CKPT = DATASETS / "world_transformer.pt"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

TRAIN_ENV = dict(n_obstacles=6, obst_speed_range=(0.30, 0.60),
                 obst_turn_range=(-0.50, 0.50))
STRESS_ENV = dict(n_obstacles=12, obst_speed_range=(0.85, 1.25),
                  obst_turn_range=(-0.90, 0.90), process_noise=0.03)
# Test 7b planning env: matched to the transformer's fixed obstacle count
# (K=6). Slightly harder than nominal TRAIN_ENV to create planning
# pressure, but same K so both backends are directly comparable.
PLAN_ENV = dict(n_obstacles=6, obst_speed_range=(0.55, 0.85),
                obst_turn_range=(-0.70, 0.70), process_noise=0.02)


def _load_models():
    with open(DATASETS / "ensemble_model.pkl", "rb") as f:
        ens = pickle.load(f)
    return ens


# ====================================================================== #
def stage1(n_eval=40, horizon=30):
    """Test 7a — prediction ADE/FDE."""
    ens = _load_models()
    # transformer trained with n_obstacles=6 latent; eval in-distribution env
    tf = TransformerWorldModelWrapper(CKPT, n_obstacles=TRAIN_ENV["n_obstacles"])

    res = {k: [] for k in ("ADE_tf", "FDE_tf", "ADE_ens", "FDE_ens",
                           "ADE_cv", "FDE_cv")}
    for i in range(n_eval):
        env = DynamicWorld(seed=61000 + i, **TRAIN_ENV)
        tr = [env.observe()["obst_pos"].copy()]
        for _ in range(HISTORY + horizon):
            obs, _, _ = env.step(np.array([0.0, 0.0]))
            tr.append(obs["obst_pos"].copy())
        tr = np.asarray(tr)
        hist, future = tr[:HISTORY + 1], tr[HISTORY + 1:]

        p_tf = tf.rollout(hist, horizon)
        p_ens = ens.rollout(hist, horizon)
        p_cv = constant_velocity_rollout(hist, horizon)
        for name, pred in (("tf", p_tf), ("ens", p_ens), ("cv", p_cv)):
            e = np.linalg.norm(pred - future, axis=-1)
            res[f"ADE_{name}"].append(e.mean())
            res[f"FDE_{name}"].append(e[-1].mean())

    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["horizon_s"] = horizon * DT
    out["episodes"] = n_eval
    print(f"[test7a] ADE  transformer {out['ADE_tf']:.3f} | "
          f"ensemble {out['ADE_ens']:.3f} | const-vel {out['ADE_cv']:.3f}")
    print(f"[test7a] FDE  transformer {out['FDE_tf']:.3f} | "
          f"ensemble {out['FDE_ens']:.3f} | const-vel {out['FDE_cv']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ["Const-vel", "MLP ensemble", "Transformer"]
    ade = [out["ADE_cv"], out["ADE_ens"], out["ADE_tf"]]
    fde = [out["FDE_cv"], out["FDE_ens"], out["FDE_tf"]]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, ade, w, label="ADE", color="#4caf7d")
    ax.bar(x + w / 2, fde, w, label="FDE", color="#7aa6c2")
    ax.set_xticks(x, labels); ax.set_ylabel("error [m] (lower = better)")
    ax.set_title("Test 7a — Prediction error: Transformer vs MLP ensemble")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "stage2_prediction.png", bbox_inches="tight")
    plt.close(fig)

    st = {"test7a": out}
    with open(STATE, "wb") as f:
        pickle.dump(st, f)


# ====================================================================== #
def stage2(i0=0, i1=20):
    """
    Test 7b — planning (chunkable). Runs BOTH backends over [i0, i1).

    NOTE: the transformer's latent dimension is fixed at training time
    (robot 3 + obstacles 2K, here K=6), so it can ONLY be evaluated in an
    environment with the same obstacle count. We therefore run Test 7b in
    the 6-obstacle TRAIN_ENV (matched conditions for a fair planner
    comparison), not the 12-obstacle stress env. Handling variable
    obstacle counts needs a set-structured encoder (e.g. attention over a
    padded obstacle set) — a Stage 2b item logged in docs/ROADMAP.md.
    """
    robot_model, _ = load_models(DATASETS / "world_model.pkl")
    ens = _load_models()
    tf = TransformerWorldModelWrapper(CKPT, n_obstacles=PLAN_ENV["n_obstacles"])

    st = pickle.load(open(STATE, "rb")) if STATE.exists() else {}
    raw = st.setdefault("test7b_raw", {})

    backends = {"Ensemble-MPC": ens, "Transformer-MPC": tf}
    for name, obst_model in backends.items():
        store = raw.setdefault(name, {})
        planner = NeuralMPCPlanner(robot_model, obst_model, seed=0)
        t0 = time.time()
        for i in range(i0, i1):
            env = DynamicWorld(seed=63000 + i, **PLAN_ENV)
            r = run_episode(env, planner)
            store[i] = {k: float(r[k]) for k in ("success", "collision", "energy")}
        s = [d["success"] for d in store.values()]
        c = [d["collision"] for d in store.values()]
        print(f"[test7b] {name:<16} eps {i0}-{i1-1} "
              f"(cum n={len(store)}) success {np.mean(s)*100:.1f}% "
              f"collision {np.mean(c)*100:.1f}%  ({time.time()-t0:.0f}s)")
    with open(STATE, "wb") as f:
        pickle.dump(st, f)


# ====================================================================== #
def stage3():
    st = pickle.load(open(STATE, "rb"))
    rows = {}
    for name, eps in st["test7b_raw"].items():
        rows[name] = {
            "success_rate": float(np.mean([d["success"] for d in eps.values()])),
            "collision_rate": float(np.mean([d["collision"] for d in eps.values()])),
            "energy": float(np.mean([d["energy"] for d in eps.values()])),
            "episodes": len(eps)}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    names = list(rows)
    for ax, key, title in zip(axes, ("success_rate", "collision_rate"),
                              ("Success rate", "Collision rate")):
        vals = [rows[n][key] * 100 for n in names]
        bars = ax.bar(names, vals, color=["#7aa6c2", "#e2a24b"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.2)
    fig.suptitle("Test 7b — Planning: Transformer vs MLP ensemble backend "
                 "(6-obstacle env)", y=1.04)
    fig.tight_layout()
    fig.savefig(ASSETS / "stage2_planning.png", bbox_inches="tight")
    plt.close(fig)

    with open(RESULTS / "benchmark_results.json") as f:
        results = json.load(f)
    results["test7_transformer_vs_ensemble"] = {
        "prediction": st["test7a"], "planning_stress": rows,
        "scope_note": ("Transformer and MLP ensemble both predict in the "
                       "privileged-state latent (robot state + obstacle "
                       "coords); this compares architectures, not raw-sensor "
                       "perception. The transformer's latent is fixed at K=6 "
                       "obstacles, so Test 7b runs in a matched 6-obstacle "
                       "env; variable obstacle counts need a set-structured "
                       "encoder (Stage 2b).")}
    with open(RESULTS / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    STATE.unlink(missing_ok=True)
    print("[done] Stage 2 Test 7 results written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=int, choices=[1, 2, 3])
    ap.add_argument("--i0", type=int, default=0)
    ap.add_argument("--i1", type=int, default=20)
    args = ap.parse_args()
    if args.stage == 1:
        stage1()
    elif args.stage == 2:
        stage2(args.i0, args.i1)
    else:
        stage3()

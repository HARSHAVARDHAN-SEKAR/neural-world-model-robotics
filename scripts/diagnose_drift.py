"""
Stage 2 diagnostic — autoregressive drift (the Test 7 finding).

Shows WHY the transformer world model fails at planning despite excellent
one-step training loss: prediction error stays low for 1 step but grows
sharply as the model feeds its own predictions back in over a horizon,
while the rotation-invariant MLP obstacle model stays flat.

Produces assets/autoregressive_drift.png and prints per-horizon ADE.

Requires torch + datasets/world_transformer.pt + datasets/ensemble_model.pkl.
Run: python scripts/diagnose_drift.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models_pytorch"))

from nwm.env.simulator import DynamicWorld                          # noqa: E402
from nwm.models.world_model import HISTORY, constant_velocity_rollout  # noqa: E402

TRAIN_ENV = dict(n_obstacles=6, obst_speed_range=(0.30, 0.60),
                 obst_turn_range=(-0.50, 0.50))
HORIZON = 30
N_EVAL = 40


def main():
    import torch
    from world_transformer import TransformerWorldModel

    ckpt = torch.load(ROOT / "datasets/world_transformer.pt",
                      map_location="cpu", weights_only=False)
    L, A, C = ckpt["latent_dim"], ckpt["action_dim"], ckpt["context"]
    tf = TransformerWorldModel(latent_dim=L, action_dim=A, context=C)
    tf.load_state_dict(ckpt["model"]); tf.eval()

    with open(ROOT / "datasets/ensemble_model.pkl", "rb") as f:
        ens = pickle.load(f)

    # per-horizon-step error accumulators
    err_tf = np.zeros(HORIZON)
    err_ens = np.zeros(HORIZON)
    err_cv = np.zeros(HORIZON)

    for i in range(N_EVAL):
        env = DynamicWorld(seed=61000 + i, **TRAIN_ENV)
        obs = env.observe()
        robot = [obs["robot"].copy()]
        obst = [obs["obst_pos"].copy()]
        acts = []
        a = np.array([0.0, 0.0])
        for _ in range(HISTORY + HORIZON):
            obs, _, _ = env.step(a)
            robot.append(obs["robot"].copy())
            obst.append(obs["obst_pos"].copy())
            acts.append(a.copy())
        robot = np.asarray(robot, dtype=np.float32)
        obst = np.asarray(obst, dtype=np.float32)
        acts = np.asarray(acts, dtype=np.float32)

        hist_pos = obst[:HISTORY + 1]
        hist_robot = robot[:HISTORY + 1]
        hist_act = acts[:HISTORY + 1]
        future = obst[HISTORY + 1:HISTORY + 1 + HORIZON]

        # transformer rollout (real robot state + actions, correct usage)
        def lat(t):
            return np.concatenate([hist_robot[t], hist_pos[t].reshape(-1)]).astype(np.float32)
        z_hist = np.stack([lat(t) for t in range(HISTORY + 1)])
        if HISTORY + 1 < C:
            zpad = np.repeat(z_hist[:1], C - (HISTORY + 1), axis=0)
            apad = np.repeat(hist_act[:1], C - (HISTORY + 1), axis=0)
            z_ctx = np.concatenate([zpad, z_hist], axis=0)
            a_ctx = np.concatenate([apad, hist_act], axis=0)
        else:
            z_ctx = z_hist[-C:]; a_ctx = hist_act[-C:]
        a_fut = np.repeat(a_ctx[-1:], HORIZON, axis=0)
        with torch.no_grad():
            pred = tf.imagine(torch.tensor(z_ctx[None]),
                              torch.tensor(a_ctx[None]),
                              torch.tensor(a_fut[None]))[0].numpy()
        p_tf = pred[:, 3:].reshape(HORIZON, 6, 2)

        p_ens = ens.rollout(hist_pos, HORIZON)
        p_cv = constant_velocity_rollout(hist_pos, HORIZON)

        err_tf += np.linalg.norm(p_tf - future, axis=-1).mean(axis=1)
        err_ens += np.linalg.norm(p_ens - future, axis=-1).mean(axis=1)
        err_cv += np.linalg.norm(p_cv - future, axis=-1).mean(axis=1)

    err_tf /= N_EVAL; err_ens /= N_EVAL; err_cv /= N_EVAL
    steps = np.arange(1, HORIZON + 1)
    t = steps * 0.1

    # persist curve for run_stage2_benchmark.py stage3 to merge into JSON
    curve = {"horizon_s": [float(x) for x in t],
             "ADE_transformer": [float(x) for x in err_tf],
             "ADE_mlp_ensemble": [float(x) for x in err_ens],
             "ADE_const_velocity": [float(x) for x in err_cv]}
    with open(ROOT / "results/_drift.pkl", "wb") as f:
        pickle.dump(curve, f)

    print("horizon(s)  transformer  MLP-ensemble  const-vel")
    for k in (0, 4, 9, 19, 29):
        print(f"  {t[k]:4.1f}       {err_tf[k]:8.3f}    {err_ens[k]:8.3f}   {err_cv[k]:8.3f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(t, err_tf, "-o", ms=3, color="#e2564b", label="Transformer (absolute latent)")
    ax.plot(t, err_ens, "-o", ms=3, color="#4caf7d", label="MLP ensemble (rotation-inv. velocity)")
    ax.plot(t, err_cv, "--", color="#9aa0a6", label="Constant velocity")
    ax.axvline(0.1, color="k", lw=0.7, alpha=0.4)
    ax.text(0.12, ax.get_ylim()[1] * 0.9, "1-step\n(teacher-forced\nregime)",
            fontsize=7, color="k", alpha=0.6)
    ax.set(xlabel="prediction horizon [s]", ylabel="obstacle ADE [m]",
           title="Autoregressive drift: one-step accuracy ≠ rollout accuracy")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "assets/autoregressive_drift.png", dpi=140,
                bbox_inches="tight")
    print("\nsaved assets/autoregressive_drift.png")


if __name__ == "__main__":
    main()

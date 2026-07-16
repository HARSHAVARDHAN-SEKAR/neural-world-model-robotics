"""
Stage 2 — train the Transformer world model on GPU (see docs/ROADMAP.md).

This trains `models_pytorch/world_transformer.py` on the SAME logged
experience the CPU pipeline uses (`datasets/robot_experience.npz` +
the per-episode obstacle tracks), so its rollout can be dropped into
`OccupancyNeuralMPC`/`NeuralMPCPlanner` in place of the MLP models and
benchmarked in the same tables.

Not run in CI (requires torch + meaningfully more data/compute than the
CPU smoke tests budget for). Run locally:

    pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cpu wheel
    python scripts/train_world_transformer.py --epochs 30 --device cuda

Data note: `datasets/robot_experience.npz` only stores (s, a, s') robot
transitions, not the full per-episode obstacle tracks needed for
sequence training. Regenerate an episode-indexed dataset first:

    python scripts/collect_sequence_data.py   # writes datasets/sequences.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models_pytorch"))


def build_latent_sequences(seq_npz: Path, context: int = 100, horizon: int = 50):
    """
    Load datasets/sequences.npz (written by collect_sequence_data.py) and
    build training windows of [z_t, a_t] -> z_{t+1}. Here z is simply the
    concatenation of normalized robot state + flattened obstacle
    positions (a placeholder "latent" — swap in models_pytorch/vae_encoder.py
    once training from raw lidar/camera rasters instead of ground-truth
    state).
    """
    import torch

    data = np.load(seq_npz, allow_pickle=True)
    robot_seqs = data["robot"]        # (n_episodes,) object array of (T, 3)
    action_seqs = data["actions"]     # (n_episodes,) object array of (T, 2)
    obst_seqs = data["obst"]          # (n_episodes,) object array of (T, K, 2)

    windows_z, windows_a, windows_target = [], [], []
    for robot, actions, obst in zip(robot_seqs, action_seqs, obst_seqs):
        T = len(actions)
        obst_flat = obst.reshape(T + 1, -1)                    # (T+1, 2K)
        z = np.concatenate([robot, obst_flat[:len(robot)]], axis=-1)  # (T, 3+2K)
        if T < context + horizon:
            continue
        for start in range(0, T - context - horizon, max(1, horizon // 2)):
            ctx_z = z[start:start + context]
            ctx_a = actions[start:start + context]
            fut_z = z[start + 1:start + context + 1]           # next-step targets
            windows_z.append(ctx_z)
            windows_a.append(ctx_a)
            windows_target.append(fut_z)

    Z = torch.tensor(np.stack(windows_z), dtype=torch.float32)
    A = torch.tensor(np.stack(windows_a), dtype=torch.float32)
    Y = torch.tensor(np.stack(windows_target), dtype=torch.float32)
    return Z, A, Y


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from world_transformer import TransformerWorldModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--context", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "world_transformer.pt"))
    args = ap.parse_args()

    seq_path = ROOT / "datasets" / "sequences.npz"
    if not seq_path.exists():
        raise SystemExit(
            f"{seq_path} not found. Run `python scripts/collect_sequence_data.py` "
            "first to build episode-indexed sequences for Transformer training.")

    print(f"[stage2] loading sequences from {seq_path}")
    Z, A, Y = build_latent_sequences(seq_path, args.context, args.horizon)
    latent_dim, action_dim = Z.shape[-1], A.shape[-1]
    print(f"[stage2] {len(Z)} training windows, latent_dim={latent_dim}, "
         f"action_dim={action_dim}, device={args.device}")

    ds = TensorDataset(Z, A, Y)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TransformerWorldModel(latent_dim=latent_dim, action_dim=action_dim,
                                  context=args.context).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for z, a, y in train_loader:
            z, a, y = z.to(args.device), a.to(args.device), y.to(args.device)
            pred = model(z, a)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(z)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z, a, y in val_loader:
                z, a, y = z.to(args.device), a.to(args.device), y.to(args.device)
                val_loss += loss_fn(model(z, a), y).item() * len(z)
        val_loss /= len(val_ds)

        print(f"[stage2] epoch {epoch+1:>3}/{args.epochs}  "
             f"train MSE {train_loss:.4e}  val MSE {val_loss:.4e}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                       "latent_dim": latent_dim, "action_dim": action_dim,
                       "context": args.context}, args.out)

    print(f"[stage2] done. best val MSE {best_val:.4e}. saved -> {args.out}")
    print("[stage2] next: wire this checkpoint into an "
         "OccupancyNeuralMPC-style planner (rollout via model.imagine(...)) "
         "and re-run scripts/run_stage1_occupancy.py's benchmarks for a "
         "head-to-head row against the MLP ensemble.")


if __name__ == "__main__":
    main()

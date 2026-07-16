"""Stage-1 occupancy planner benchmark, run in chunks.
Usage: python scripts/bench_stage1_chunk.py <planner_idx 0|1|2> <i0> <i1>
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage1_occupancy import HORIZON, STRESS_ENV, STATE  # noqa: E402
from nwm.env.simulator import DynamicWorld                   # noqa: E402
from nwm.models.world_model import load_models                # noqa: E402
from nwm.planners.planners import (NeuralMPCPlanner,          # noqa: E402
                                   OccupancyNeuralMPC,
                                   RiskAwareNeuralMPC, run_episode)

pi, i0, i1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rm, _ = load_models(ROOT / "datasets/world_model.pkl")
ens = pickle.load(open(ROOT / "datasets/ensemble_model.pkl", "rb"))
planner = [
    NeuralMPCPlanner(rm, ens, horizon=HORIZON, seed=0),
    RiskAwareNeuralMPC(rm, ens, horizon=HORIZON, risk_q=0.4, seed=0),
    OccupancyNeuralMPC(rm, ens, horizon=HORIZON, seed=0),
][pi]

st = pickle.load(open(STATE, "rb"))
raw = st.setdefault("test6b_raw", {}).setdefault(planner.name, {})
t0 = time.time()
for i in range(i0, i1):
    env = DynamicWorld(seed=51000 + i, **STRESS_ENV)
    r = run_episode(env, planner)
    raw[i] = {k: float(r[k]) for k in ("success", "collision", "energy")}
pickle.dump(st, open(STATE, "wb"))
s = [d["success"] for d in raw.values()]
c = [d["collision"] for d in raw.values()]
print(f"{planner.name}: eps {i0}-{i1-1} done ({time.time()-t0:.0f}s) | "
      f"cumulative n={len(raw)} success {np.mean(s)*100:.1f}% "
      f"collision {np.mean(c)*100:.1f}%")

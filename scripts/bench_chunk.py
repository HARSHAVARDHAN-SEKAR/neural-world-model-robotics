"""Stress-benchmark chunk runner: python scripts/bench_chunk.py <planner 0|1> <i0> <i1>"""
import sys, pickle, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from run_upgrade import STRESS_ENV, STATE
from nwm.env.simulator import DynamicWorld
from nwm.models.world_model import load_models
from nwm.planners.planners import NeuralMPCPlanner, RiskAwareNeuralMPC, run_episode

pi, i0, i1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rm, _ = load_models(ROOT/'datasets/world_model.pkl')
ens = pickle.load(open(ROOT/'datasets/ensemble_model.pkl','rb'))
planner = [NeuralMPCPlanner(rm, ens, seed=0),
           RiskAwareNeuralMPC(rm, ens, risk_q=0.4, seed=0)][pi]

st = pickle.load(open(STATE,'rb'))
raw = st.setdefault('stress_raw', {}).setdefault(planner.name, {})
t0 = time.time()
for i in range(i0, i1):
    env = DynamicWorld(seed=77000+i, **STRESS_ENV)
    r = run_episode(env, planner)
    raw[i] = {k: float(r[k]) for k in ('success','collision','steps','energy')}
pickle.dump(st, open(STATE,'wb'))
done = raw
print(f"{planner.name}: eps {i0}-{i1-1} done ({time.time()-t0:.0f}s) | "
      f"cumulative n={len(done)} success {np.mean([d['success'] for d in done.values()])*100:.1f}% "
      f"collision {np.mean([d['collision'] for d in done.values()])*100:.1f}%")

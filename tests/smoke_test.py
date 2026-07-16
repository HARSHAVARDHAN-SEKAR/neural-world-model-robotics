"""Fast CI smoke test (<60 s): env physics, model training on tiny data,
every planner produces valid actions, ensemble uncertainty API works."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nwm.env.simulator import DynamicWorld, V_MAX, W_MAX
from nwm.models.world_model import (ObstacleMotionModel, RobotDynamicsModel,
                                    constant_velocity_rollout, HISTORY)
from nwm.models.ensemble import EnsembleObstacleModel
from nwm.planners.planners import (DWAPlanner, MPPIPlanner, NeuralMPCPlanner,
                                   ReactivePlanner, RiskAwareNeuralMPC,
                                   run_episode)


def collect(n_ep=3, steps=120):
    S, A, SN, tracks = [], [], [], []
    rng = np.random.default_rng(0)
    for ep in range(n_ep):
        env = DynamicWorld(seed=ep)
        obs = env.observe()
        tr = [obs["obst_pos"].copy()]
        for _ in range(steps):
            a = np.array([rng.uniform(0, V_MAX), rng.uniform(-W_MAX, W_MAX)])
            s = obs["robot"].copy()
            obs, _, _ = env.step(a)
            S.append(s); A.append(a); SN.append(obs["robot"].copy())
            tr.append(obs["obst_pos"].copy())
        tracks.append(np.asarray(tr))
    return map(np.asarray, (S, A, SN)), tracks


def test_all():
    (S, A, SN), tracks = collect()
    rm = RobotDynamicsModel(seed=0)
    assert rm.fit(S, A, SN) < 1e-2, "robot dynamics failed to fit"

    X, Y = ObstacleMotionModel.make_dataset(np.concatenate(tracks, axis=0))
    om = ObstacleMotionModel(seed=0); om.fit(X, Y)
    ens = EnsembleObstacleModel(n_members=2, seed=0); ens.fit(X, Y)

    hist = tracks[0][:HISTORY + 1]
    for pred in (om.rollout(hist, 10), constant_velocity_rollout(hist, 10),
                 ens.rollout(hist, 10)):
        assert pred.shape == (10, hist.shape[1], 2) and np.isfinite(pred).all()
    futures = ens.rollout_all(hist, 10)
    assert EnsembleObstacleModel.disagreement(futures).shape == (10, hist.shape[1])

    for pl in (ReactivePlanner(), DWAPlanner(), MPPIPlanner(seed=0),
               NeuralMPCPlanner(rm, om, seed=0),
               RiskAwareNeuralMPC(rm, ens, seed=0)):
        env = DynamicWorld(seed=42)
        r = run_episode(env, pl, max_steps=40)
        assert np.isfinite(r["path_length"]), f"{pl.name} produced NaN"
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    test_all()

"""Run the pipeline in resumable stages: python scripts/run_stage.py <1|2|3>."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline as rp  # noqa: E402
from nwm.models.world_model import load_models  # noqa: E402

STATE = rp.RESULTS / "_stage_state.pkl"


def main(stage: int):
    if stage == 1:
        S, A, SN, tracks = rp.collect_experience()
        robot_model, obst_model = rp.train_world_model(S, A, SN, tracks)
        pred_in, sample = rp.test1_prediction(obst_model)
        rp.plot_prediction(sample)
        pred_ood, _ = rp.test1_prediction(obst_model, env_cfg=rp.OOD_ENV,
                                          tag="unseen-harder")
        with open(STATE, "wb") as f:
            pickle.dump({"pred_in": pred_in, "pred_ood": pred_ood}, f)

    elif stage == 2:
        robot_model, obst_model = load_models(rp.DATASETS / "world_model.pkl")
        plan_in, demo = rp.test2_planning(robot_model, obst_model)
        rp.plot_planning(plan_in, "training environment",
                         "benchmark_planning.png")
        rp.plot_episode(demo, "episode_trajectories.png")
        with open(STATE, "rb") as f:
            st = pickle.load(f)
        st["plan_in"] = plan_in
        with open(STATE, "wb") as f:
            pickle.dump(st, f)

    elif stage == 3:
        robot_model, obst_model = load_models(rp.DATASETS / "world_model.pkl")
        plan_ood, _ = rp.test2_planning(robot_model, obst_model, n_eps=30,
                                        env_cfg=rp.OOD_ENV,
                                        tag="unseen-harder", seed0=42000)
        rp.plot_planning(plan_ood, "unseen harder environment",
                         "benchmark_planning_ood.png")
        with open(STATE, "rb") as f:
            st = pickle.load(f)
        rp.plot_generalization(st["pred_in"], st["pred_ood"],
                               st["plan_in"], plan_ood)
        results = {"test1_prediction": {"in_distribution": st["pred_in"],
                                        "out_of_distribution": st["pred_ood"]},
                   "test2_planning": {"in_distribution": st["plan_in"],
                                      "out_of_distribution": plan_ood}}
        with open(rp.RESULTS / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        STATE.unlink(missing_ok=True)
        print("[done] all results written")


if __name__ == "__main__":
    main(int(sys.argv[1]))

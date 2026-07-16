# Research Roadmap

Target architecture (long-term):

```
Camera · LiDAR · IMU · GPS · Encoders
                │
   Localization + Sensor Fusion (EKF / Factor Graph)
                │
        World State Builder
  (robot state · dynamic obstacles · static map ·
   semantic objects · costmap · goal)
                │
        LATENT WORLD MODEL
     (Transformer / RSSM / Dreamer)
                │
     Imagine N futures (multi-hypothesis)
                │
        Risk Evaluation
  (collision probability · time-to-collision ·
   visibility · dynamic occupancy)
                │
     Risk-aware Neural MPC / MPPI
                │
   Local controller → /cmd_vel → Robot
                │
   New experience → continual model update ↺
```

## Stage 0 — DONE (this repo, v1.0)
- [x] Learned robot dynamics + rotation-invariant obstacle motion model
- [x] Imagination planner (MPPI over learned model), DWA/MPPI/reactive baselines
- [x] ADE/FDE, planning, and generalization benchmarks (Tests 1–3)
- [x] 5-member bootstrapped ensemble; calibrated uncertainty (r = 0.49)
- [x] Multi-future planning: −40 % collisions under stress (Tests 4–5)
- [x] Docker, CI, smoke tests, technical report, demo animation

## Stage 1 — Occupancy-based prediction — DONE
- [x] `src/nwm/env/occupancy.py`: world<->grid transforms, ensemble
      rasterization (member disagreement = natural uncertainty widening,
      no separate scalar needed), occupancy IoU / soft-IoU metrics
- [x] `OccupancyNeuralMPC` planner: reads predicted occupancy directly
      under imagined trajectories, in addition to point-based cost
- [x] Test 6a (prediction quality): ensemble occupancy IoU 0.656 vs
      constant-velocity 0.569 — consistent with the ADE/FDE gap
- [x] Test 6b (planning, stress env, horizon 1.5s): Occupancy-MPC 60%
      success / 25% collision vs Neural-MPC 45%/35% and Risk-MPC 40%/20%
      — occupancy-based planning is the strongest planner at this
      shorter horizon; full results in `results/benchmark_results.json`
- [x] Note (calibration pitfall, documented for future stages): the
      ensemble grid must be splatted with the SAME sigma as the
      ground-truth grid, or blur alone destroys IoU regardless of
      accuracy — see the docstring in `rasterize_ensemble`.
- [x] Kept CPU-runnable; reproduce with
      `python scripts/run_stage1_occupancy.py 1|2|3` (2 is chunked via
      `scripts/bench_stage1_chunk.py` to stay under long-running budgets)

## Stage 2 — Latent sequence world model (scaffolded, GPU required)
- [x] `scripts/collect_sequence_data.py`: episode-indexed dataset for
      sequence training (verified — produces correctly shaped
      per-episode robot/action/obstacle arrays)
- [x] `scripts/train_world_transformer.py`: full training loop for
      `models_pytorch/world_transformer.py` (train/val split, AdamW,
      checkpointing) — syntax-checked; not run in this environment
      (no GPU / torch here) — **run locally**:
      `python scripts/collect_sequence_data.py && python scripts/train_world_transformer.py --epochs 30`
- [x] **Trained** on your hardware: 60 epochs, val MSE 26.8 -> 0.104,
      clean curve, no overfitting (checkpoint datasets/world_transformer.pt)
- [x] `src/nwm/models/transformer_wrapper.py`: loads the checkpoint and
      exposes the same `.rollout()` interface as the MLP obstacle model,
      dropping into NeuralMPCPlanner unchanged (torch imported lazily so
      the rest of the repo + CI still run without torch)
- [x] `scripts/run_stage2_benchmark.py`: **Test 7** — Transformer vs MLP
      ensemble, prediction (7a) + planning (7b), same metrics/episodes
- [x] **Test 7 run — honest negative result (the Stage 2 finding).**
      The transformer reaches excellent ONE-STEP prediction (val MSE
      ~0.07) but suffers severe **autoregressive drift**: 3 s rollout ADE
      ~2.0 m vs the rotation-invariant MLP ensemble's ~0.10 m. The MLP
      predicts velocity in a heading-aligned canonical frame (a
      well-conditioned, rollout-stable target); the transformer predicts
      the absolute next latent, so per-step errors compound. Diagnostic:
      `scripts/diagnose_drift.py` -> assets/autoregressive_drift.png.
      This reproduces the central stability challenge of learned world
      models — a stronger portfolio result than a tuned win, because it
      shows *why* the classical parameterization wins.

## Stage 2c — Rollout-stable world model (the fix, next real experiment)
- [ ] Multi-step rollout loss / scheduled sampling: during training,
      periodically feed the model its OWN prediction and penalize
      accumulated K-step error, so it learns to recover from its own
      mistakes rather than only one-step teacher-forced accuracy.
- [ ] Or change the target: predict per-obstacle velocity in a
      rotation-invariant frame (as the MLP does) instead of absolute
      latent — better-conditioned for rollout.
- [ ] Or latent regularization (DreamerV3-style KL) to keep predicted
      latents on-manifold across the rollout.
- [ ] Re-run Test 7 after any of these; expect the drift curve to flatten.

## Stage 2b — Variable obstacle counts (found during Stage 2)
- [ ] The transformer's latent is fixed at K=6 (robot 3 + 2K), so it
      cannot ingest the 12-obstacle stress env. Test 7b runs in a matched
      6-obstacle env. Handling variable K needs a **set-structured
      encoder** — attention over a padded/masked obstacle set, or a
      per-obstacle token stream — so the model is permutation-invariant
      and count-agnostic. This is the natural bridge to the
      interaction-aware (GNN/attention) models in Stage 3.
- [ ] Replace ground-truth-state "latent" placeholder with
      `models_pytorch/vae_encoder.py` over rasterized lidar (raw sensors,
      not privileged state)
- [ ] Train `dreamer_rssm.py`; multi-hypothesis via stochastic latents
      (replaces bootstrap ensemble)

## Stage 3 — ROS 2 + Gazebo/Isaac validation
- [ ] Wire `ros2_ws/world_model_node` to a diff-drive robot in Gazebo
- [ ] Perception front-end: obstacle tracker (LiDAR clustering; later YOLO + tracking)
- [ ] RViz overlays: imagined futures, uncertainty fans, chosen trajectory
- [ ] Record demo videos; report sim benchmark table

## Stage 4 — Real robot (Jetson + LiDAR + camera)
- [ ] Deploy 10–20 Hz pipeline on hardware; EKF localization from the UGV repo
- [ ] Online/continual world-model updates from live experience
- [ ] Risk-aware costs: collision probability, time-to-collision, visibility, comfort
- [ ] Published benchmark + updated technical report

## Non-goals for this repo
High-level LLM task planning (lives in CRIP) and the classical navigation stack
(lives in Learning-Based-Adaptive-Navigation-Controller-for-UGV). This repo owns
one question: **can the robot learn, imagine, and act on predicted futures?**

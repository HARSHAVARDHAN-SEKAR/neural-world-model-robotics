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
- [ ] Wire the trained checkpoint's `.imagine(...)` into an
      Occupancy/Neural-MPC-style planner and benchmark head-to-head
      against the MLP ensemble in the same tables (Test 7, not yet run)
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

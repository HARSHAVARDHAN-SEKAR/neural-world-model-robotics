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

## Stage 1 — Occupancy-based prediction (next)
- [ ] Replace per-obstacle (x, y) prediction with **future occupancy grids**
      (predict the map, not the points); MPC cost evaluated on predicted grids
- [ ] Metrics: occupancy IoU/soft-IoU over horizon, alongside ADE/FDE
- [ ] Keep CPU-runnable variant for CI

## Stage 2 — Latent sequence world model
- [ ] Train `models_pytorch/world_transformer.py` and `dreamer_rssm.py` (GPU)
      on logged experience; benchmark vs MLP rows in the same tables
- [ ] VAE/encoder over lidar rasters → latent z; predict future z, decode occupancy
- [ ] Multi-hypothesis sampling from stochastic latents (replaces bootstrap ensemble)

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

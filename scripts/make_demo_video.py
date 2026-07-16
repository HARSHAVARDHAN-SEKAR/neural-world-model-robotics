"""Render an animated demo (GIF + MP4 if ffmpeg) of Neural-MPC navigating,
showing the robot's *imagined* obstacle futures live — the money shot."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nwm.env.simulator import ARENA, DT, OBST_RADIUS, ROBOT_RADIUS, DynamicWorld
from nwm.models.world_model import load_models
from nwm.planners.planners import NeuralMPCPlanner

rm, om = load_models(ROOT / "datasets/world_model.pkl")
env = DynamicWorld(seed=9003, n_obstacles=6,
                   obst_speed_range=(0.30, 0.60), obst_turn_range=(-0.5, 0.5))
pl = NeuralMPCPlanner(rm, om, seed=0); pl.reset()
obs = env.observe()

frames = []
for _ in range(300):
    a = pl.act(obs)
    pred = pl._last_pred if hasattr(pl, "_last_pred") else None
    frames.append((obs["robot"].copy(), obs["obst_pos"].copy(),
                   obs["goal"].copy(),
                   om.rollout(np.stack(pl.pos_hist), 25)
                   if len(pl.pos_hist) > 6 else None))
    obs, hit, reached = env.step(a)
    if hit or reached:
        frames.append((obs["robot"].copy(), obs["obst_pos"].copy(),
                       obs["goal"].copy(), None))
        print("episode end:", "SUCCESS" if reached else "COLLISION",
              f"({len(frames)} frames)")
        break

fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=90)

def draw(i):
    ax.clear()
    robot, opos, goal, pred = frames[i]
    ax.set(xlim=(0, ARENA), ylim=(0, ARENA), aspect="equal",
           xticks=[], yticks=[])
    ax.set_title("Neural-MPC: dashed = robot's imagined obstacle futures",
                 fontsize=9)
    if pred is not None:
        for k in range(pred.shape[1]):
            ax.plot(pred[:, k, 0], pred[:, k, 1], "--", color="tomato",
                    lw=1.2, alpha=0.8)
    for p in opos:
        ax.add_patch(plt.Circle(p, OBST_RADIUS, color="tomato", alpha=0.9))
    ax.add_patch(plt.Circle(robot[:2], ROBOT_RADIUS, color="#1565c0"))
    ax.plot([robot[0], robot[0] + 0.6 * np.cos(robot[2])],
            [robot[1], robot[1] + 0.6 * np.sin(robot[2])], "-",
            color="white", lw=2)
    ax.plot(*goal, "*", color="green", ms=16)
    tr = np.array([f[0][:2] for f in frames[:i + 1]])
    ax.plot(tr[:, 0], tr[:, 1], "-", color="#1565c0", lw=1, alpha=0.5)

anim = FuncAnimation(fig, draw, frames=len(frames), interval=50)
out = ROOT / "assets" / "demo_neural_mpc.gif"
anim.save(out, writer=PillowWriter(fps=15))
print("saved", out)

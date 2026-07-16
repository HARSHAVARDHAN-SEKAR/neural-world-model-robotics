"""Render the system architecture diagram to assets/architecture.png."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "architecture.png"

fig, ax = plt.subplots(figsize=(9.2, 11.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")


def box(x, y, w, h, text, fc, ec="#37474f", fs=10, tc="#212121"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold")


def arrow(x1, y1, x2, y2, label=None, color="#455a64"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color=color))
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label,
                fontsize=8, color="#455a64", ha="left")


# ---- sensors ---------------------------------------------------------- #
box(0.6, 12.6, 2.4, 0.9, "Camera", "#bbdefb")
box(3.8, 12.6, 2.4, 0.9, "LiDAR", "#bbdefb")
box(7.0, 12.6, 2.4, 0.9, "IMU / Odom", "#bbdefb")

box(2.8, 11.0, 4.4, 0.9, "Sensor Encoder\n(VAE / PCA → latent z)", "#c8e6c9", fs=9)
for x in (1.8, 5.0, 8.2):
    arrow(x, 12.6, 5.0, 11.95)

# ---- world model ------------------------------------------------------ #
box(1.2, 8.9, 7.6, 1.5,
    "NEURAL WORLD MODEL\nRobot Dynamics Model  +  Obstacle Motion Predictor\n"
    "(MLP runnable core · Transformer / Dreamer-style in models_pytorch/)",
    "#fff3c4", fs=9.5)
arrow(5.0, 11.0, 5.0, 10.45, " z_t, a_t")

box(2.3, 7.2, 5.4, 1.0,
    "Imagined Futures\nŝ(t+1) … ŝ(t+H)   ·   3 s prediction horizon",
    "#ffe0b2", fs=9)
arrow(5.0, 8.9, 5.0, 8.25)

# ---- imagination planner ---------------------------------------------- #
box(1.2, 5.2, 7.6, 1.5,
    "IMAGINATION PLANNER (Neural-MPC)\nMPPI: sample 220 action sequences → "
    "roll out inside the\nlearned model → cost(collision, goal, smooth, energy)",
    "#d1c4e9", fs=9.5)
arrow(5.0, 7.2, 5.0, 6.75)

box(3.3, 3.7, 3.4, 0.9, "Best action a*_t\n(v, ω)", "#c5cae9", fs=9.5)
arrow(5.0, 5.2, 5.0, 4.65)

# ---- robot ------------------------------------------------------------ #
box(2.8, 2.0, 4.4, 1.0, "ROS 2 Robot / Simulator\n(Gazebo · Isaac · this repo's 2D sim)",
    "#b2dfdb", fs=9)
arrow(5.0, 3.7, 5.0, 3.05, " /cmd_vel")

# ---- experience loop --------------------------------------------------- #
box(0.3, 0.4, 9.4, 0.9,
    "Experience replay:  (s, a, s′) transitions + obstacle tracks → retrain world model",
    "#f8bbd0", fs=9)
arrow(5.0, 2.0, 5.0, 1.35)
ax.add_patch(FancyArrowPatch((9.4, 0.85), (9.4, 9.6),
                             connectionstyle="arc3,rad=-0.25",
                             arrowstyle="-|>", mutation_scale=15,
                             lw=1.6, color="#ad1457", ls="--"))
ax.text(9.55, 5.2, "learning loop", rotation=90, fontsize=8,
        color="#ad1457", va="center")

ax.set_title("Neural World Model & Predictive Robot Intelligence\n"
             "sense → learn world → imagine futures → choose action → act",
             fontsize=12, weight="bold", pad=14)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)

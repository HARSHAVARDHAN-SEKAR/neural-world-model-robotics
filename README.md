# Artemis Perception Testing Workspace

## Architecture

```
HOST LAPTOP (your machine — any ROS2 distro)
  ├── ros2 bag play  → publishes /cloud + /camera/...
  └── rviz2          → visualizes output topics
              ↕  localhost network (ROS_DOMAIN_ID=0)
DOCKER (artemisros-dev — Foxy)
  ├── lidar_perception_node   (RANSAC + DBSCAN + costmap)
  ├── fusion_node             (LiDAR + camera fusion)
  └── dwa_planner_node        (path planning)
```

## File Map

| File | Where to run | Purpose |
|------|-------------|---------|
| `setup_test_ws.sh` | HOST once | Creates `~/artemis_test_ws/`, links bags |
| `play_bag.sh` | HOST | Plays rosbag into ROS network |
| `run_perception.sh` | DOCKER | Starts all perception nodes |
| `check_topics.sh` | DOCKER | Checks topic Hz rates |
| `perception_visualizer.py` | DOCKER | Live matplotlib monitor |
| `test_terrain_classifier.py` | DOCKER | Offline unit test (no live ROS needed) |

---

## Quickstart — 3 Terminals

### Terminal 1 — HOST: Setup (once)
```bash
cd ~/artemis_test_ws    # or wherever you put these scripts
bash setup_test_ws.sh
```

### Terminal 2 — HOST: Play bag
```bash
bash play_bag.sh                              # newest bag, 1x speed
bash play_bag.sh rosbag2_2026_04_23-12_56_50  # specific bag
bash play_bag.sh rosbag2_... 0.5              # half speed for debugging
```

### Terminal 3 — DOCKER: Run nodes
```bash
# Enter Docker
cd /mnt/sd/pynav_pcp && make exec-dev

# Inside Docker:
bash /mnt/sd/pynav_pcp/scripts/run_perception.sh

# Or just one node for focused debugging:
bash run_perception.sh --node lidar
bash run_perception.sh --node fusion
bash run_perception.sh --node terrain   # runs offline test
bash run_perception.sh --lidar-only     # no fusion, no DWA
```

### Terminal 4 — DOCKER: Monitor
```bash
# Check all topic rates
bash /mnt/sd/pynav_pcp/scripts/check_topics.sh

# Live matplotlib visualizer (needs display or X11 forwarding)
python3 /mnt/sd/pynav_pcp/scripts/perception_visualizer.py

# Or just tail logs
tail -f /mnt/sd/pynav_pcp/test_logs/lidar_*.log
```

---

## Offline Terrain Test (no bag player needed)

This runs directly inside Docker — reads the bag file directly:

```bash
# List available bags
python3 test_terrain_classifier.py --list-bags

# Run test on newest bag (10 frames)
python3 test_terrain_classifier.py

# Specific bag, more frames
python3 test_terrain_classifier.py \
    --bag ~/Projects/rosbag_new_1/rosbag2_2026_04_23-12_56_50 \
    --frames 30

# Results saved to:
ls /mnt/sd/pynav_pcp/test_results/terrain_*/
```

Output: one PNG per frame showing camera + LiDAR side by side with terrain label.

---

## Known Issues & Fixes

### fusion_node FPS = 0
**Cause**: Stale image timeout too short  
**Fix**: `STALE_IMAGE_MS=6000` already applied in final version

### RANSAC terrain always 86° (steep)
**Cause**: Running indoors near walls — RANSAC picks a wall plane  
**Fix**: In lidar_processor.py set `max_range=8.0, dbscan_min_pts=3`  
Or: point the robot toward open space during bag recording

### lidar_perception_node.py line 59 NameError
**Fix**: Delete the bare word `dwa` on line 59 (just `dwa` on its own)

### No display for visualizer inside Docker
```bash
# Option A: X11 forwarding
ssh -X user@machine
DISPLAY=:0 python3 perception_visualizer.py

# Option B: Use offline test instead (saves PNG files)
python3 test_terrain_classifier.py --frames 20

# Option C: Open rviz2 on host (if host has ROS2)
rviz2 -d ~/artemis_test_ws/config/artemis_perception.rviz
```

---

## Deploy to Docker

Copy all scripts to the Docker working directory:
```bash
# From host:
docker cp artemis_perception_test/. artemisros-dev:/mnt/sd/pynav_pcp/scripts/

# Or if using volume mount (preferred):
cp *.py *.sh /mnt/sd/pynav_pcp/scripts/
```

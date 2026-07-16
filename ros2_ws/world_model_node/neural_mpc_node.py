#!/usr/bin/env python3
"""
ROS 2 node: Neural World Model controller.

Subscribes
    /odom                  nav_msgs/Odometry        robot state
    /obstacles             geometry_msgs/PoseArray  tracked obstacle centers
    /goal_pose             geometry_msgs/PoseStamped

Publishes
    /cmd_vel               geometry_msgs/Twist       chosen action
    /predicted_obstacles   geometry_msgs/PoseArray   imagined obstacle
                                                     positions H steps ahead
                                                     (for RViz overlay)

The node loads the trained world model (datasets/world_model.pkl produced
by scripts/run_pipeline.py) and runs the same NeuralMPCPlanner used in the
benchmarks at 10 Hz. Obstacle tracking (e.g. from LiDAR clustering or a
camera detector) is assumed to be provided by an upstream perception node.

Build: drop this package into a ROS 2 (Humble+) workspace, add
`nwm` to PYTHONPATH (or pip-install the repo), then:
    ros2 run world_model_node neural_mpc_node
"""

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
except ImportError:  # allows importing this file outside a ROS install
    rclpy = None
    Node = object

from nwm.models.world_model import load_models
from nwm.planners.planners import NeuralMPCPlanner


def yaw_from_quat(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class NeuralMPCNode(Node):
    def __init__(self):
        super().__init__("neural_mpc_node")
        self.declare_parameter("model_path", "datasets/world_model.pkl")
        self.declare_parameter("rate_hz", 10.0)
        path = self.get_parameter("model_path").value
        robot_model, obst_model = load_models(path)
        self.planner = NeuralMPCPlanner(robot_model, obst_model)
        self.planner.reset()

        self.robot = None
        self.goal = None
        self.obst_pos = np.zeros((0, 2))
        self.prev_obst = None

        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(PoseArray, "obstacles", self.on_obst, 10)
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.pred_pub = self.create_publisher(PoseArray,
                                              "predicted_obstacles", 10)
        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / rate, self.control_step)
        self.get_logger().info(f"Neural-MPC ready (model: {path})")

    # ------------------------------------------------------------------ #
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot = np.array([p.x, p.y, yaw_from_quat(msg.pose.pose.orientation)])

    def on_goal(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.planner.reset()

    def on_obst(self, msg: PoseArray):
        self.prev_obst = self.obst_pos if len(self.obst_pos) else None
        self.obst_pos = np.array([[p.position.x, p.position.y]
                                  for p in msg.poses])

    # ------------------------------------------------------------------ #
    def control_step(self):
        if self.robot is None or self.goal is None or not len(self.obst_pos):
            return
        vel = (np.zeros_like(self.obst_pos) if self.prev_obst is None
               or self.prev_obst.shape != self.obst_pos.shape
               else (self.obst_pos - self.prev_obst) * 10.0)
        obs = {"robot": self.robot, "goal": self.goal,
               "obst_pos": self.obst_pos, "obst_vel": vel}
        a = self.planner.act(obs)

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(a[0]), float(a[1])
        self.cmd_pub.publish(cmd)

        # publish the planner's imagined obstacle end-positions for RViz
        pred = self.planner._predict_obstacles(obs)[-1]
        pa = PoseArray()
        pa.header.frame_id = "map"
        pa.header.stamp = self.get_clock().now().to_msg()
        for x, y in pred:
            pose = Pose()
            pose.position.x, pose.position.y = float(x), float(y)
            pa.poses.append(pose)
        self.pred_pub.publish(pa)


def main():
    rclpy.init()
    node = NeuralMPCNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
TeleopBridgeNode — subscribes to the MyBotShop IK WebSocket (port 9001) and
forwards joint solution broadcasts as ROS2 joint commands.

The webserver UI sends cartesian/joint targets to the IK server; the IK server
broadcasts its solutions to ALL connected WebSocket clients. This node
connects as one such client and republishes the solutions to ROS2.

WebSocket URL: ws://<host>:<ik_ws_port>   (default port 9001)

Message formats handled:
  Joint solution broadcast:
    {"arm": "...", "joint_names": [...], "joint_positions": [...], "success": true}
  Finger/gripper broadcast:
    {"arm": "...", "fingers": {"<joint_suffix>": <float_rad>, ...}}

Publishes:
  /joint_commands      trajectory_msgs/JointTrajectory
  /gripper_command     std_msgs/Float32  (0.0 closed – 1.0 open)
"""

from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32

try:
    import websocket
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


class TeleopBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('teleop_bridge')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('il_pipeline.webserver.host', 'localhost')
        self.declare_parameter('il_pipeline.webserver.ik_ws_port', 9001)
        self.declare_parameter('il_pipeline.webserver.reconnect_interval_s', 5.0)
        self.declare_parameter('il_pipeline.webserver.gripper_max_position', 0.04)
        self.declare_parameter('il_pipeline.robot.joint_names', ['joint1'])
        self.declare_parameter('il_pipeline.robot.num_joints', 7)

        self._joint_names: list[str] = self.get_parameter(
            'il_pipeline.robot.joint_names').value
        self._gripper_max_pos: float = self.get_parameter(
            'il_pipeline.webserver.gripper_max_position').value

        # ── Publishers ───────────────────────────────────────────────────────
        self._joint_pub = self.create_publisher(
            JointTrajectory, '/joint_commands', 10)
        self._gripper_pub = self.create_publisher(
            Float32, '/gripper_command', 10)

        # ── WebSocket connection (background thread) ─────────────────────────
        self._connected = False
        if not _WS_AVAILABLE:
            self.get_logger().error(
                'websocket-client not installed — TeleopBridgeNode inactive')
        else:
            t = threading.Thread(target=self._connect_loop, daemon=True)
            t.start()
            self.get_logger().info('TeleopBridgeNode started — connecting to IK WebSocket')

    # ── WebSocket loop ───────────────────────────────────────────────────────

    def _connect_loop(self) -> None:
        """Runs in a background thread. Connects and reconnects indefinitely."""
        while rclpy.ok():
            host = self.get_parameter('il_pipeline.webserver.host').value
            port = self.get_parameter('il_pipeline.webserver.ik_ws_port').value
            url = 'ws://%s:%d' % (host, port)
            self.get_logger().info('Connecting to IK WS: %s' % url)
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_close=self._on_close,
                    on_error=self._on_error,
                )
                ws.run_forever()
            except Exception as exc:
                self.get_logger().error('IK WS error: %s' % str(exc))
            self._connected = False
            interval = self.get_parameter(
                'il_pipeline.webserver.reconnect_interval_s').value
            self.get_logger().info(
                'IK WS disconnected — retrying in %.1fs' % interval)
            time.sleep(interval)

    def _on_open(self, ws) -> None:
        self._connected = True
        self.get_logger().info('IK WebSocket connected')

    def _on_close(self, ws, code, msg) -> None:
        self._connected = False
        self.get_logger().info('IK WebSocket closed (code=%s)' % code)

    def _on_error(self, ws, error) -> None:
        self.get_logger().warn('IK WebSocket error: %s' % str(error))

    def _on_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Joint solution broadcast
        if msg.get('success') and 'joint_positions' in msg:
            self._handle_joint_solution(msg)
            return

        # Finger/gripper broadcast
        if 'fingers' in msg:
            self._handle_fingers(msg)

    # ── Message handlers ─────────────────────────────────────────────────────

    def _handle_joint_solution(self, msg: dict) -> None:
        names = msg.get('joint_names', [])
        positions = msg.get('joint_positions', [])
        if not names or not positions or len(names) != len(positions):
            return

        name_to_pos = dict(zip(names, positions))
        ordered = [float(name_to_pos.get(n, 0.0)) for n in self._joint_names]

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(self._joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = ordered
        pt.time_from_start.nanosec = 100_000_000  # 100 ms execution window
        traj.points = [pt]

        self._joint_pub.publish(traj)

    def _handle_fingers(self, msg: dict) -> None:
        fingers = msg.get('fingers', {})
        if not fingers:
            return
        vals = list(fingers.values())
        mean_val = sum(vals) / len(vals)
        gripper = max(0.0, min(1.0, mean_val / self._gripper_max_pos))

        out = Float32()
        out.data = float(gripper)
        self._gripper_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

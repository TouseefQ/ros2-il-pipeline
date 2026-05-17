"""
TeleopBridgeNode — translates MyBotShop webserver teleoperation commands
into ROS2 joint commands for the robot.

WebSocket message format expected from the webserver (subject to change
once webserver access is available):
  {
    "type": "teleop",
    "joint_deltas": [j1, j2, ..., j7],   // rad/s scaled by dt
    "gripper": 0.0                         // 0.0 closed – 1.0 open
  }

Publishes:
  /joint_commands         trajectory_msgs/JointTrajectory
  /gripper_command        std_msgs/Float32

The WebSocket connection is non-blocking: if the server is unavailable
the node logs a warning and retries at reconnect_interval_s.
"""

import json
import threading

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

        self.declare_parameter('il_pipeline.webserver.host', 'localhost')
        self.declare_parameter('il_pipeline.webserver.port', 9000)
        self.declare_parameter('il_pipeline.webserver.reconnect_interval_s', 5.0)
        self.declare_parameter('il_pipeline.robot.joint_names', ['joint1'])
        self.declare_parameter('il_pipeline.robot.num_joints', 7)

        self._joint_names: list[str] = self.get_parameter(
            'il_pipeline.robot.joint_names').value

        # ── Publishers ───────────────────────────────────────────────────────
        self._joint_pub = self.create_publisher(
            JointTrajectory, '/joint_commands', 10)
        self._gripper_pub = self.create_publisher(
            Float32, '/gripper_command', 10)

        # ── WebSocket connection (runs in background thread) ─────────────────
        self._ws: websocket.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None
        self._connected = False

        reconnect_s = self.get_parameter(
            'il_pipeline.webserver.reconnect_interval_s').value
        self._reconnect_timer = self.create_timer(reconnect_s, self._try_connect)

        self.get_logger().info('TeleopBridgeNode ready')

    # ── WebSocket management ─────────────────────────────────────────────────

    def _build_ws_url(self) -> str:
        host = self.get_parameter('il_pipeline.webserver.host').value
        port = self.get_parameter('il_pipeline.webserver.port').value
        return f'ws://{host}:{port}/ws/teleop'

    def _try_connect(self) -> None:
        if self._connected:
            return
        if not _WS_AVAILABLE:
            self.get_logger().warn('websocket-client not installed; bridge inactive')
            return
        # TODO Stage 3: implement WebSocket connect, on_message → _handle_message
        self.get_logger().debug('Attempting webserver connection...')

    def _handle_message(self, ws, raw: str) -> None:  # noqa: ANN001
        # TODO Stage 3: parse JSON, build JointTrajectory, publish
        pass

    # ── Command publishing ───────────────────────────────────────────────────

    def _publish_joint_command(self, deltas: list[float]) -> None:
        # TODO Stage 3: build and publish JointTrajectory from deltas
        pass

    def _publish_gripper_command(self, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        self._gripper_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

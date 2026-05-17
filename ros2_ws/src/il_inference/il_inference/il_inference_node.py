"""
ILInferenceNode — loads a trained policy and publishes joint commands at control_hz.

Services offered:
  ~/load_policy   (LoadPolicy)  — hot-swap the active policy checkpoint

Actions offered:
  ~/run_episode   (RunEpisode)  — execute policy for up to max_steps steps

Subscribes to:
  /joint_states               sensor_msgs/JointState
  /gripper_state              std_msgs/Float32
  /camera/wrist/image_raw     sensor_msgs/Image  (if policy uses images)
  /camera/top/image_raw       sensor_msgs/Image  (if policy uses images)

Publishes:
  /joint_commands             trajectory_msgs/JointTrajectory
  /gripper_command            std_msgs/Float32
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectory

from il_interfaces.srv import LoadPolicy
from il_interfaces.action import RunEpisode

from .policy_loader import PolicyLoader  # implemented in Stage 6


class ILInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__('il_inference')

        self.declare_parameter('il_pipeline.inference.default_checkpoint', '')
        self.declare_parameter('il_pipeline.inference.default_algorithm', 'bc')
        self.declare_parameter('il_pipeline.inference.control_hz', 10.0)
        self.declare_parameter('il_pipeline.inference.action_chunk_size', 10)
        self.declare_parameter('il_pipeline.robot.joint_names', ['joint1'])
        self.declare_parameter('il_pipeline.robot.num_joints', 7)
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('algorithm', 'bc')

        self._policy: PolicyLoader | None = None
        self._active = False
        self._latest_joint_state: JointState | None = None
        self._latest_gripper: float = 0.0
        self._latest_wrist_img: Image | None = None
        self._latest_top_img: Image | None = None

        # ── Subscriptions ────────────────────────────────────────────────────
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self._gripper_sub = self.create_subscription(
            Float32, '/gripper_state', self._gripper_cb, 10)

        # ── Services ─────────────────────────────────────────────────────────
        self._load_srv = self.create_service(
            LoadPolicy, '~/load_policy', self._load_policy_cb)

        # ── Publishers ───────────────────────────────────────────────────────
        self._joint_pub = self.create_publisher(
            JointTrajectory, '/joint_commands', 10)
        self._gripper_pub = self.create_publisher(
            Float32, '/gripper_command', 10)

        # ── Control loop ─────────────────────────────────────────────────────
        hz = self.get_parameter('il_pipeline.inference.control_hz').value
        self._control_timer = self.create_timer(1.0 / hz, self._control_tick)

        # ── Load checkpoint if provided at launch ────────────────────────────
        checkpoint = self.get_parameter('checkpoint').value
        algorithm = self.get_parameter('algorithm').value
        if checkpoint:
            self.get_logger().info('Auto-loading checkpoint: %s (%s)' % (
                checkpoint, algorithm))
            # TODO Stage 6: call _load_policy(checkpoint, algorithm)

        self.get_logger().info('ILInferenceNode ready')

    # ── Subscription callbacks ───────────────────────────────────────────────

    def _joint_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _gripper_cb(self, msg: Float32) -> None:
        self._latest_gripper = float(msg.data)

    # ── Service callbacks ────────────────────────────────────────────────────

    def _load_policy_cb(
        self,
        request: LoadPolicy.Request,
        response: LoadPolicy.Response,
    ) -> LoadPolicy.Response:
        # TODO Stage 6: instantiate PolicyLoader, load checkpoint
        response.success = False
        response.message = 'Not yet implemented — coming in Stage 6'
        return response

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_tick(self) -> None:
        # TODO Stage 6: build observation dict, call policy.predict(), publish
        pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ILInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

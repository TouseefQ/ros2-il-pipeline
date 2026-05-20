"""
ILInferenceNode — loads a trained policy and publishes joint commands at control_hz.

Services offered:
  ~/load_policy   (LoadPolicy)  — hot-swap the active policy checkpoint

Actions offered:
  ~/run_episode   (RunEpisode)  — execute policy for up to max_steps steps

Subscribes to:
  /joint_states               sensor_msgs/JointState
  /gripper_state              std_msgs/Float32
  /il/autonomous_mode         std_msgs/Bool   — True enables continuous control loop
  /il/emergency_stop          std_msgs/Bool   — True halts all autonomous motion
  /camera/wrist/image_raw     sensor_msgs/Image  (only if loaded policy uses images)
  /camera/top/image_raw       sensor_msgs/Image  (only if loaded policy uses images)

Publishes:
  /joint_commands             trajectory_msgs/JointTrajectory
  /gripper_command            std_msgs/Float32
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
import rclpy.executors
import rclpy.time
import tf2_ros
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from il_interfaces.srv import LoadPolicy
from il_interfaces.action import RunEpisode

from .policy_loader import PolicyLoader

try:
    from cv_bridge import CvBridge
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False


def _reorder_to_config(
    js: JointState,
    joint_names: list[str],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract joint position and velocity in the configured joint_names order."""
    pos = np.zeros(n, dtype=np.float32)
    vel = np.zeros(n, dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(js.name)}
    for our_i, name in enumerate(joint_names):
        src = name_to_idx.get(name)
        if src is None:
            continue
        if src < len(js.position):
            pos[our_i] = float(js.position[src])
        if js.velocity and src < len(js.velocity):
            vel[our_i] = float(js.velocity[src])
    return pos, vel


class ILInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__('il_inference')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('il_pipeline.inference.default_checkpoint', '')
        self.declare_parameter('il_pipeline.inference.default_algorithm', 'bc')
        self.declare_parameter('il_pipeline.inference.control_hz', 10.0)
        self.declare_parameter('il_pipeline.inference.action_chunk_size', 10)
        self.declare_parameter('il_pipeline.inference.training_src_dir', '')
        self.declare_parameter(
            'il_pipeline.inference.joint_limits_lower',
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        )
        self.declare_parameter(
            'il_pipeline.inference.joint_limits_upper',
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        )
        self.declare_parameter('il_pipeline.robot.joint_names', ['joint1'])
        self.declare_parameter('il_pipeline.robot.num_joints', 7)
        self.declare_parameter('il_pipeline.robot.eef_link', 'panda_link8')
        self.declare_parameter('il_pipeline.robot.base_frame', 'world')
        # Launch-time overrides (ros2 run il_inference ... --ros-args -p checkpoint:=...)
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('algorithm', 'bc')

        self._joint_names: list[str] = self.get_parameter(
            'il_pipeline.robot.joint_names').value
        self._num_joints: int = self.get_parameter(
            'il_pipeline.robot.num_joints').value
        self._eef_link: str = self.get_parameter(
            'il_pipeline.robot.eef_link').value
        self._base_frame: str = self.get_parameter(
            'il_pipeline.robot.base_frame').value
        self._training_src_dir: str = self.get_parameter(
            'il_pipeline.inference.training_src_dir').value

        lo = self.get_parameter('il_pipeline.inference.joint_limits_lower').value
        hi = self.get_parameter('il_pipeline.inference.joint_limits_upper').value
        self._jlim_lo = np.array(lo, dtype=np.float32)
        self._jlim_hi = np.array(hi, dtype=np.float32)

        # ── State ────────────────────────────────────────────────────────────
        self._policy: PolicyLoader | None = None
        self._policy_lock = threading.Lock()
        self._active: bool = False          # continuous mode (via /il/autonomous_mode)
        self._estop: bool = False           # latched until cleared
        self._action_running: bool = False  # gates control_tick during RunEpisode

        self._latest_joint_state: JointState | None = None
        self._latest_gripper: float = 0.0
        self._latest_wrist_img: Image | None = None
        self._latest_top_img: Image | None = None
        self._cv_bridge = CvBridge() if _CV_AVAILABLE else None
        self._image_subs_created: bool = False

        # ── TF2 ─────────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscriptions ────────────────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Float32, '/gripper_state', self._gripper_cb, 10)
        self.create_subscription(Bool, '/il/autonomous_mode', self._mode_cb, 10)
        self.create_subscription(Bool, '/il/emergency_stop', self._estop_cb, 10)

        # ── Services ─────────────────────────────────────────────────────────
        self.create_service(LoadPolicy, '~/load_policy', self._load_policy_cb)

        # ── Action server ────────────────────────────────────────────────────
        self._action_cb_group = ReentrantCallbackGroup()
        ActionServer(
            self,
            RunEpisode,
            '~/run_episode',
            execute_callback=self._run_episode_execute,
            goal_callback=self._run_episode_goal,
            cancel_callback=self._run_episode_cancel,
            callback_group=self._action_cb_group,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self._joint_pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)
        self._gripper_pub = self.create_publisher(Float32, '/gripper_command', 10)

        # ── Control loop ─────────────────────────────────────────────────────
        hz = self.get_parameter('il_pipeline.inference.control_hz').value
        self.create_timer(1.0 / hz, self._control_tick)

        # ── Auto-load checkpoint ─────────────────────────────────────────────
        checkpoint = self.get_parameter('checkpoint').value
        algorithm  = self.get_parameter('algorithm').value
        if not checkpoint:
            checkpoint = self.get_parameter('il_pipeline.inference.default_checkpoint').value
            algorithm  = self.get_parameter('il_pipeline.inference.default_algorithm').value
        if checkpoint:
            self.get_logger().info('Auto-loading checkpoint: %s (%s)' % (checkpoint, algorithm))
            ok, msg = self._do_load_policy(checkpoint, algorithm)
            if ok:
                self.get_logger().info('Checkpoint loaded: %s' % msg)
            else:
                self.get_logger().error('Auto-load failed: %s' % msg)

        self.get_logger().info(
            'ILInferenceNode ready  joints=%d  hz=%.1f' % (self._num_joints, hz)
        )

    # ── Subscription callbacks ───────────────────────────────────────────────

    def _joint_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _gripper_cb(self, msg: Float32) -> None:
        self._latest_gripper = float(msg.data)

    def _mode_cb(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if self._estop and requested:
            self.get_logger().warn('Autonomous mode blocked — emergency stop is active')
            return
        if requested != self._active:
            self._active = requested
            self.get_logger().info('Mode: %s' % ('AUTONOMOUS' if self._active else 'TELEOP'))

    def _estop_cb(self, msg: Bool) -> None:
        if msg.data:
            self._estop = True
            self._active = False
            self.get_logger().warn('Emergency stop triggered — autonomous motion halted')
        else:
            self._estop = False
            self.get_logger().info('Emergency stop cleared')

    def _wrist_img_cb(self, msg: Image) -> None:
        self._latest_wrist_img = msg

    def _top_img_cb(self, msg: Image) -> None:
        self._latest_top_img = msg

    # ── Policy loading ───────────────────────────────────────────────────────

    def _do_load_policy(self, path: str, algorithm: str) -> tuple[bool, str]:
        """Shared load logic. Returns (success, message)."""
        try:
            loader = PolicyLoader.from_checkpoint(
                path,
                algorithm=algorithm,
                training_src_dir=self._training_src_dir,
            )
        except Exception as exc:
            return False, str(exc)

        if loader.config.num_joints != self._num_joints:
            return False, (
                'Checkpoint has %d joints but robot is configured for %d'
                % (loader.config.num_joints, self._num_joints)
            )

        with self._policy_lock:
            self._policy = loader

        if loader.config.uses_images and not self._image_subs_created:
            self._create_image_subscriptions()

        return True, '%s  joints=%d  checkpoint=%s' % (
            loader.config.algorithm, loader.config.num_joints, path
        )

    def _create_image_subscriptions(self) -> None:
        if _CV_AVAILABLE:
            self.create_subscription(
                Image, '/camera/wrist/image_raw', self._wrist_img_cb, 10)
            self.create_subscription(
                Image, '/camera/top/image_raw', self._top_img_cb, 10)
            self._image_subs_created = True
            self.get_logger().info('Image subscriptions created for visual policy')
        else:
            self.get_logger().warn(
                'Policy uses images but cv_bridge is not available — image obs will be None')

    # ── Service callback ─────────────────────────────────────────────────────

    def _load_policy_cb(
        self,
        request: LoadPolicy.Request,
        response: LoadPolicy.Response,
    ) -> LoadPolicy.Response:
        ok, msg = self._do_load_policy(request.checkpoint_path, request.algorithm)
        response.success = ok
        response.message = msg
        if ok:
            with self._policy_lock:
                p = self._policy
            response.num_joints = p.config.num_joints
            response.algorithm  = p.config.algorithm
        return response

    # ── Action server ────────────────────────────────────────────────────────

    def _run_episode_goal(self, goal_request) -> GoalResponse:
        if self._estop:
            self.get_logger().warn('RunEpisode rejected — emergency stop is active')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _run_episode_cancel(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _run_episode_execute(self, goal_handle) -> RunEpisode.Result:
        goal   = goal_handle.request
        result = RunEpisode.Result()

        # Optionally load a different checkpoint for this episode
        if goal.policy_checkpoint:
            ok, msg = self._do_load_policy(goal.policy_checkpoint, 'bc')
            if not ok:
                result.success = False
                result.steps_executed = 0
                result.message = 'Checkpoint load failed: %s' % msg
                goal_handle.abort()
                return result

        with self._policy_lock:
            policy = self._policy

        if policy is None:
            result.success = False
            result.steps_executed = 0
            result.message = 'No policy loaded'
            goal_handle.abort()
            return result

        hz       = float(goal.control_hz) if goal.control_hz > 0.0 else \
                   self.get_parameter('il_pipeline.inference.control_hz').value
        period   = 1.0 / hz
        max_steps = int(goal.max_steps) if goal.max_steps > 0 else 500
        step = 0

        self._action_running = True
        try:
            while step < max_steps:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.steps_executed = step
                    result.message = 'Cancelled after %d steps' % step
                    return result

                if self._estop:
                    goal_handle.abort()
                    result.success = False
                    result.steps_executed = step
                    result.message = 'Emergency stop after %d steps' % step
                    return result

                published = self._publish_action(policy)
                if published:
                    step += 1
                    feedback = RunEpisode.Feedback()
                    feedback.current_step = step
                    feedback.completion_ratio = step / max_steps
                    if self._latest_joint_state is not None:
                        pos, _ = _reorder_to_config(
                            self._latest_joint_state, self._joint_names, self._num_joints)
                        feedback.current_joint_pos = pos.tolist()
                    goal_handle.publish_feedback(feedback)

                time.sleep(period)
        finally:
            self._action_running = False

        goal_handle.succeed()
        result.success = True
        result.steps_executed = step
        result.message = 'Completed %d steps' % step
        return result

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_tick(self) -> None:
        if not self._active or self._estop or self._action_running:
            return
        with self._policy_lock:
            policy = self._policy
        if policy is None:
            return
        self._publish_action(policy)

    # ── Core inference + publish ─────────────────────────────────────────────

    def _publish_action(self, policy: PolicyLoader) -> bool:
        """Build obs dict, run policy, clamp to joint limits, publish. Returns True on success."""
        if self._latest_joint_state is None:
            return False

        joint_pos, joint_vel = _reorder_to_config(
            self._latest_joint_state, self._joint_names, self._num_joints)
        eef_pos, eef_quat = self._lookup_eef()

        obs: dict = {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'eef_pos':   eef_pos,
            'eef_quat':  eef_quat,
        }

        if policy.config.uses_images:
            obs['camera_wrist'] = self._convert_image(self._latest_wrist_img)
            obs['camera_top']   = self._convert_image(self._latest_top_img)

        try:
            action = policy.predict(obs)
        except Exception as exc:
            self.get_logger().error(
                'Policy predict failed: %s' % exc, throttle_duration_sec=2.0)
            return False

        action = np.clip(action, self._jlim_lo[:len(action)], self._jlim_hi[:len(action)])
        self._publish_joint_trajectory(action)
        return True

    def _publish_joint_trajectory(self, positions: np.ndarray) -> None:
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = list(self._joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100 ms execution window
        msg.points.append(pt)
        self._joint_pub.publish(msg)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _lookup_eef(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame, self._eef_link, rclpy.time.Time())
            tr  = t.transform.translation
            rot = t.transform.rotation
            return (
                np.array([tr.x,  tr.y,  tr.z],           dtype=np.float32),
                np.array([rot.x, rot.y, rot.z, rot.w],   dtype=np.float32),
            )
        except TransformException:
            return (
                np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )

    def _convert_image(self, msg: Image | None) -> np.ndarray | None:
        if msg is None or self._cv_bridge is None:
            return None
        try:
            return self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.uint8)
        except Exception:
            return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ILInferenceNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

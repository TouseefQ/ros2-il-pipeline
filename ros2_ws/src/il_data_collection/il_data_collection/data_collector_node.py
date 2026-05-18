"""
DataCollectorNode — records timestamped observations during teleoperation.

Services offered:
  ~/start_recording  (StartRecording)  — begin a new episode
  ~/stop_recording   (StopRecording)   — end episode and optionally save to HDF5

Subscribes to:
  /joint_states                  sensor_msgs/JointState
  /gripper_state                 std_msgs/Float32  (0.0 closed – 1.0 open)
  /camera/wrist/image_raw        sensor_msgs/Image  (only when cameras configured)
  /camera/top/image_raw          sensor_msgs/Image  (only when cameras configured)

Publishes:
  /il/observation_frame          il_interfaces/ObservationFrame  (live monitor feed)
  /il/episode_status             std_msgs/String
"""

from __future__ import annotations

import uuid

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32, String
import tf2_ros
from tf2_ros import TransformException

from il_interfaces.msg import ObservationFrame
from il_interfaces.srv import StartRecording, StopRecording

from .episode_writer import EpisodeWriter, ObsStep, ActionStep, Step

try:
    from cv_bridge import CvBridge
    import cv2
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False


def _reorder_to_config(
    js: JointState,
    joint_names: list[str],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and velocity in the order of configured joint_names."""
    pos = np.zeros(n, dtype=np.float32)
    vel = np.zeros(n, dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(js.name)}
    for our_i, name in enumerate(joint_names):
        src = name_to_idx.get(name)
        if src is None:
            continue
        if src < len(js.position):
            pos[our_i] = js.position[src]
        if js.velocity and src < len(js.velocity):
            vel[our_i] = js.velocity[src]
    return pos, vel


class DataCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__('data_collector')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('il_pipeline.data_collection.output_dir', '/data/episodes')
        self.declare_parameter('il_pipeline.data_collection.record_images', False)
        self.declare_parameter('il_pipeline.data_collection.image_size', [128, 128])
        self.declare_parameter('il_pipeline.data_collection.recording_hz', 10.0)
        self.declare_parameter('il_pipeline.data_collection.max_episode_steps', 500)
        self.declare_parameter('il_pipeline.robot.joint_names', ['joint1'])
        self.declare_parameter('il_pipeline.robot.num_joints', 7)
        self.declare_parameter('il_pipeline.robot.eef_link', 'panda_link8')
        self.declare_parameter('il_pipeline.robot.base_frame', 'world')
        self.declare_parameter('il_pipeline.robot.name', 'panda')

        self._output_dir: str = self.get_parameter(
            'il_pipeline.data_collection.output_dir').value
        self._cameras_enabled: bool = self.get_parameter(
            'il_pipeline.data_collection.record_images').value
        img_size = self.get_parameter('il_pipeline.data_collection.image_size').value
        self._image_size: tuple[int, int] = (int(img_size[0]), int(img_size[1]))
        self._max_steps: int = self.get_parameter(
            'il_pipeline.data_collection.max_episode_steps').value
        self._joint_names: list[str] = self.get_parameter(
            'il_pipeline.robot.joint_names').value
        self._num_joints: int = self.get_parameter('il_pipeline.robot.num_joints').value
        self._eef_link: str = self.get_parameter('il_pipeline.robot.eef_link').value
        self._base_frame: str = self.get_parameter('il_pipeline.robot.base_frame').value
        self._robot_name: str = self.get_parameter('il_pipeline.robot.name').value

        # ── Internal state ───────────────────────────────────────────────────
        self._recording: bool = False
        self._writer: EpisodeWriter | None = None
        self._prev_joint_pos: np.ndarray | None = None
        self._latest_joint_state: JointState | None = None
        self._latest_gripper: float = 0.0
        self._latest_wrist_img: Image | None = None
        self._latest_top_img: Image | None = None

        if _CV_AVAILABLE and self._cameras_enabled:
            self._cv_bridge = CvBridge()
        else:
            self._cv_bridge = None

        # ── TF2 ─────────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscriptions ────────────────────────────────────────────────────
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self._gripper_sub = self.create_subscription(
            Float32, '/gripper_state', self._gripper_cb, 10)

        if self._cameras_enabled:
            if not _CV_AVAILABLE:
                self.get_logger().warn(
                    'record_images=true but cv_bridge/cv2 not available — images will be skipped')
            self._wrist_sub = self.create_subscription(
                Image, '/camera/wrist/image_raw', self._wrist_img_cb, 10)
            self._top_sub = self.create_subscription(
                Image, '/camera/top/image_raw', self._top_img_cb, 10)

        # ── Services ─────────────────────────────────────────────────────────
        self._start_srv = self.create_service(
            StartRecording, '~/start_recording', self._start_recording_cb)
        self._stop_srv = self.create_service(
            StopRecording, '~/stop_recording', self._stop_recording_cb)

        # ── Publishers ───────────────────────────────────────────────────────
        self._obs_pub = self.create_publisher(
            ObservationFrame, '/il/observation_frame', 10)
        self._status_pub = self.create_publisher(String, '/il/episode_status', 10)

        # ── Recording timer ──────────────────────────────────────────────────
        hz = self.get_parameter('il_pipeline.data_collection.recording_hz').value
        self._timer = self.create_timer(1.0 / hz, self._record_tick)

        self.get_logger().info(
            'DataCollectorNode ready  hz=%.1f  joints=%d  cameras=%s' % (
                hz, self._num_joints, self._cameras_enabled))

    # ── Subscription callbacks ───────────────────────────────────────────────

    def _joint_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _gripper_cb(self, msg: Float32) -> None:
        self._latest_gripper = float(msg.data)

    def _wrist_img_cb(self, msg: Image) -> None:
        self._latest_wrist_img = msg

    def _top_img_cb(self, msg: Image) -> None:
        self._latest_top_img = msg

    # ── Service callbacks ────────────────────────────────────────────────────

    def _start_recording_cb(
        self,
        request: StartRecording.Request,
        response: StartRecording.Response,
    ) -> StartRecording.Response:
        if self._recording:
            response.success = False
            response.episode_id = ''
            response.message = 'Already recording: %s' % self._writer.episode_id
            return response

        episode_id = request.episode_id or ('ep_%s' % uuid.uuid4().hex[:8])
        # Service request can only enable images if cameras are configured at startup
        record_images = request.record_images and self._cameras_enabled

        self._writer = EpisodeWriter(
            output_dir=self._output_dir,
            episode_id=episode_id,
            robot_description=self._robot_name,
            collection_mode=request.collection_mode or 'teleop',
            num_joints=self._num_joints,
            record_images=record_images,
            image_size=self._image_size,
        )
        self._recording = True
        self._prev_joint_pos = None

        self._publish_status('started:%s' % episode_id)
        self.get_logger().info('Started recording: %s  (images=%s)' % (
            episode_id, record_images))

        response.success = True
        response.episode_id = episode_id
        response.message = 'Recording started'
        return response

    def _stop_recording_cb(
        self,
        request: StopRecording.Request,
        response: StopRecording.Response,
    ) -> StopRecording.Response:
        if not self._recording or self._writer is None:
            response.success = False
            response.episode_id = ''
            response.num_steps = 0
            response.message = 'Not currently recording'
            return response

        episode_id = self._writer.episode_id
        num_steps = self._writer.num_steps
        response.episode_id = episode_id
        response.num_steps = num_steps

        if request.save and num_steps > 0:
            try:
                path = self._writer.save()
                self._publish_status('saved:%s' % episode_id)
                self.get_logger().info(
                    'Saved %s  steps=%d  path=%s' % (episode_id, num_steps, path))
                response.message = 'Saved: %s' % path
            except Exception as exc:
                self.get_logger().error('Save failed: %s' % str(exc))
                response.success = False
                response.message = 'Save failed: %s' % str(exc)
                return response
        else:
            self._writer.discard()
            reason = 'empty' if num_steps == 0 else 'discarded'
            self._publish_status('discarded:%s' % episode_id)
            self.get_logger().info(
                'Discarded episode %s (%s)' % (episode_id, reason))
            response.message = 'Discarded'

        self._recording = False
        self._writer = None
        self._prev_joint_pos = None
        response.success = True
        return response

    # ── Timer callback ───────────────────────────────────────────────────────

    def _record_tick(self) -> None:
        if self._latest_joint_state is None:
            return

        now = self.get_clock().now()
        timestamp = now.nanoseconds * 1e-9

        js = self._latest_joint_state
        joint_pos, joint_vel = _reorder_to_config(js, self._joint_names, self._num_joints)
        eef_pos, eef_quat = self._lookup_eef()

        wrist_img = None
        top_img = None
        if self._recording and self._writer is not None and self._writer._record_images:
            wrist_img = self._convert_image(self._latest_wrist_img)
            top_img = self._convert_image(self._latest_top_img)

        obs = ObsStep(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper_state=self._latest_gripper,
            camera_wrist=wrist_img,
            camera_top=top_img,
        )

        self._publish_obs_frame(obs, now)

        if not self._recording or self._writer is None:
            return

        delta = (
            joint_pos - self._prev_joint_pos
            if self._prev_joint_pos is not None
            else np.zeros(self._num_joints, dtype=np.float32)
        )
        self._prev_joint_pos = joint_pos.copy()

        action = ActionStep(
            joint_pos_abs=joint_pos.copy(),
            joint_pos_delta=delta,
        )
        self._writer.add_step(Step(timestamp=timestamp, obs=obs, action=action))

        if self._writer.num_steps >= self._max_steps:
            self.get_logger().warn(
                'Max steps (%d) reached — auto-saving episode %s' % (
                    self._max_steps, self._writer.episode_id))
            self._do_stop(save=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _lookup_eef(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._eef_link,
                rclpy.time.Time(),
            )
            tr = t.transform.translation
            rot = t.transform.rotation
            eef_pos = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
            eef_quat = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float32)
        except TransformException:
            eef_pos = np.zeros(3, dtype=np.float32)
            eef_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return eef_pos, eef_quat

    def _convert_image(self, msg: Image | None) -> np.ndarray | None:
        if msg is None or self._cv_bridge is None:
            return None
        try:
            img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            H, W = self._image_size
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            return img.astype(np.uint8)
        except Exception:
            return None

    def _publish_obs_frame(self, obs: ObsStep, stamp) -> None:
        msg = ObservationFrame()
        msg.stamp = stamp.to_msg()
        msg.joint_pos = obs.joint_pos.tolist()
        msg.joint_vel = obs.joint_vel.tolist()
        msg.eef_pos.x = float(obs.eef_pos[0])
        msg.eef_pos.y = float(obs.eef_pos[1])
        msg.eef_pos.z = float(obs.eef_pos[2])
        msg.eef_quat.x = float(obs.eef_quat[0])
        msg.eef_quat.y = float(obs.eef_quat[1])
        msg.eef_quat.z = float(obs.eef_quat[2])
        msg.eef_quat.w = float(obs.eef_quat[3])
        msg.gripper_state = obs.gripper_state
        self._obs_pub.publish(msg)

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _do_stop(self, save: bool) -> None:
        if self._writer is None:
            return
        episode_id = self._writer.episode_id
        num_steps = self._writer.num_steps
        if save and num_steps > 0:
            try:
                path = self._writer.save()
                self._publish_status('saved:%s' % episode_id)
                self.get_logger().info(
                    'Auto-saved %s  steps=%d  → %s' % (episode_id, num_steps, path))
            except Exception as exc:
                self.get_logger().error('Auto-save failed: %s' % str(exc))
        else:
            self._writer.discard()
            self._publish_status('discarded:%s' % episode_id)
        self._recording = False
        self._writer = None
        self._prev_joint_pos = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

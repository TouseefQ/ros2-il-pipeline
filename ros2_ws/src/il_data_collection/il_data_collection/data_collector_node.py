"""
DataCollectorNode — records timestamped observations during teleoperation.

Services offered:
  ~/start_recording  (StartRecording)  — begin a new episode
  ~/stop_recording   (StopRecording)   — end episode and optionally save to HDF5

Subscribes to:
  /joint_states                  sensor_msgs/JointState
  /gripper_state                 std_msgs/Float32  (0.0 closed – 1.0 open)
  /camera/wrist/image_raw        sensor_msgs/Image  (only when record_images=true)
  /camera/top/image_raw          sensor_msgs/Image  (only when record_images=true)

Publishes:
  /il/observation_frame          il_interfaces/ObservationFrame  (live monitor feed)
  /il/episode_status             std_msgs/String
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32, String

from il_interfaces.msg import ObservationFrame
from il_interfaces.srv import StartRecording, StopRecording

from .episode_writer import EpisodeWriter  # implemented in Stage 2


class DataCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__('data_collector')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('il_pipeline.data_collection.output_dir', '/data/episodes')
        self.declare_parameter('il_pipeline.data_collection.record_images', False)
        self.declare_parameter('il_pipeline.data_collection.recording_hz', 10.0)
        self.declare_parameter('il_pipeline.data_collection.max_episode_steps', 500)
        self.declare_parameter('il_pipeline.robot.joint_names', ['joint1'])
        self.declare_parameter('il_pipeline.robot.num_joints', 7)

        # ── Internal state ───────────────────────────────────────────────────
        self._recording: bool = False
        self._writer: EpisodeWriter | None = None
        self._latest_joint_state: JointState | None = None
        self._latest_gripper: float = 0.0
        self._latest_wrist_img: Image | None = None
        self._latest_top_img: Image | None = None

        # ── Subscriptions ────────────────────────────────────────────────────
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self._gripper_sub = self.create_subscription(
            Float32, '/gripper_state', self._gripper_cb, 10)

        record_images = self.get_parameter(
            'il_pipeline.data_collection.record_images').value
        if record_images:
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

        self.get_logger().info('DataCollectorNode ready (recording_hz=%.1f)' % hz)

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
        # TODO Stage 2: create EpisodeWriter, set self._recording = True
        response.success = False
        response.message = 'Not yet implemented — coming in Stage 2'
        return response

    def _stop_recording_cb(
        self,
        request: StopRecording.Request,
        response: StopRecording.Response,
    ) -> StopRecording.Response:
        # TODO Stage 2: flush writer, optionally discard episode
        response.success = False
        response.message = 'Not yet implemented — coming in Stage 2'
        return response

    # ── Timer callback ───────────────────────────────────────────────────────

    def _record_tick(self) -> None:
        # TODO Stage 2: build ObservationFrame, call writer.add_step(), publish obs
        pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

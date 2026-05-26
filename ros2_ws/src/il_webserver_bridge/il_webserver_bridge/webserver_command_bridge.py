import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String
from il_interfaces.srv import StartRecording, StopRecording, LoadPolicy


class WebserverCommandBridge(Node):
    """
    Translates /il/pipeline_command String messages from libroscustom.py
    into il_interfaces service calls.

    This keeps il_interfaces out of the robot_webserver package entirely —
    libroscustom.py publishes a plain std_msgs/String and this node handles
    the ROS2 service plumbing.

    Commands:
      start_recording  → /data_collector/start_recording (StartRecording)
      stop_recording   → /data_collector/stop_recording  (StopRecording, save=True)
      discard_episode  → /data_collector/stop_recording  (StopRecording, save=False)
      load_policy      → /il_inference/load_policy       (LoadPolicy)
      emergency_stop   → /il/emergency_stop              (Bool, True)
    """

    def __init__(self):
        super().__init__('webserver_command_bridge')

        self.declare_parameter(
            'il_pipeline.inference.default_checkpoint',
            '/workspace/data/models/best.pt')
        self.declare_parameter(
            'il_pipeline.inference.default_algorithm',
            'bc')

        self._default_checkpoint = self.get_parameter(
            'il_pipeline.inference.default_checkpoint'
        ).get_parameter_value().string_value
        self._default_algorithm = self.get_parameter(
            'il_pipeline.inference.default_algorithm'
        ).get_parameter_value().string_value

        self._start_rec_cli = self.create_client(
            StartRecording, '/data_collector/start_recording')
        self._stop_rec_cli = self.create_client(
            StopRecording, '/data_collector/stop_recording')
        self._load_policy_cli = self.create_client(
            LoadPolicy, '/il_inference/load_policy')

        self._estop_pub = self.create_publisher(Bool, '/il/emergency_stop', 10)

        self.create_subscription(String, '/il/pipeline_command', self._on_command, 10)

        self.get_logger().info('webserver_command_bridge ready')

    def _on_command(self, msg: String):
        cmd = msg.data.strip()
        if cmd == 'start_recording':
            self._call_start_recording()
        elif cmd == 'stop_recording':
            self._call_stop_recording(save=True)
        elif cmd == 'discard_episode':
            self._call_stop_recording(save=False)
        elif cmd == 'load_policy':
            self._call_load_policy()
        elif cmd == 'emergency_stop':
            self._estop_pub.publish(Bool(data=True))
            self.get_logger().info('Emergency stop published')
        else:
            self.get_logger().warn(f'Unknown pipeline command: {cmd!r}')

    def _call_start_recording(self):
        if not self._start_rec_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('start_recording service not available')
            return
        req = StartRecording.Request()
        req.collection_mode = 'teleop'
        req.record_images = False
        future = self._start_rec_cli.call_async(req)
        future.add_done_callback(
            lambda f: self._log_response(f, 'start_recording'))

    def _call_stop_recording(self, save: bool):
        if not self._stop_rec_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('stop_recording service not available')
            return
        req = StopRecording.Request()
        req.save = save
        future = self._stop_rec_cli.call_async(req)
        label = 'stop_recording (save)' if save else 'discard_episode'
        future.add_done_callback(lambda f: self._log_response(f, label))

    def _call_load_policy(self):
        if not self._load_policy_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('load_policy service not available')
            return
        req = LoadPolicy.Request()
        req.checkpoint_path = self._default_checkpoint
        req.algorithm = self._default_algorithm
        future = self._load_policy_cli.call_async(req)
        future.add_done_callback(lambda f: self._log_response(f, 'load_policy'))

    def _log_response(self, future, label: str):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'{label} OK: {result.message}')
            else:
                self.get_logger().warn(f'{label} failed: {result.message}')
        except Exception as exc:
            self.get_logger().error(f'{label} call raised: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = WebserverCommandBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

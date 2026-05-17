"""
WebserverActionNode — maps custom action buttons on the MyBotShop webserver
to pipeline control service calls (start/stop recording, load policy, etc.).

Expected WebSocket message format from webserver:
  {
    "type": "action",
    "action_id": "start_recording" | "stop_recording" | "load_policy" | ...,
    "params": { ... }
  }

This node acts as a thin router: it receives webserver button presses
and translates them into calls to the pipeline service endpoints.
"""

import json
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from il_interfaces.srv import StartRecording, StopRecording, LoadPolicy

try:
    import websocket
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

# Maps webserver action_id → handler method name
_ACTION_MAP = {
    'start_recording': '_handle_start_recording',
    'stop_recording':  '_handle_stop_recording',
    'load_policy':     '_handle_load_policy',
    'discard_episode': '_handle_discard_episode',
}


class WebserverActionNode(Node):
    def __init__(self) -> None:
        super().__init__('webserver_actions')

        self.declare_parameter('il_pipeline.webserver.host', 'localhost')
        self.declare_parameter('il_pipeline.webserver.port', 9000)
        self.declare_parameter('il_pipeline.webserver.reconnect_interval_s', 5.0)

        # ── Service clients ──────────────────────────────────────────────────
        self._start_client = self.create_client(
            StartRecording, '/data_collector/start_recording')
        self._stop_client = self.create_client(
            StopRecording, '/data_collector/stop_recording')
        self._load_policy_client = self.create_client(
            LoadPolicy, '/il_inference/load_policy')

        # ── Status publisher (for webserver UI feedback) ─────────────────────
        self._status_pub = self.create_publisher(String, '/il/pipeline_control', 10)

        reconnect_s = self.get_parameter(
            'il_pipeline.webserver.reconnect_interval_s').value
        self._reconnect_timer = self.create_timer(reconnect_s, self._try_connect)

        self.get_logger().info('WebserverActionNode ready')

    # ── WebSocket management ─────────────────────────────────────────────────

    def _try_connect(self) -> None:
        # TODO Stage 3: connect to webserver WebSocket action endpoint
        pass

    def _handle_message(self, ws, raw: str) -> None:  # noqa: ANN001
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if payload.get('type') != 'action':
            return
        action_id = payload.get('action_id', '')
        handler_name = _ACTION_MAP.get(action_id)
        if handler_name:
            getattr(self, handler_name)(payload.get('params', {}))
        else:
            self.get_logger().warn('Unknown webserver action_id: %s' % action_id)

    # ── Action handlers (called from WebSocket thread) ────────────────────────

    def _handle_start_recording(self, params: dict) -> None:
        # TODO Stage 3: async call to start_recording service
        pass

    def _handle_stop_recording(self, params: dict) -> None:
        # TODO Stage 3: async call to stop_recording service
        pass

    def _handle_load_policy(self, params: dict) -> None:
        # TODO Stage 3: async call to load_policy service
        pass

    def _handle_discard_episode(self, params: dict) -> None:
        # TODO Stage 3: stop recording with save=False
        pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebserverActionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

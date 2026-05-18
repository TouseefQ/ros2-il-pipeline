"""
WebserverActionNode — self-hosted HTTP control panel (port 9010) for
pipeline control (start/stop recording, load policy, discard episode).

Since the MyBotShop webserver exposes no extension hooks for external action
triggers, this node runs its own minimal HTTP server that the operator opens
alongside the webserver UI in a browser tab.

HTTP endpoints:
  GET  /                   — minimal HTML control panel
  POST /start_recording    — body JSON: {"episode_id":"","collection_mode":"teleop","record_images":false}
  POST /stop_recording     — saves and stops the active episode
  POST /discard_episode    — stops and discards (no HDF5 written)
  POST /load_policy        — body JSON: {"checkpoint_path":"...","algorithm":"bc"}

All POST handlers block until the ROS2 service call completes (up to
ACTION_TIMEOUT_S seconds) then return a JSON response body.

Service clients:
  /data_collector/start_recording  (StartRecording)
  /data_collector/stop_recording   (StopRecording)
  /il_inference/load_policy        (LoadPolicy)

Publishes:
  /il/pipeline_control   std_msgs/String  (JSON status after every action)
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from il_interfaces.srv import StartRecording, StopRecording, LoadPolicy

ACTION_TIMEOUT_S = 10.0

_HTML_PANEL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IL Pipeline Control</title>
<style>
  body { font-family: sans-serif; max-width: 520px; margin: 40px auto; }
  h1   { font-size: 1.3em; margin-bottom: 1em; }
  .section { border: 1px solid #ccc; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
  h2   { font-size: 1em; margin: 0 0 10px; }
  label { display: block; margin-bottom: 6px; font-size: 0.9em; }
  input[type=text], select { width: 100%; box-sizing: border-box; padding: 4px 6px; }
  button { margin-top: 8px; padding: 6px 18px; cursor: pointer; }
  #status { font-size: 0.85em; white-space: pre-wrap; background: #f5f5f5;
             padding: 10px; border-radius: 4px; min-height: 40px; }
</style>
</head>
<body>
<h1>IL Pipeline Control</h1>

<div class="section">
  <h2>Recording</h2>
  <label>Episode ID (blank = auto)<input type="text" id="ep_id" placeholder="auto"></label>
  <label>Mode
    <select id="mode">
      <option value="teleop">teleop</option>
      <option value="kinesthetic">kinesthetic</option>
      <option value="scripted">scripted</option>
    </select>
  </label>
  <label><input type="checkbox" id="rec_img"> Record images</label>
  <button onclick="startRecording()">Start Recording</button>
  <button onclick="stopRecording()">Stop &amp; Save</button>
  <button onclick="discardEpisode()">Discard Episode</button>
</div>

<div class="section">
  <h2>Policy</h2>
  <label>Checkpoint path<input type="text" id="ckpt_path" placeholder="/data/models/checkpoint.pt"></label>
  <label>Algorithm
    <select id="algo">
      <option value="bc">bc</option>
      <option value="act">act</option>
      <option value="diffusion">diffusion</option>
    </select>
  </label>
  <button onclick="loadPolicy()">Load Policy</button>
</div>

<div class="section">
  <h2>Status</h2>
  <pre id="status">—</pre>
</div>

<script>
async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}
function show(data) {
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
}
async function startRecording() {
  show(await post('/start_recording', {
    episode_id: document.getElementById('ep_id').value,
    collection_mode: document.getElementById('mode').value,
    record_images: document.getElementById('rec_img').checked
  }));
}
async function stopRecording() {
  show(await post('/stop_recording', {}));
}
async function discardEpisode() {
  show(await post('/discard_episode', {}));
}
async function loadPolicy() {
  show(await post('/load_policy', {
    checkpoint_path: document.getElementById('ckpt_path').value,
    algorithm: document.getElementById('algo').value
  }));
}
</script>
</body>
</html>
"""


def _make_handler(node: 'WebserverActionNode'):
    """Return a BaseHTTPRequestHandler subclass that holds a reference to the ROS node."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default access log
            node.get_logger().debug('HTTP %s' % (fmt % args))

        def do_GET(self):
            if self.path != '/':
                self.send_response(404)
                self.end_headers()
                return
            body = _HTML_PANEL.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b'{}'
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                params = {}

            action = self.path.lstrip('/')
            ev = threading.Event()
            result: list[Any] = [None]

            node._action_queue.put((action, params, ev, result))
            ev.wait(timeout=ACTION_TIMEOUT_S)

            if result[0] is None:
                resp = {'success': False, 'message': 'timeout'}
            else:
                resp = result[0]

            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _Handler


class WebserverActionNode(Node):
    def __init__(self) -> None:
        super().__init__('webserver_actions')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('il_pipeline.webserver.action_server_port', 9010)

        port: int = self.get_parameter(
            'il_pipeline.webserver.action_server_port').value

        # ── Service clients ──────────────────────────────────────────────────
        self._start_client = self.create_client(
            StartRecording, '/data_collector/start_recording')
        self._stop_client = self.create_client(
            StopRecording, '/data_collector/stop_recording')
        self._load_policy_client = self.create_client(
            LoadPolicy, '/il_inference/load_policy')

        # ── Publisher ────────────────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, '/il/pipeline_control', 10)

        # ── Thread-safe queue drained by ROS2 timer ──────────────────────────
        self._action_queue: queue.Queue = queue.Queue()
        self.create_timer(0.05, self._drain_queue)  # 20 Hz

        # ── HTTP server in a daemon thread ───────────────────────────────────
        self._http_server = HTTPServer(('0.0.0.0', port), _make_handler(self))
        t = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        t.start()

        self.get_logger().info(
            'WebserverActionNode HTTP panel: http://localhost:%d' % port)

    # ── Queue drain (runs in ROS2 executor thread) ────────────────────────────

    def _drain_queue(self) -> None:
        try:
            action, params, ev, result = self._action_queue.get_nowait()
        except queue.Empty:
            return

        if action == 'start_recording':
            self._call_start_recording(params, ev, result)
        elif action == 'stop_recording':
            self._call_stop_recording(save=True, ev=ev, result=result)
        elif action == 'discard_episode':
            self._call_stop_recording(save=False, ev=ev, result=result)
        elif action == 'load_policy':
            self._call_load_policy(params, ev, result)
        else:
            result[0] = {'success': False, 'message': 'unknown action: %s' % action}
            ev.set()

    # ── Async service dispatchers ─────────────────────────────────────────────

    def _call_start_recording(self, params: dict, ev: threading.Event, result: list) -> None:
        req = StartRecording.Request()
        req.episode_id = str(params.get('episode_id', ''))
        req.collection_mode = str(params.get('collection_mode', 'teleop'))
        req.record_images = bool(params.get('record_images', False))

        future = self._start_client.call_async(req)

        def _done(f):
            try:
                resp = f.result()
                r = {'success': resp.success,
                     'episode_id': resp.episode_id,
                     'message': resp.message}
            except Exception as exc:
                r = {'success': False, 'message': str(exc)}
            self._publish_status(r)
            result[0] = r
            ev.set()

        future.add_done_callback(_done)

    def _call_stop_recording(self, save: bool, ev: threading.Event, result: list) -> None:
        req = StopRecording.Request()
        req.save = save

        future = self._stop_client.call_async(req)

        def _done(f):
            try:
                resp = f.result()
                r = {'success': resp.success,
                     'episode_id': resp.episode_id,
                     'num_steps': resp.num_steps,
                     'message': resp.message}
            except Exception as exc:
                r = {'success': False, 'message': str(exc)}
            self._publish_status(r)
            result[0] = r
            ev.set()

        future.add_done_callback(_done)

    def _call_load_policy(self, params: dict, ev: threading.Event, result: list) -> None:
        req = LoadPolicy.Request()
        req.checkpoint_path = str(params.get('checkpoint_path', ''))
        req.algorithm = str(params.get('algorithm', 'bc'))

        future = self._load_policy_client.call_async(req)

        def _done(f):
            try:
                resp = f.result()
                r = {'success': resp.success,
                     'message': resp.message,
                     'num_joints': resp.num_joints,
                     'algorithm': resp.algorithm}
            except Exception as exc:
                r = {'success': False, 'message': str(exc)}
            self._publish_status(r)
            result[0] = r
            ev.set()

        future.add_done_callback(_done)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_status(self, data: dict) -> None:
        msg = String()
        msg.data = json.dumps(data)
        self._status_pub.publish(msg)

    def destroy_node(self) -> None:
        self._http_server.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebserverActionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

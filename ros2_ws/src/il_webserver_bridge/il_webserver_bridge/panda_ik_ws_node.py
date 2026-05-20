"""
panda_ik_ws_node — Panda joint-state WebSocket adapter.

Subscribes to /joint_states and broadcasts on ws://0.0.0.0:<ik_ws_port>
in the format teleop_bridge_node expects:

    {"joint_names": [...], "joint_positions": [...], "success": true}

This replaces the robot_webserver IK WebSocket (port 9001) when running
the Panda on fake/simulated hardware without the robot_arm_ik package.
"""

import asyncio
import json
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    import websockets
except ImportError:
    raise SystemExit(
        'websockets package required — run: pip3 install websockets'
    )

_PANDA_JOINTS = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
]


class PandaIKWSNode(Node):
    def __init__(self) -> None:
        super().__init__('panda_ik_ws_node')
        self.declare_parameter('il_pipeline.webserver.ik_ws_port', 9001)

        self._port: int = (
            self.get_parameter('il_pipeline.webserver.ik_ws_port')
            .get_parameter_value().integer_value
        )
        self._lock = threading.Lock()
        self._latest: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_clients: set = set()

        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10
        )
        self.get_logger().info(
            f'panda_ik_ws_node ready — will serve on ws://0.0.0.0:{self._port}'
        )

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            positions = [name_to_pos[j] for j in _PANDA_JOINTS]
        except KeyError:
            return  # not all joints present yet

        payload = json.dumps({
            'joint_names':     _PANDA_JOINTS,
            'joint_positions': [round(p, 6) for p in positions],
            'success':         True,
        })
        with self._lock:
            self._latest = payload

        if self._loop is not None and self._ws_clients:
            self._loop.call_soon_threadsafe(self._schedule_broadcast, payload)

    def _schedule_broadcast(self, payload: str) -> None:
        asyncio.ensure_future(self._broadcast(payload))

    async def _broadcast(self, payload: str) -> None:
        dead: set = set()
        for ws in set(self._ws_clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    async def _handler(self, websocket) -> None:
        addr = getattr(websocket, 'remote_address', '?')
        self.get_logger().info(f'WebSocket client connected: {addr}')
        self._ws_clients.add(websocket)
        try:
            with self._lock:
                latest = self._latest
            if latest is not None:
                await websocket.send(latest)
            async for _ in websocket:
                pass  # clients are receive-only; discard any incoming data
        except Exception:
            pass
        finally:
            self._ws_clients.discard(websocket)
            self.get_logger().info(f'WebSocket client disconnected: {addr}')

    async def run_server(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with websockets.serve(self._handler, '0.0.0.0', self._port):
            self.get_logger().info(
                f'WebSocket server listening on ws://0.0.0.0:{self._port}'
            )
            await asyncio.Future()  # run until cancelled


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PandaIKWSNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        asyncio.run(node.run_server())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

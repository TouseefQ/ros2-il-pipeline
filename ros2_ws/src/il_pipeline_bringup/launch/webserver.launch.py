"""
Minimal webserver launch: robot_webserver (port 9000 + ik_ws on 9001) + rosbridge (9004).

Excludes LLM, STT, WebRTC, VNC, and annotation servers — only what's needed
for the browser UI and IK WebSocket in simulation.

Namespace is set by the ROBOT_NS environment variable. Defaults to empty (no
namespace) so that simulation topic paths (e.g. arm_ik_node/cartesian_target)
resolve directly without a robot-unit prefix. Set ROBOT_NS=robot_unit_0 in
the environment to match a real multi-robot deployment.
"""
import os

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource


def generate_launch_description():
    robot_ns = os.environ.get('ROBOT_NS', '')

    config_file = PathJoinSubstitution([
        FindPackageShare('robot_webserver'), 'config', 'robot_webserver.yaml'
    ])

    webserver_node = Node(
        package='robot_webserver',
        executable='webserver',
        name='robot_webserver',
        namespace=robot_ns or None,
        parameters=[config_file],
        output='screen',
    )

    rosbridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('rosbridge_server'),
                'launch', 'rosbridge_websocket_launch.xml',
            ])
        ]),
        launch_arguments={'port': '9004'}.items(),
    )

    return LaunchDescription([
        webserver_node,
        rosbridge,
    ])

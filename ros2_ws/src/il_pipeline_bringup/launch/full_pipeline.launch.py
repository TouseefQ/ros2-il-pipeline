"""
Full pipeline launch: simulator + collection + webserver bridge.

Inference runs in a separate container via the inference docker-compose profile.

Usage:
  ros2 launch il_pipeline_bringup full_pipeline.launch.py

  # With image recording enabled
  ros2 launch il_pipeline_bringup full_pipeline.launch.py record_images:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = FindPackageShare("il_pipeline_bringup")
    config_file   = PathJoinSubstitution([bringup_share, "config", "pipeline.yaml"])

    # ── Simulator ─────────────────────────────────────────────────────────────
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup_share, "launch", "sim.launch.py"])
        ])
    )

    # ── Data collection stack ─────────────────────────────────────────────────
    collection = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup_share, "launch", "collection.launch.py"])
        ]),
        launch_arguments={
            "record_images": LaunchConfiguration("record_images"),
        }.items(),
    )

    # ── Webserver bridge ──────────────────────────────────────────────────────
    teleop_bridge = Node(
        package="il_webserver_bridge",
        executable="teleop_bridge_node",
        name="teleop_bridge",
        parameters=[config_file],
        output="screen",
        remappings=[('/joint_commands', '/panda_arm_controller/joint_trajectory')],
    )

    webserver_actions = Node(
        package="il_webserver_bridge",
        executable="webserver_action_node",
        name="webserver_actions",
        parameters=[config_file],
        output="screen",
    )

    panda_ik_ws = Node(
        package="il_webserver_bridge",
        executable="panda_ik_ws_node",
        name="panda_ik_ws",
        parameters=[config_file],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "record_images", default_value="false",
            description="Include camera images in collected episodes",
        ),
        sim,
        collection,
        teleop_bridge,
        webserver_actions,
        panda_ik_ws,
    ])

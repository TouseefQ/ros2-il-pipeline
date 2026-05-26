"""
Full pipeline launch: simulator + collection + webserver + inference.

Covers the complete loop:
  teleoperation → episode collection → (offline training) → inference

The robot_webserver (port 9000 UI + port 9001 IK WebSocket) replaces the
former panda_ik_ws_node and webserver_action_node:
  - sim_arm_ik_node handles MoveIt2 IK for the webserver's drag-arm UI
  - webserver_command_bridge translates webserver button presses to IL service calls

Usage:
  # Collection only (no policy loaded)
  ros2 launch il_pipeline_bringup full_pipeline.launch.py

  # With a trained policy auto-loaded
  ros2 launch il_pipeline_bringup full_pipeline.launch.py \\
      checkpoint:=/data/models/best.pt algorithm:=bc \\
      training_src_dir:=/path/to/ros2-il-pipeline/training

  # With image recording enabled
  ros2 launch il_pipeline_bringup full_pipeline.launch.py record_images:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    bringup_share = FindPackageShare("il_pipeline_bringup")
    config_file   = PathJoinSubstitution([bringup_share, "config", "pipeline.yaml"])

    # Mirror ROBOT_NS so sim_arm_ik topics align with what ik_ws.py publishes.
    # With ROBOT_NS unset (sim default): arm_ik_ns = "arm_ik_node"
    # With ROBOT_NS=robot_unit_0:        arm_ik_ns = "robot_unit_0/arm_ik_node"
    _robot_ns = os.environ.get('ROBOT_NS', '')
    arm_ik_ns = f'{_robot_ns}/arm_ik_node'.lstrip('/') if _robot_ns else 'arm_ik_node'

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

    sim_arm_ik = Node(
        package="il_webserver_bridge",
        executable="sim_arm_ik_node",
        name="arm_ik_node",
        namespace=arm_ik_ns,
        parameters=[config_file],
        output="screen",
    )

    webserver_cmd_bridge = Node(
        package="il_webserver_bridge",
        executable="webserver_command_bridge",
        name="webserver_command_bridge",
        parameters=[config_file],
        output="screen",
    )

    # ── Robot webserver (port 9000 UI + port 9001 IK WebSocket) ──────────────
    webserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup_share, "launch", "webserver.launch.py"])
        ])
    )

    # ── Inference node (optional — no-ops if no checkpoint given) ─────────────
    inference = Node(
        package="il_inference",
        executable="il_inference_node",
        name="il_inference",
        parameters=[config_file, {
            "checkpoint":    LaunchConfiguration("checkpoint"),
            "algorithm":     LaunchConfiguration("algorithm"),
        }],
        output="screen",
        remappings=[('/joint_commands', '/panda_arm_controller/joint_trajectory')],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "record_images", default_value="false",
            description="Include camera images in collected episodes",
        ),
        DeclareLaunchArgument(
            "checkpoint", default_value="",
            description="Absolute path to a trained policy checkpoint (.pt). "
                        "Leave empty to skip auto-loading.",
        ),
        DeclareLaunchArgument(
            "algorithm", default_value="bc",
            description="Policy algorithm matching the checkpoint: bc | act",
        ),
        DeclareLaunchArgument(
            "training_src_dir", default_value="",
            description="Absolute path to the training/ directory so the inference "
                        "node can import model classes at runtime.",
        ),
        sim,
        collection,
        teleop_bridge,
        sim_arm_ik,
        webserver_cmd_bridge,
        webserver,
        inference,
    ])

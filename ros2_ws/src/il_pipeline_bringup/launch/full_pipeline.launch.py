"""
Full pipeline launch: simulator + collection + webserver bridge + inference.

Covers the complete loop:
  teleoperation → episode collection → (offline training) → inference

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
    )

    webserver_actions = Node(
        package="il_webserver_bridge",
        executable="webserver_action_node",
        name="webserver_actions",
        parameters=[config_file],
        output="screen",
    )

    # ── Inference node (optional — no-ops if no checkpoint given) ─────────────
    inference = Node(
        package="il_inference",
        executable="il_inference_node",
        name="il_inference",
        parameters=[config_file, {
            "checkpoint":    LaunchConfiguration("checkpoint"),
            "algorithm":     LaunchConfiguration("algorithm"),
            "il_pipeline.inference.training_src_dir":
                LaunchConfiguration("training_src_dir"),
        }],
        output="screen",
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
        webserver_actions,
        inference,
    ])

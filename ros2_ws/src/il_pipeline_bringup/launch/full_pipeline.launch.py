"""
Full pipeline launch: simulator + collection nodes + webserver bridge.

Usage:
  ros2 launch il_pipeline_bringup full_pipeline.launch.py
  ros2 launch il_pipeline_bringup full_pipeline.launch.py record_images:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = FindPackageShare("il_pipeline_bringup")
    config_file = PathJoinSubstitution([bringup_share, "config", "pipeline.yaml"])

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup_share, "launch", "sim.launch.py"])
        ])
    )

    collection = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup_share, "launch", "collection.launch.py"])
        ])
    )

    webserver_bridge = Node(
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "record_images", default_value="false",
            description="Include camera images in episodes",
        ),
        sim,
        collection,
        webserver_bridge,
        webserver_actions,
    ])

"""
Launches the data collection stack:
  - data_collector_node
  - dataset_manager_node
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("il_pipeline_bringup"), "config", "pipeline.yaml"
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "record_images",
            default_value="false",
            description="Include raw camera images in episodes",
        ),

        Node(
            package="il_data_collection",
            executable="data_collector_node",
            name="data_collector",
            parameters=[config_file, {
                "il_pipeline.data_collection.record_images":
                    LaunchConfiguration("record_images"),
            }],
            output="screen",
        ),

        Node(
            package="il_data_collection",
            executable="dataset_manager_node",
            name="dataset_manager",
            parameters=[config_file],
            output="screen",
        ),
    ])

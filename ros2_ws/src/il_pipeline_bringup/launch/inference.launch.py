"""
Launches the inference stack:
  - il_inference_node
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
            "checkpoint",
            default_value="",
            description="Path to policy checkpoint (.pt file)",
        ),
        DeclareLaunchArgument(
            "algorithm",
            default_value="bc",
            description="Policy algorithm: bc | act | diffusion",
        ),

        Node(
            package="il_inference",
            executable="il_inference_node",
            name="il_inference",
            parameters=[config_file, {
                "checkpoint": LaunchConfiguration("checkpoint"),
                "algorithm": LaunchConfiguration("algorithm"),
            }],
            output="screen",
        ),
    ])

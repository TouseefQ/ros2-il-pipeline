"""
Launches the MoveIt2 Panda demo on fake hardware.

Provides:
  - robot_state_publisher  (panda URDF)
  - move_group             (MoveIt2 planning)
  - rviz2                  (MoveIt2 plugin)
  - joint_state_broadcaster / fake joint controllers

This is the simulator used for development without real hardware.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    panda_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("moveit_resources_panda_moveit_config"),
                "launch", "demo.launch.py",
            ])
        ])
    )

    return LaunchDescription([panda_demo])

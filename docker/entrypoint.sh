#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# Build workspace if not already built
if [ ! -f /ros2_ws/install/setup.bash ]; then
    echo "[entrypoint] Building ROS2 workspace..."
    cd /ros2_ws
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
fi

source /ros2_ws/install/setup.bash

exec "$@"

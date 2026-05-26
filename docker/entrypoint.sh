#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# Build ros2_ws if not already built
if [ ! -f /ros2_ws/install/setup.bash ]; then
    echo "[entrypoint] Building ROS2 workspace..."
    cd /ros2_ws
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
fi

source /ros2_ws/install/setup.bash

# Build webserver workspace (overlays ros2_ws so robot_webserver can see il_interfaces)
if [ ! -f /webserver_ws/install/setup.bash ]; then
    echo "[entrypoint] Building webserver workspace..."
    cd /webserver_ws
    colcon build --symlink-install \
        --packages-select robot_webserver \
        --cmake-args -DCMAKE_BUILD_TYPE=Release
fi

[ -f /webserver_ws/install/setup.bash ] && source /webserver_ws/install/setup.bash

exec "$@"

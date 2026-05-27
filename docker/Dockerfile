FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-moveit \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-rviz2 \
    ros-humble-ros2bag \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python deps (collection + inference)
RUN pip3 install --no-cache-dir \
    h5py \
    numpy \
    opencv-python-headless \
    websocket-client \
    pyyaml \
    scipy
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /ros2_ws

# Build workspace on first run via entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

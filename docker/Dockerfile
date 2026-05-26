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
    ros-humble-rosbridge-server \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python deps (collection + inference)
RUN pip3 install --no-cache-dir \
    h5py \
    numpy \
    opencv-python-headless \
    websocket-client \
    pyyaml \
    scipy \
    torch --index-url https://download.pytorch.org/whl/cpu

# Webserver Python deps (core web-serving stack; no LLM/STT/TTS)
RUN pip3 install --no-cache-dir \
    flask \
    flask-limiter \
    werkzeug \
    waitress \
    Pillow \
    psutil \
    requests \
    pyproj

WORKDIR /ros2_ws

# Webserver workspace source — built at container start-up via entrypoint
# (must overlay ros2_ws for il_interfaces, so build is deferred to entrypoint)
COPY robot_webserver /webserver_ws/src/

# Build workspace on first run via entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

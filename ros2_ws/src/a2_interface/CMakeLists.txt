cmake_minimum_required(VERSION 3.8)
project(a2_interface)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetLocomotion.srv"
  "srv/PlayEmoji.srv"
  "srv/PlayVideo.srv"
)

ament_package()

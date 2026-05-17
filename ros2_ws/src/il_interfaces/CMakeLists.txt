cmake_minimum_required(VERSION 3.8)
project(il_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Episode.msg"
  "msg/ObservationFrame.msg"
  "srv/StartRecording.srv"
  "srv/StopRecording.srv"
  "srv/LoadPolicy.srv"
  "action/RunEpisode.action"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs builtin_interfaces
)

ament_package()

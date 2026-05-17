# A single timestep observation snapshot published during recording
builtin_interfaces/Time stamp
float32[] joint_pos          # length = num_joints
float32[] joint_vel          # length = num_joints
geometry_msgs/Point eef_pos
geometry_msgs/Quaternion eef_quat
float32 gripper_state        # 0.0 (closed) – 1.0 (open)
# Camera images are zero-size when record_images is false
sensor_msgs/Image camera_wrist
sensor_msgs/Image camera_top

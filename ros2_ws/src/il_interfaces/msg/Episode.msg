# Metadata for a completed or in-progress episode
string episode_id
string robot_description
builtin_interfaces/Time timestamp_start
builtin_interfaces/Time timestamp_end
uint32 num_steps
string collection_mode   # "teleop" | "kinesthetic" | "scripted"
string hdf5_path
bool is_complete

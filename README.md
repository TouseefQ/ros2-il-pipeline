# ROS2 Imitation Learning Pipeline

A hardware-agnostic ROS2 (Humble) pipeline for imitation learning on manipulation tasks, integrated with the [MyBotShop Robotic Webserver](https://docs.mybotshop.de/projects/product_robot_webserver/html/index.html).

The pipeline covers the full loop:

```
Teleoperation → Episode Collection → HDF5 Dataset → Training (BC / ACT) → Inference → Robot
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              MyBotShop Robotic Webserver                 │
│  (port 9000 · WebSocket · Custom Actions · Teleop UI)   │
└──────────────┬──────────────────────┬───────────────────┘
               │ WebSocket            │ WebSocket
               ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│  teleop_bridge_node  │  │  webserver_action_node       │
│  (joystick → joint   │  │  (buttons → start/stop rec,  │
│   commands)          │  │   load policy, mode switch)  │
└──────────┬───────────┘  └──────────────┬───────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────────┐
│                     ROS2 Core Bus                        │
│  /joint_states  /gripper_state  /camera/*/image_raw  /tf │
└──┬───────────────────────┬──────────────────────┬────────┘
   │                       │                      │
   ▼                       ▼                      ▼
┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ data_       │  │ dataset_manager_ │  │ il_inference_    │
│ collector_  │  │ node             │  │ node             │
│ node        │  │ (episode index,  │  │ (loads policy,   │
│ (records    │  │  stats, export)  │  │  obs → action,   │
│  HDF5 eps)  │  └────────┬─────────┘  │  publishes cmds) │
└─────────────┘           │            └──────────────────┘
                          ▼
             ┌────────────────────────┐
             │  data/episodes/*.hdf5  │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │   Training Pipeline    │
             │   train_bc.py          │
             │   train_act.py         │
             └────────────┬───────────┘
                          │  checkpoints
                          ▼
             ┌────────────────────────┐
             │   data/models/*.pt     │
             └────────────────────────┘
```

---

## Repository Structure

```
ros2-il-pipeline/
├── docker/                          # Docker setup
│   ├── Dockerfile                   # ROS2 Humble runtime
│   ├── Dockerfile.training          # PyTorch training environment
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── ros2_ws/src/
│   ├── il_interfaces/               # Custom ROS2 msgs / srvs / actions
│   ├── il_pipeline_bringup/         # Launch files + pipeline.yaml config
│   ├── il_data_collection/          # Data collector + dataset manager nodes
│   ├── il_webserver_bridge/         # Webserver WebSocket ↔ ROS2 bridge
│   └── il_inference/                # Policy inference node
│
├── training/                        # Offline training (pure Python / PyTorch)
│   ├── configs/                     # bc_config.yaml, act_config.yaml
│   ├── datasets/                    # HDF5 → PyTorch Dataset
│   ├── models/                      # BCPolicy, ACTPolicy
│   ├── train_bc.py
│   ├── train_act.py
│   ├── evaluate.py
│   └── requirements.txt
│
├── tools/
│   ├── dataset_viewer.py            # Visualize collected episodes
│   ├── dataset_stats.py             # Episode count / length summary
│   └── export_to_lerobot.py         # Convert to HuggingFace LeRobot format
│
└── data/
    ├── episodes/                    # HDF5 files (git-ignored)
    └── models/                      # Trained checkpoints (git-ignored)
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/TouseefQ/ros2-il-pipeline.git
cd ros2-il-pipeline
```

### 2. Run with Docker (recommended)

```bash
cd docker

# Build images
docker compose build

# Start simulator + collection stack
docker compose up simulator runtime

# (separate terminal) Start full pipeline including webserver bridge
docker compose up
```

### 3. Run natively (ROS2 Humble required)

```bash
# Build workspace
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Launch simulator (MoveIt2 Panda on fake hardware)
ros2 launch il_pipeline_bringup sim.launch.py

# Launch data collection stack
ros2 launch il_pipeline_bringup collection.launch.py

# Start recording an episode
ros2 service call /data_collector/start_recording \
  il_interfaces/srv/StartRecording \
  "{episode_id: '', collection_mode: 'teleop', record_images: false}"

# Stop and save
ros2 service call /data_collector/stop_recording \
  il_interfaces/srv/StopRecording "{save: true}"
```

### 4. Train

```bash
cd training
pip install -r requirements.txt

# Behavior Cloning
python train_bc.py --episodes_dir ../data/episodes/

# ACT
python train_act.py --episodes_dir ../data/episodes/
```

### 5. Run inference

```bash
ros2 launch il_pipeline_bringup inference.launch.py \
  checkpoint:=/data/models/bc_v1.pt algorithm:=bc
```

---

## Simulator

Development uses **MoveIt2 with the Franka Panda on fake hardware** — no Gazebo required.
This gives realistic joint states and a full MoveIt2 planning interface via RViz2.

To swap in a different robot:
1. Edit `ros2_ws/src/il_pipeline_bringup/config/pipeline.yaml` — update `robot:` section
2. Update `sim.launch.py` to point to the new robot's MoveIt2 config package

---

## Dataset Format (HDF5)

Each episode is stored as a single `.hdf5` file:

```
ep_000.hdf5
├── meta/
│   ├── episode_id, robot_description, collection_mode
│   ├── timestamp_start, timestamp_end, num_steps
│   └── record_images
└── data/
    ├── observation/
    │   ├── joint_pos      (T, N_joints)
    │   ├── joint_vel      (T, N_joints)
    │   ├── eef_pos        (T, 3)
    │   ├── eef_quat       (T, 4)
    │   ├── gripper_state  (T, 1)
    │   └── images/
    │       ├── camera_wrist  (T, H, W, 3)  optional
    │       └── camera_top    (T, H, W, 3)  optional
    └── action/
        ├── joint_pos_delta  (T, N_joints)   for BC
        └── joint_pos_abs    (T, N_joints)   for ACT
```

---

## Webserver Integration

The pipeline is designed to connect to the MyBotShop Robotic Webserver (port 9000) via WebSocket.
When webserver access is available:

- `teleop_bridge_node` connects to `ws://<host>:9000/ws/teleop` and translates joystick commands
- `webserver_action_node` connects to the action endpoint and maps custom buttons to pipeline services
- Camera feeds from the webserver are re-published as ROS2 image topics

Without webserver access, all pipeline nodes work standalone via direct ROS2 service calls.

---

## Development Stages

| Stage | Focus | Status |
|-------|-------|--------|
| 1 | Foundation — workspace, interfaces, Docker, skeletons | ✅ Done |
| 2 | Data collection — HDF5 writer, recorder implementation | 🔜 Next |
| 3 | Webserver bridge — WebSocket client, action routing | ⏳ Pending |
| 4 | Dataset tooling — viewer, stats, LeRobot export | ⏳ Pending |
| 5 | BC training — dataset loader, policy, training loop | ⏳ Pending |
| 6 | Inference node — policy loader, control loop | ⏳ Pending |
| 7 | ACT training — transformer model, chunked prediction | ⏳ Pending |
| 8 | Integration & demo | ⏳ Pending |

---

## ROS2 Interface Reference

| Topic / Service | Type | Direction |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | in |
| `/gripper_state` | `std_msgs/Float32` | in |
| `/camera/wrist/image_raw` | `sensor_msgs/Image` | in (optional) |
| `/joint_commands` | `trajectory_msgs/JointTrajectory` | out |
| `/gripper_command` | `std_msgs/Float32` | out |
| `/il/observation_frame` | `il_interfaces/ObservationFrame` | pub |
| `/il/episode_status` | `std_msgs/String` | pub |
| `/il/dataset_stats` | `std_msgs/String` (JSON) | pub |
| `/data_collector/start_recording` | `il_interfaces/srv/StartRecording` | srv |
| `/data_collector/stop_recording` | `il_interfaces/srv/StopRecording` | srv |
| `/il_inference/load_policy` | `il_interfaces/srv/LoadPolicy` | srv |

---

## License

MIT

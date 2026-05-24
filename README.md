# ROS2 Imitation Learning Pipeline

A hardware-agnostic ROS2 (Humble) pipeline for imitation learning on manipulation tasks, integrated with the [MyBotShop Robotic Webserver](https://docs.mybotshop.de/projects/product_robot_webserver/html/index.html).

The pipeline covers the full loop:

```
Teleoperation → Episode Collection → HDF5 Dataset → Training (BC / ACT) → Inference → Robot
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                MyBotShop Robotic Webserver                   │
│  port 9000 (UI)  ·  port 9001 (IK WebSocket broadcast)       │
└───────────────────────────┬──────────────────────────────────┘
                            │ ws://host:9001
                            │ (IK joint solution broadcasts)
                            ▼
               ┌────────────────────────┐
               │   teleop_bridge_node   │
               │   (joint solutions →   │
               │    /joint_commands,    │
               │    fingers →           │
               │    /gripper_command)   │
               └────────────┬───────────┘
                            │
 Browser tab ───────────────│──────────────────────────────────
 http://localhost:9010      ▼
               ┌──────────────────────────┐
               │   webserver_action_node  │
               │   (HTTP server port 9010 │
               │    → start/stop rec,     │
               │      load policy)        │
               └────────────┬─────────────┘
                            │  services / topics
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                       ROS2 Core Bus                          │
│   /joint_states  /gripper_state  /camera/*/image_raw  /tf    │
└──┬────────────────────────┬──────────────────────┬───────────┘
   │                        │                      │
   ▼                        ▼                      ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ data_        │  │ dataset_manager_ │  │ il_inference_    │
│ collector_   │  │ node             │  │ node             │
│ node         │  │ (episode index,  │  │ (loads policy,   │
│ (records     │  │  stats)          │  │  obs → action,   │
│  HDF5 eps)   │  └────────┬─────────┘  │  publishes cmds) │
└──────────────┘           │            └──────────────────┘
                           ▼
            ┌──────────────────────────┐
            │  data/episodes/*.hdf5    │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   Training Pipeline      │
            │   train_bc.py            │
            │   train_act.py           │
            └─────────────┬────────────┘
                          │  checkpoints
                          ▼
            ┌──────────────────────────┐
            │   data/models/*.pt       │
            └──────────────────────────┘
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

## End-to-End Workflow (Docker)

All commands below run from the **project root** (`~/ros2-il-pipeline`), not from inside `docker/`.

### Step 1 — Clone and prepare data directories

```bash
git clone https://github.com/TouseefQ/ros2-il-pipeline.git
cd ros2-il-pipeline
mkdir -p data/episodes data/models
```

### Step 2 — Build Docker images

```bash
docker compose -f docker/docker-compose.yml build
```

### Step 3 — Start the runtime stack

```bash
docker compose -f docker/docker-compose.yml up
```

This starts: MoveIt2 Panda simulator · RViz2 · MoveGroup · ros2_control · data collector · dataset manager · webserver bridge · inference node · IL Pipeline Control panel.

Wait for the log line:
```
[il_inference_node-14] [INFO] [...] [il_inference]: ILInferenceNode ready  joints=7  hz=10.0
```

> **Note:** Every time RViz2 opens, add the Motion Planning panel manually:
> **Panels → Add New Panel → MotionPlanning** — this is required for the arm to respond to trajectory commands.

### Step 4 — Collect demonstration episodes

1. Open **http://localhost:9010** (IL Pipeline Control) in a browser
2. Use the **Start Recording** button to begin an episode
3. Move the arm in RViz2 using the Motion Planning panel (drag the end-effector ball, then click **Plan & Execute**)
4. Click **Stop & Save** when done
5. Repeat for at least 20 episodes — more is better

Episodes are saved to `data/episodes/` on the host as `.hdf5` files.

### Step 5 — Train a BC policy

Open a **second terminal** and run:

```bash
docker compose -f docker/docker-compose.yml --profile training run --rm training \
  python training/train_bc.py \
  --config /workspace/training/configs/bc_config.yaml
```

Training runs for 100 epochs by default. When complete, `data/models/best.pt` will be on the host.

### Step 6 — Run inference (autonomous arm control)

The runtime stack from Step 3 must still be running. The inference node auto-loads `data/models/best.pt` at startup — if you just trained, restart the stack first so it picks up the new checkpoint:

```bash
# Press Ctrl+C to stop, then:
docker compose -f docker/docker-compose.yml up
```

In a second terminal, stop the teleop nodes (they flood the arm controller and block inference):

```bash
docker exec ros2-il-pipeline-ros-1 pkill -f teleop_bridge_node
docker exec ros2-il-pipeline-ros-1 pkill -f panda_ik_ws_node
```

Enable autonomous mode:

```bash
docker exec ros2-il-pipeline-ros-1 bash -c \
  "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && \
   ros2 topic pub --once /il/autonomous_mode std_msgs/Bool 'data: true'"
```

The arm will begin moving in RViz2 based on the trained policy.

### Step 7 — Stop autonomous mode

```bash
docker exec ros2-il-pipeline-ros-1 bash -c \
  "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && \
   ros2 topic pub --once /il/autonomous_mode std_msgs/Bool 'data: false'"
```

Or press `Ctrl+C` in the docker compose terminal to stop everything.

---

## Improving Policy Quality

The quality of autonomous motion depends entirely on the training data:

| Episodes | Expected result |
|---|---|
| < 10 | Near-random motion |
| 20–50 | Rough approximation of demonstrated behaviour |
| 100+ | Consistent task execution |

To improve: collect more demonstrations → retrain → restart the stack.

---

## Configuration

All pipeline parameters are in `ros2_ws/src/il_pipeline_bringup/config/pipeline.yaml`.

Key settings:

| Parameter | Default | Description |
|---|---|---|
| `robot.name` | `panda` | Robot name |
| `robot.num_joints` | `7` | Number of arm joints |
| `data_collection.output_dir` | `/workspace/data/episodes` | Where HDF5 episodes are saved |
| `inference.default_checkpoint` | `/workspace/data/models/best.pt` | Checkpoint loaded at startup |
| `inference.training_src_dir` | `/workspace/training` | Path to training source (needed for policy imports) |
| `inference.control_hz` | `10.0` | Policy inference rate |
| `webserver.port` | `9000` | Robotic Webserver UI port |
| `webserver.action_server_port` | `9010` | IL Pipeline Control panel port |

---

## Simulator

Development uses **MoveIt2 with the Franka Panda on fake hardware** — no Gazebo required. This gives realistic joint states and a full MoveIt2 planning interface via RViz2.

To swap in a different robot:
1. Edit `pipeline.yaml` — update the `robot:` section
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

The pipeline integrates with the MyBotShop Robotic Webserver as a **pure client** — no webserver source files are modified.

**Teleoperation** (`teleop_bridge_node`): connects as a WebSocket client to the IK solver broadcast endpoint (`ws://<host>:9001`). Every joint solution the IK server computes is forwarded to `/joint_commands`; finger/gripper broadcasts are forwarded to `/gripper_command`.

**Pipeline control** (`webserver_action_node`): hosts its own minimal HTTP control panel on port 9010. Open `http://localhost:9010` in a browser tab to start/stop recording, discard an episode, or load a policy checkpoint.

---

## Development Stages

| Stage | Focus | Status |
|-------|-------|--------|
| 1 | Foundation — workspace, interfaces, Docker, skeletons | ✅ Done |
| 2 | Data collection — HDF5 writer, recorder implementation | ✅ Done |
| 3 | Webserver bridge — IK WS client, HTTP control panel | ✅ Done |
| 4 | Dataset tooling — viewer, stats, LeRobot export | ✅ Done |
| 5 | BC training — dataset loader, policy, training loop | ✅ Done |
| 6 | Inference node — policy loader, control loop | ✅ Done |
| 7 | ACT training — transformer model, chunked prediction | ✅ Done |
| 8 | Integration & demo — full launch, CI | ✅ Done |

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
| `/il/pipeline_control` | `std_msgs/String` (JSON) | pub |
| `/data_collector/start_recording` | `il_interfaces/srv/StartRecording` | srv |
| `/data_collector/stop_recording` | `il_interfaces/srv/StopRecording` | srv |
| `/il_inference/load_policy` | `il_interfaces/srv/LoadPolicy` | srv |
| `/il/autonomous_mode` | `std_msgs/Bool` | in — enables continuous policy control |
| `/il/emergency_stop` | `std_msgs/Bool` | in — latching halt of autonomous motion |
| `/il_inference/run_episode` | `il_interfaces/action/RunEpisode` | action |

---

## License

MIT

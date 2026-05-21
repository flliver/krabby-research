# HAL data collector (rosbag2 / mcap)

The collector exists so you can **keep a time-aligned trace of what the robot (or sim) actually sensed and reported** while policies run—useful for debugging behaviors, comparing runs, building datasets, and replaying or analyzing data later with normal ROS 2 bag tooling. It turns the same **`HardwareObservations`** stream the policy already consumes into **standard rosbag2 v9 / mcap** files without running a separate recorder service or a full ROS 2 distro in the container (**[rosbags](https://pypi.org/project/rosbags/)** writes the bags).

That is done with a **second `HalClient`** in the **same process** as the primary client (Jetson `hal.server.jetson.main` or Isaac `hal.server.isaac.main`), on the **same in-proc ZMQ endpoints** and shared **`zmq.Context`**, so the recording matches the observation contract the stack already publishes.

## Run (Jetson or Isaac Sim container)

1. Build **locomotion** (`images/locomotion/Dockerfile`) or **Isaac Sim** (`images/isaacsim/Dockerfile`). Both images include `data_collection/` and pin **`rosbags`** (and **`PyYAML`** for optional `load_config` / tests) in their `requirements.txt`.
2. Use either:
   - Python defaults in **`data_collection/collector_settings.py`**, or
   - your own YAML file via **`--data-collector-config <path>`**.
3. Pass **`--data-collector-output-dir <container_path>`** (optional) to override `output_dir` from config/defaults.
4. Mount a host directory to the chosen container output path so bags persist after container exit.

### Host mount requirement

Recordings are written to the container filesystem unless you mount a host path. Use a bind mount so rosbags persist after container exit.

Create a host directory once:

```bash
mkdir -p /path/to/krabby_bags
```

### Jetson (locomotion image)

Collector output path in container (`/workspace/bags`):

```bash
docker run --rm --runtime=nvidia \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -v /path/to/krabby_bags:/workspace/bags \
  -v /path/to/collector.yaml:/workspace/config/collector.yaml \
  -v /dev:/dev \
  --privileged \
  krabby-locomotion:latest \
  --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
  --data-collector-config /workspace/config/collector.yaml \
  --data-collector-output-dir /workspace/bags
```

### Isaac Sim image (`krabby-isaacsim`)

Collector output path in container (`/workspace/bags`):

```bash
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -v /path/to/krabby_bags:/workspace/bags \
  -v /path/to/collector.yaml:/workspace/config/collector.yaml \
  krabby-isaacsim:latest \
  --task 'Your-Task-v0' \
  --checkpoint /workspace/checkpoints/your.ckpt \
  --data-collector-config /workspace/config/collector.yaml \
  --data-collector-output-dir /workspace/bags
```

If you launch from an interactive shell inside a running container (instead of `docker run`), pass `--data-collector-config` and/or `--data-collector-output-dir`; the chosen output path must be on a mounted volume if you want host persistence.

## Configuration (YAML + Python defaults)

If **`--data-collector-config`** is provided, the entrypoint loads that YAML and uses it for collector settings. If no config path is provided, defaults are assembled from **`data_collection/collector_settings.py`** via `build_data_collector_config(...)`. In all cases, entrypoints enforce HAL endpoints from runtime server wiring so collector transport matches the primary HAL client endpoints.

Canonical example in-repo: **`data_collection/config/collector.yaml`**. Copy or mount it, then override `output_dir` at runtime with **`--data-collector-output-dir`** when using Docker bind mounts (for example `/workspace/bags`).

Example YAML (`collector.yaml`):

```yaml
hal:
  observation_endpoint: "inproc://hal_observation"
  command_endpoint: "inproc://hal_commands"
output_dir: "/data/krabby_bags"
max_disk_usage_fraction: 0.5
rotation_max_bytes: 1073741824
rotation_max_minutes: 30.0
rates:
  images_hz: 10.0
  joints_imu_hz: 50.0
topics:
  joints_state: true
  joints_command: true
  imu: true
joint_names: []
joints_command_source: "previous_action"
polling_timeout_ms: 10
```

| Area | Where it is set |
|------|-----------------|
| HAL transport | Always enforced from entrypoint runtime endpoints (must match primary `HalClientConfig`; Jetson uses fixed inproc endpoints). |
| Output | From YAML/default config (`output_dir`) with optional CLI override via **`--data-collector-output-dir`**. |
| Rates | **`RECORDING_RATES`** — wall-clock caps; actual rate is still bounded by HAL publish rate and **latest-only** `poll()` semantics. |
| Topics | **`TOPIC_ENABLE`** — booleans for `/joints/*` and `/imu` only; catalog cameras are always recorded when present in **`rgbd_by_catalog_id`**. |
| Commands | **`joints_command_source`** is only **`previous_action`** in code — `/joints/command` is filled from **`HardwareObservations.previous_action`**, not from a separate tap on `put_joint_command`. |
| Joints | **`JOINT_NAMES`** — if empty or length mismatch, serialization uses `joint_0`, `joint_1`, … |
| Polling | **`POLLING_TIMEOUT_MS`** — ZMQ receive timeout per collector poll. |

## Topic map (HAL → ROS)

| HAL source | ROS topic | Message type |
|------------|-----------|--------------|
| `rgbd_by_catalog_id[id].rgb` | `/camera/{id}/rgb` | `sensor_msgs/Image` (`rgb8`, or `mono8` when the catalog row’s RGB is 2D) |
| `rgbd_by_catalog_id[id].depth` | `/camera/{id}/depth` | `sensor_msgs/Image` (`32FC1`, meters) |
| `joint_positions`, `joint_velocities` | `/joints/state` | `sensor_msgs/JointState` |
| `previous_action` | `/joints/command` | `sensor_msgs/JointState` (positions only; see above) |
| `base_quat_w`, `base_ang_vel_b` | `/imu` | `sensor_msgs/Imu` — orientation + angular velocity; linear acceleration is zero (not estimated here). |

## Playback

- **ROS 2:** `ros2 bag info` / `ros2 bag play` on the bag directory.
- **This repo:** `python scripts/playback_krabby_bag.py <bag_dir> --topic /camera/front_rgbd/rgb --max 5` (optional `--display` if OpenCV is installed).

## Architecture

The collector is not a separate service. When **`--data-collector-output-dir`** is set, the HAL server entrypoint (`hal.server.jetson.main` or `hal.server.isaac.main`) constructs a **second `HalClient`** in the **same process** as the policy client, sharing the **`zmq.Context`** and using the **same** observation and command endpoint strings as the primary `HalClientConfig`. Both subscribers receive the same published `HardwareObservations`; the collector only serializes and writes them. That keeps recordings aligned with what the stack already exposes, avoids an extra HAL bridge or network hop, and leaves HAL’s contract unchanged—recording is an optional side path in the existing process.

Serialization and bags use **[rosbags](https://pypi.org/project/rosbags/)** (rosbag2 layout, no full ROS 2 distro required). If `rosbags` cannot be imported, the writer is skipped and the rest of the server still runs.

## Recording lifecycle

Under `output_dir`, each **segment** is a standard **rosbag2 v9** directory with **mcap** storage, named `krabby_<seq>_<YYYYMMDD_HHMMSS>`, containing `metadata.yaml` and mcap data—compatible with `ros2 bag info` / `ros2 bag play` and the repo playback helper in the section above.

**Rotation:** when the current segment’s on-disk size reaches `rotation_max_bytes` **or** its age reaches `rotation_max_minutes`, the writer closes that directory and opens a new one (sequence increments).

**Disk cap:** `max_disk_usage_fraction` applies to the **filesystem** that backs `output_dir` (`total` capacity from the OS). The implementation sums sizes of all bag directories under `output_dir` (those with `metadata.yaml`) and deletes **oldest** bags first until usage is under the limit. Mount `output_dir` on a volume sized for the retention you want.

**Effective sample rate:** `rates.images_hz` and `rates.joints_imu_hz` cap how often the collector samples; together with HAL’s **latest-only** subscriber behavior and the publisher’s own rate, the bag may contain fewer messages than a naive “Hz × duration” estimate.

## Configuration semantics

- **`topics.*`:** Disabling `joints_state`, `joints_command`, or `imu` removes those topics from the bag. Catalog camera topics are written for every key in `HardwareObservations.rgbd_by_catalog_id` on each sampled observation (authoritative catalog list: `JETSON_SENSOR_CATALOG` in `hal/server/jetson/sensor_backend_jetson.py`; overview in [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md)).
- **HAL endpoints:** YAML values are overridden by entrypoint runtime endpoint wiring so transport always matches the primary client.
- **Source of truth:** Keep defaults in **`collector_settings.py`**; use **`--data-collector-config`** only when you want per-run YAML overrides.

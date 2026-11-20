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
min_free_fraction: 0.10
rotation_max_bytes: 262144000
rotation_max_minutes: 5.0
camera_encode: "h265"
rates:
  default_images_hz: 10.0
  images_hz_by_catalog_id:
    front_rgbd: 10.0
    belly_left_tof: 20.0
    belly_right_tof: 20.0
    rear_tof: 20.0
  joints_imu_hz: 50.0
topics:
  joints_state: true
  joints_command: true
  imu: true
  base_twist: true
joint_names: []
joints_command_source: "previous_action"
polling_timeout_ms: 10
```

| Area | Where it is set |
|------|-----------------|
| HAL transport | Always enforced from entrypoint runtime endpoints (must match primary `HalClientConfig`; Jetson uses fixed inproc endpoints). |
| Output | From YAML/default config (`output_dir`) with optional CLI override via **`--data-collector-output-dir`**. |
| Encode | **`camera_encode`**: `"h265"` (default) writes `sensor_msgs/CompressedImage` HEVC; `"raw"` keeps uncompressed `sensor_msgs/Image` for bench work. |
| Rates | **`RECORDING_RATES`** — per-catalog-id wall-clock caps (`images_hz_by_catalog_id`, else `default_images_hz`); still bounded by HAL publish rate and **latest-only** `poll()`. |
| Topics | **`TOPIC_ENABLE`** — booleans for `/joints/*`, `/imu`, and `/base/twist`; catalog cameras are always recorded when present in **`rgbd_by_catalog_id`**. With H.265, topics are **`/camera/{catalog_id}/{channel}_h265`**. Bags pre-register from **`hal_rgbd_catalog_channels_for_recording()`** (depth-only catalog rows register only depth). Image **`frame_id`** matches the topic. |
| Commands | **`joints_command_source`** is only **`previous_action`** in code — `/joints/command` is filled from **`HardwareObservations.previous_action`**, not from a separate tap on `put_joint_command`. |
| Joints | **`JOINT_NAMES`** — if empty or length mismatch, serialization uses `joint_0`, `joint_1`, … |
| Polling | **`POLLING_TIMEOUT_MS`** — ZMQ receive timeout per collector poll. |

## Topic map (HAL → ROS)

| HAL source | ROS topic (H.265 default) | Message type |
|------------|---------------------------|--------------|
| `rgbd_by_catalog_id[id].rgb` | `/camera/{id}/rgb_h265` | `sensor_msgs/CompressedImage` (`format=hevc`) |
| `rgbd_by_catalog_id[id].depth` | `/camera/{id}/depth_h265` | `sensor_msgs/CompressedImage` (`format=hevc; range=d_min,d_max`) |
| same channels with `camera_encode=raw` | `/camera/{id}/rgb`, `/camera/{id}/depth` | `sensor_msgs/Image` (`rgb8`/`mono8`, `32FC1`) |
| `joint_positions`, `joint_velocities` | `/joints/state` | `sensor_msgs/JointState` |
| `previous_action` | `/joints/command` | `sensor_msgs/JointState` (positions only; see above) |
| `base_quat_w`, `base_ang_vel_b` | `/imu` | `sensor_msgs/Imu` — orientation + angular velocity; linear acceleration is zero (not on `HardwareObservations`). |
| `base_lin_vel_b`, `base_ang_vel_b` | `/base/twist` | `geometry_msgs/TwistStamped` — body-frame linear + angular velocity (e.g. ZED tracking + IMU via Jetson HAL). |

Depth is quantized to 8-bit gray over the catalog `gst_depth_quant_range_m` band before HEVC (about 5.3 mm/step on the 0.15–1.5 m ToF band). Encoding runs on the collector thread (`nvv4l2h265enc` on Jetson when present, else `x265enc`), never on the 100 Hz control path. If NVENC open or encode fails at runtime (e.g. unsupported caps), the encoder **sticks to `x265enc` for the rest of the process** — bags still record; look for fallback log lines rather than treating NVENC errors as a hard stop.

## Playback

- **ROS 2:** `ros2 bag info` / `ros2 bag play` on the bag directory (standard rosbag2 layout under `output_dir`).
- **H.265 decode / metric depth:** `python -m data_collection.tools.decode_h265 <bag_dir> --catalog-id <id> --channel rgb|depth [--out-dir …]`. Depth reconstructs float32 meters from `format`’s `range=` metadata. `--check-roundtrip` asserts 8-bit quant error stays within one step (no HEVC).
- **Training / S3 consumers:** end-to-end download, replay, and decode guide in [CAMERA_RECORDINGS.md](CAMERA_RECORDINGS.md).

## Architecture

The collector is not a separate service. When **`--data-collector-output-dir`** is set, the HAL server entrypoint (`hal.server.jetson.main` or `hal.server.isaac.main`) constructs a **second `HalClient`** in the **same process** as the policy client, sharing the **`zmq.Context`** and using the **same** observation and command endpoint strings as the primary `HalClientConfig`. Both subscribers receive the same published `HardwareObservations`; the collector only serializes and writes them. That keeps recordings aligned with what the stack already exposes, avoids an extra HAL bridge or network hop, and leaves HAL’s contract unchanged—recording is an optional side path in the existing process.

Serialization and bags use **[rosbags](https://pypi.org/project/rosbags/)** (rosbag2 layout, no full ROS 2 distro required). If `rosbags` cannot be imported, the writer is skipped and the rest of the server still runs.

## Recording lifecycle

Under `output_dir`, each **segment** is a standard **rosbag2 v9** directory with **mcap** storage, named `krabby_<seq>_<YYYYMMDD_HHMMSS>`, containing `metadata.yaml` and mcap data—compatible with `ros2 bag info` / `ros2 bag play`.

**Rotation:** when the current segment’s on-disk size reaches `rotation_max_bytes` **or** its age reaches `rotation_max_minutes`, the writer closes that directory and opens a new one (sequence increments).

**Disk cap:** `min_free_fraction` is the share of filesystem **total** capacity that must remain free. The max allowed bag bytes is `filesystem_free + bag_bytes_in_output_dir - (min_free_fraction × total)` — existing bags count as reclaimable space. When summed bag size exceeds that limit, **oldest** bags are deleted first. Quota is checked when opening each segment (startup and rotation). Mount `output_dir` on a volume sized for the retention you want.

**Effective sample rate:** per-catalog `images_hz_for(id)` and `rates.joints_imu_hz` cap how often the collector samples; together with HAL’s **latest-only** subscriber behavior and the publisher’s own rate, the bag may contain fewer messages than a naive “Hz × duration” estimate. Default segment length is **5 minutes** (250 MiB size cap still applies).

## Fleet robots

On a fleet-enrolled Orin, set `"data_collector": true` in `/etc/krabby/locomotion.json`. `krabby run` then passes `--data-collector-output-dir /data/krabby_bags` and bind-mounts host `/var/krabby/bags`. The host-side `krabby agent` offloader uploads closed segments to S3 when `"bags_offload": true` — download, auth, and replay: [CAMERA_RECORDINGS.md](CAMERA_RECORDINGS.md). The release locomotion image must include the `krabby-data-collection` wheel.

## Configuration semantics

- **`topics.*`:** Disabling `joints_state`, `joints_command`, `imu`, or `base_twist` removes those topics from the bag. Catalog camera topics are written for every key in `HardwareObservations.rgbd_by_catalog_id` on each sampled observation (authoritative catalog list: `JETSON_SENSOR_CATALOG` in `hal/server/jetson/sensor_backend_jetson.py`; overview in [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md)).
- **HAL endpoints:** YAML values are overridden by entrypoint runtime endpoint wiring so transport always matches the primary client.
- **Source of truth:** Keep defaults in **`collector_settings.py`**; use **`--data-collector-config`** only when you want per-run YAML overrides.

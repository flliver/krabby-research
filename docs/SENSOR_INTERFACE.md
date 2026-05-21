# GStreamer multi-sensor interface (HAL)

The HAL exposes a **GStreamer-based sensor interface** so clients can discover sensors, obtain pipeline handles, and build encoded or raw video pipelines that work the same on **Jetson** (multi-sensor on-robot hardware, **nvenc** when available or software encode) and **Isaac Sim** (synthetic sensors + software encode).

## What is the GStreamer multi-sensor interface?

- **GStreamer** is an open-source framework for building media pipelines: sources (cameras, files, synthetic) feed into filters and encoders, and sinks (displays, files, network). Pipelines are described as strings or graphs of elements (e.g. `videotestsrc ! videoconvert ! autovideosink`). Using GStreamer lets the same code path produce **encoded streams** (e.g. H.264 for recording or streaming) or **raw frames** (for visualization or ML), and lets the robot stack use hardware encoders on Jetson (nvenc) or software encoders in simulation (x264).
- **Multi-sensor** means the robot can have several heterogeneous sensors (front depth camera, side RGB cameras, side RGB-D, radar). The interface treats each row uniformly: you **list** what the backend exposes, map a **`SensorInfo`** to a **handle**, and **build a pipeline** per stream. That way clients don’t care whether a sensor is a ZED, a USB camera, or a synthetic Isaac sensor—they use the same API.
- **Interface** here is the HAL API: `list_sensors()`, `get_gstreamer_handle(sensor)`, and `build_pipeline(handle, ...)`. **`get_gstreamer_handle`** is a **pure mapping** from the fields on **`sensor`** to a **`GStreamerHandle`** (it does not reject unknown `sensor.id` values). Typical use is to pass **`SensorInfo`** from **`list_sensors()`**; you may also construct **`SensorInfo`** yourself if caps and metadata are intentional. Implementations exist for **Jetson** (real hardware catalog) and **Isaac Sim** (scene introspection or explicit config). The pipelines produced use **appsrc** as the source so your code (ZED SDK, Isaac render, etc.) pushes frames into GStreamer; the interface does not run the pipelines itself.

## Concepts and terminology

| Term | Meaning |
|------|--------|
| **Backend** | Implementation of the sensor interface: **Jetson** (real ZED, USB cameras, radar on robot) or **Isaac** (synthetic sensors in simulation). Same API, different data source. |
| **SensorInfo** | Read-only metadata for one sensor: `id`, `type`, `modality`, `resolution`, `fps`, `pose`, **`camera_driver`** (HAL capture driver id when applicable, from the Jetson catalog or Isaac introspection; `None` for sensors with no in-process driver). Returned by `list_sensors()`. |
| **GStreamerHandle** | Opaque handle for one sensor, used only to call `build_pipeline(handle, ...)`. Carries `sensor_id`, `sensor_type`, `modality`, `resolution`, `fps`, **`camera_driver`** (mirrors `SensorInfo`), **`appsrc_pixel_format`** (caps for `video/x-raw,format=...`, e.g. `RGB`, `GRAY8`, **`GRAY16_LE`** for metric depth Gst), optional **`depth_range_m`** `(d_min, d_max)` in meters when format is **`GRAY16_LE`**: **per camera**, the usable distance band for that stream; quantization maps that band linearly to uint16 **0..65534** (same code space for every camera, stretched to each device’s min–max), and optional backend-specific `backend_data`. |
| **Pipeline (string)** | A GStreamer pipeline description (e.g. `appsrc ! video/x-raw,... ! nvvidconv ! nvv4l2h264enc ! h264parse ! fakesink`) that you can run with `Gst.parse_launch()` or `gst-launch-1.0`. The interface **returns** the string; your code or GStreamer runs it. |
| **appsrc** | GStreamer element that accepts raw video buffers **pushed by the application** (e.g. from ZED SDK or Isaac Sim). The pipeline string uses `appsrc` as the source; you must feed frames in the format specified by the pipeline **caps**. |
| **Caps (capabilities)** | Format specification in a pipeline (e.g. `video/x-raw,format=RGB,width=640,height=480,framerate=30/1`). Your app must push buffers that match the caps of the `appsrc` you use. |
| **Modality** | What the sensor provides: `rgb` (color only), `depth` (depth only), `rgbd` (color + depth), or `radar` (radar visualization, often as a 2D intensity image). |
| **ZED 2i** | Stereolabs stereo depth camera used as the front robot camera. Supplies RGB and depth; the HAL uses the left-eye RGB and depth in the same way as the policy’s front camera. |
| **RGB-D** | Camera that provides both RGB and depth (e.g. ZED, RealSense). In this interface, “RGB-D” sensors can have one pipeline for the RGB stream (and optionally a second for depth if needed). |
| **nvenc / nvv4l2h264enc** | NVIDIA hardware H.264 encoder on Jetson. Uses the GPU; lower CPU and power than software encoding. Not available on all Jetson SKUs (e.g. some Orin Nano); the interface can fall back to **x264enc** (software). |
| **nvvidconv** | GStreamer element that converts video formats on the Jetson GPU (e.g. RGB → NV12 in GPU memory) for nvenc. |
| **Sensor pose** | Position and orientation of the sensor in the **robot base frame**: position (x, y, z) in meters and quaternion (qx, qy, qz, qw). Used for multi-camera fusion or visualization. |
| **Sink (fakesink, appsink, autovideosink)** | End of a pipeline: **fakesink** discards data (for testing or encoding-only); **appsink** delivers buffers to your app (e.g. for OpenCV display); **autovideosink** shows video in a window. You choose via `build_pipeline(..., output_element="...")`. |
| **SensorInterface** | Abstract class (in `hal.server.sensor_interface`) that both **JetsonSensorInterface** and **IsaacSensorInterface** implement. Code that only calls `list_sensors()`, `get_gstreamer_handle()`, and `build_pipeline()` works on either backend. |

**Why GStreamer?** The same pipeline description works on real hardware (with hardware encoding) and in simulation (with software encoding). Clients get a **string** they can run with GStreamer or adapt (e.g. replace `appsrc` with a different source for testing). No need to hand-write different paths for ZED vs USB vs synthetic cameras.

## API overview

- **`list_sensors()`**  
  Returns a list of `SensorInfo` for each available sensor: `id`, `type`, `pose`, `modality`, `resolution`, `fps`, `camera_driver`.

- **`get_gstreamer_handle(sensor)`**  
  Takes a **`SensorInfo`**. Always returns a **`GStreamerHandle`**: **`sensor_id`**, **`sensor_type`**, **`modality`**, **`resolution`**, **`fps`**, and **`camera_driver`** are copied from **`sensor`**. **`appsrc_pixel_format`**: on **Jetson**, taken from the catalog when **`sensor.id`** matches a catalog row, otherwise **`RGB`**; on **Isaac**, **`GRAY8`** if **`sensor.type`** is **`radar`**, **`GRAY16_LE`** if **`sensor.type`** / **`modality`** is **`depth`** (requires **`SensorInfo.extra["depth_range_m"]`**), else **`RGB`**. Synthetic depth entries (e.g. **`front_rgbd_gray16_depth`**) appear in **`list_sensors()`** when the catalog or Isaac scene wires them. There is **no** registry lookup or “unknown id” failure for non-depth rows—callers are responsible for **`SensorInfo`** matching what they will actually push into **`appsrc`**.

- **`build_pipeline(handle, encoding='h264', output_element=..., **kwargs)`**  
  Returns a GStreamer pipeline string that:
  - On **Jetson**: uses `appsrc` (you push frames from ZED SDK or other drivers), then `nvvidconv` and `nvv4l2h264enc` when `use_nvenc=True` (fallback: `x264enc`). If `output_element` is omitted (`None`), the tail is **decode + display** (`nvv4l2decoder ! nv3dsink` with nvenc, or CPU decode + `autovideosink` with software encode). Pass `output_element="fakesink"` for headless or “encode only” tests.
  - On **Isaac**: uses `appsrc` (you push rendered frames from the sim), then `videoconvert` and `x264enc` (default sink remains `fakesink` unless you override).

Pipeline strings can be used with `Gst.parse_launch()` or `gst-launch-1.0` (replace `appsrc` with a real source for standalone testing).

## Obtaining the interface

- **Jetson**: `JetsonHalServer.get_sensor_interface()` returns a `JetsonSensorInterface` built from `JETSON_SENSOR_CATALOG`. The **front observation** camera is the unique catalog row with **`is_primary=True`**: that row sets resolution, FPS, **`camera_driver`**, and **`depth_mode`** for `initialize_cameras()` (registered in `hal.server.jetson.front_camera_factory.FRONT_RGB_DEPTH_CAMERA_FACTORIES`). The matching **`SensorInfo`** in **`list_sensors()`** carries the same **`camera_driver`**.
- **Isaac**: `IsaacSimHalServer.get_sensor_interface()` passes `scene_sensors`; `IsaacSensorInterface` **introspects** `front_rgb` + `front_camera` and `side_rgb` + `side_camera` into **`front_rgbd`** / **`side_rgbd`** (plus matching **`*_gray16_depth`** preview rows) when those prims exist; otherwise **`list_sensors()` is empty**. Live HAL observations fill **`rgbd_by_catalog_id`** from those pairs only. Callers may pass a fixed **`configured_sensors=`** tuple when no scene exists (e.g. unit tests under `tests/unit/hal/`); that path may still include extra rows such as **`radar_front`** for GStreamer pipeline tests even though sim HAL does not attach radar frames today.

Both implement the same abstract `SensorInterface`: `list_sensors()`, `get_gstreamer_handle(sensor)`, `build_pipeline(handle, ...)`.

## ZED 2i on Jetson: default front camera example

The stack still treats the **Stereolabs ZED 2i** as the **reference** front RGB-D camera on Jetson. Other devices are supported only if they implement the same **`RgbDepthCamera`** contract and register a driver (see [Other front RGB-D drivers](#other-front-rgb-drivers) below).

**Catalog → front camera:** exactly one row in **`JETSON_SENSOR_CATALOG`** has **`is_primary=True`**. `JetsonHalServer` uses that row’s **`camera_driver`** with **`create_front_rgb_depth_camera`** (`hal/server/jetson/sensor_backend_jetson.py`, `front_camera_factory.py`). A **ZED** deployment registers a **`zed`** driver that maps to **`create_zed_camera`** in `hal/server/jetson/zed_camera.py` (`ZedCamera` wraps the ZED SDK); other drivers use the same hook pattern.

**Additional HAL RGB-D rows:** any other **`rgbd`** catalog row can set **`hal_open_rgbd=True`** to open a second (or further) **`RgbDepthCamera`**. Use **`policy_scan_slot="side"`** on at most one non-primary row only if the checkpoint uses a **second policy scan** (`num_side_scan`); that fills legacy **`HardwareObservations.side_*`**. **`side_camera_rgb` / `side_camera_depth`** use that row’s catalog **resolution** (they need not match the primary **`camera_height` / `camera_width`**). **Collision / proximity** (and any use separate from the locomotion scan) should read **`HardwareObservations.rgbd_by_catalog_id[catalog_id].depth`** for each opened stream—**full metric depth is always there**, independent of `policy_scan_slot`. **ZED rows:** `zed_usb_serial_env` is optional; set it (for example `KRABBY_SIDE_ZED_USB_SERIAL`) when deterministic camera selection is needed, otherwise HAL opens the first detected ZED. **MaixSense** rows: set explicit host+port (either literals `maixsense_host` + `maixsense_port` or env-name pair `maixsense_host_env` + `maixsense_port_env`) and pass matching **`-e`** values for env-based rows ([JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md#with-maixsense-a075v-http-in-container)).

### Installing the ZED SDK (Jetson host)

1. **Match L4T** to the installer: check `cat /etc/nv_tegra_release`, then download the matching **ZED SDK for Jetson** from the [Stereolabs release page](https://www.stereolabs.com/developers/release/) (e.g. L4T 36.4 → `ZED_SDK_Tegra_L4T36.4_*.zstd.run`).
2. **Install**: `sudo apt-get install -y zstd`, then run the `.zstd.run` installer (follow Stereolabs’ prompts or silent-install flags for automation).
3. **USB**: use a **USB 3.0** port; confirm the device with `lsusb | grep -i zed`.
4. **Diagnostics**: run `ZED_Diagnostic -c -d`.

   You should see **OK** for: ZED SDK Diagnostic, Processor, Graphics Card, and CUDA Operations. Under **AI Models Diagnostic**, detection models (MULTI CLASS, HUMAN BODY, PERSON HEAD, REID) may show “not optimized” — that is **normal** and not used by the HAL RGB/depth path. For depth, ensure **NEURAL LIGHT DEPTH**, **NEURAL DEPTH**, and **NEURAL PLUS DEPTH** show **optimized**; if not, run [pre-optimization](#pre-optimizing-zed-neural-depth-models) (or the full walkthrough in [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md#pre-optimizing-zed-neural-depth-models)).

**If the SDK cannot open the camera** (device appears in `lsusb` but open fails), install udev rules, replug the camera, and retry:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2b03", MODE="0666"' | sudo tee /etc/udev/rules.d/99-slabs.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Python (`pyzed`)** is installed with the ZED SDK. The HAL imports **`pyzed.sl`** inside `ZedCamera`; there is no separate pip package requirement beyond a correct SDK install on the machine or in the image.

### Docker, USB pass-through, and image builds

The ZED is accessed over **USB (libusb)**. **`--device /dev/video0` alone is not enough** for typical ZED SDK use inside Docker. Prefer **`-v /dev:/dev`** and **`--privileged`** when the container must open the camera (same pattern as production inference in [JETSON_DEPLOYMENT.md — With ZED 2i Camera](JETSON_DEPLOYMENT.md#with-zed-2i-camera)). The **multi-stream display** Jetson Docker snippet later in this doc should mount USB as needed so the ZED is visible in the container.

The **locomotion** image can bake in a specific ZED SDK version at build time; see [DOCKER_DEPENDENCIES.md — ZED SDK Installation](DOCKER_DEPENDENCIES.md#zed-sdk-installation) for version, download layout, silent install, and JetPack/L4T notes.

### Pre-optimizing ZED NEURAL depth models

On **first use**, the SDK may **download and optimize** NEURAL depth models (several minutes, GPU-specific). To avoid repeating that on every deploy, run **`ZED_Diagnostic -nrlo`** / **`ZED_Diagnostic -nrlo_plus`** once on the target Jetson and persist **`/usr/local/zed/resources`** with a host volume. Step-by-step commands and example `docker run` lines are in [JETSON_DEPLOYMENT.md — Pre-optimizing ZED NEURAL depth models](JETSON_DEPLOYMENT.md#pre-optimizing-zed-neural-depth-models).

### HAL usage with ZED

- **`JetsonHalServer`**: **resolution**, **FPS**, and **camera_driver** come from the explicit **`JETSON_SENSOR_CATALOG`** `is_primary` row.
- **`depth_mode`** is taken from each `JETSON_SENSOR_CATALOG` rgbd row (for ZED use values such as `NEURAL_LIGHT`, `NEURAL`, `NEURAL_PLUS`, `QUALITY`, `ULTRA`).
- After **`initialize_cameras()`**, observations fill **`HardwareObservations.camera_rgb`**, **`camera_depth`**, and policy **scan features** derived in **`JetsonHalServer`** from the same depth map (see `hal/server/jetson/depth_scan_features.py`). Verify shapes against **`camera_height` / `camera_width`** and your HAL client (see **HAL_GUIDE.md** and `hal/client/data_structures/hardware.py`).

### GStreamer example (primary RGB-D)

For **streaming / recording** on the primary RGB-D stream, take the matching **`SensorInfo`** from **`list_sensors()`** (conventionally logical id **`front_rgbd`** in the table below), call **`get_gstreamer_handle(sensor)`**, then **`build_pipeline(...)`** for an **`appsrc`‑based** pipeline. Your code must **push buffers** that match the handle’s caps (resolution, FPS, format). That is independent of whether you grabbed frames with the ZED SDK, another library, or a test pattern — the **interface** only defines the GStreamer side.

### Other front RGB-D drivers

To use a **non-ZED** camera that still matches the **`RgbDepthCamera`** protocol, add a factory under a new driver name in **`FRONT_RGB_DEPTH_CAMERA_FACTORIES`** (`hal/server/jetson/front_camera_factory.py`), set **`camera_driver`** on the **`is_primary`** catalog row to that name, and keep **`id="front_rgbd"`** (or change **`id`** consistently everywhere, including Isaac introspection, if you introduce a distinct logical slot). Adjust **`type`** in the catalog to match the GStreamer branch (**`rgbd`** shares caps with other RGB-D catalog entries).

## Sensor types and IDs

**Source of truth for which sensors appear:** each backend’s code—**Jetson:** `JETSON_SENSOR_CATALOG` in `hal/server/jetson/sensor_backend_jetson.py`; **Isaac:** scene introspection or **`configured_sensors`** in `hal/server/isaac/sensor_backend_isaac.py`. This document does not duplicate that list; it only names **conventional logical `id`s** so hardware, simulation, and clients can agree on labels.

| Sensor ID         | Type  | Modality | `camera_driver` (typical) | Typical use |
|-------------------|-------|----------|---------------------------|-------------|
| `front_rgbd`      | rgbd  | rgbd     | Jetson: `zed` or `maixsense_a075v`; Isaac: `isaac_scene` | Front RGB-D (policy + streaming) |
| `side_rgbd`       | rgbd  | rgbd     | Jetson: `maixsense_a075v` (repo default); Isaac: `isaac_scene` | Second HAL RGB-D row (`hal_open_rgbd=True`); optional `policy_scan_slot="side"` + `rgbd_by_catalog_id` |
| `front_rgbd_gray16_depth` | depth | depth | — (GStreamer preview of catalog rgbd depth) | Quantized depth stream for teleop/recording |
| `side_rgbd_gray16_depth`  | depth | depth | — | Same for side rgbd |
| `side_left_rgb`   | rgb   | rgb      | —                         | Optional legacy label (not in repo `JETSON_SENSOR_CATALOG`) |
| `side_right_rgb`  | rgb   | rgb      | —                         | Optional legacy label |
| `radar_front`     | radar | radar    | —                         | Isaac unit-test / pipeline only; no Jetson catalog row |

`camera_driver` is `None` where marked as — (no HAL in-process driver for that logical role). **Repo Jetson catalog** (`hal/server/jetson/sensor_backend_jetson.py`): **`front_rgbd`** (primary, `zed`) and **`side_rgbd`** (`maixsense_a075v`, `hal_open_rgbd=True`, `policy_scan_slot="side"`), each with an optional **`{id}_gray16_depth`** row when `gst_depth_quant_range_m` is set — **no** separate side RGB-only row and **no** radar rows. On Isaac, driver ids follow scene wiring or explicit config (see backend sources above). Add more **`rgbd`** rows with **`hal_open_rgbd=True`** for extra depth (collision, logging); for a second MaixSense HTTP target use catalog **`maixsense_host_env`** / **`maixsense_port_env`** ([**JETSON_DEPLOYMENT.md** — MaixSense-A075V](JETSON_DEPLOYMENT.md#maixsense-a075v-optional-host-bring-up)).

Poses are in the robot base frame (meters, quaternion x,y,z,w). Isaac sensor poses match the real hardware layout (see `zed_like_scene_cfg.py`).

## Pipeline generation options

- **`encoding`**: `"h264"`, `"h265"`, or `"raw"` (no encoder; useful for visualization).
- **`output_element`**: e.g. `"fakesink"` (discard), `"autovideosink"`, `"appsink"` (pull frames in Python), or a full tail like `nvv4l2decoder ! nv3dsink`. **Jetson**: omit (`None`) for a sensible on-robot **preview** default; set explicitly for headless or streaming.
- **Jetson only**: `use_nvenc=True` (default) uses `nvv4l2h264enc`; set `False` to use `x264enc` (e.g. on Orin Nano where nvenc may be unavailable). `bitrate` (default 4_000_000) can be passed in `kwargs`.

Pipelines use **appsrc** as the source. You must push buffers (e.g. from ZED SDK or Isaac render) in the format specified in the pipeline caps (e.g. `video/x-raw,format=RGB,width=640,height=480,framerate=30/1`).

## Multi-sensor bandwidth and power (Jetson)

- All sensors share USB/bus bandwidth. Running many high-resolution streams at once may require:
  - Lowering **resolution** or **fps** per sensor,
  - Spreading cameras across USB buses/hubs (see robot wiring),
  - Reducing power draw (e.g. lower fps or resolution) to stay within thermal/budget limits.
- Document your hub assignments and max simultaneous streams in deployment notes. The interface does not enforce limits; it only generates pipelines. Validation that the configured set can stream simultaneously is done via the multi-stream display tool (`hal.tools.multi_stream_display`) and manual testing.

## Isaac sensor setup

- **Front ZED-like**: Use a scene config that adds `front_camera` (depth) and `front_rgb` (RGB), e.g. `ZedLikeSceneCfg` from `hal.server.isaac.zed_like_scene_cfg`. Resolution is 640×480 in the current HAL server.
- **Side and radar**: Add matching synthetic cameras/sensors in the scene and extend `IsaacSensorInterface` (or scene config) so `list_sensors()` includes them with the same IDs as on Jetson. Pipeline generation remains the same; you feed frames from the sim into `appsrc`.

## Adding new sensor backends

1. Implement the abstract class `SensorInterface` in `hal.server.sensor_interface`:
   - `list_sensors()` → list of `SensorInfo`
   - `get_gstreamer_handle(sensor: SensorInfo)` → `GStreamerHandle` (map fields from `sensor`; no id validation)
   - `build_pipeline(handle, encoding=..., output_element=..., **kwargs)` → pipeline string
2. Use `SensorInfo` and `GStreamerHandle` from `hal.server.sensor_interface`; put backend-specific data in `SensorInfo.extra` or `GStreamerHandle.backend_data`.
3. Wire the backend into your HAL server (e.g. `get_sensor_interface()` returning your implementation) so clients get it from the same entry point.

## Multi-stream display (Docker)

Optional **CLI**: **`hal.tools.multi_stream_display`** (`hal/tools/multi_stream_display.py`). Not part of the HAL server API.

There is **no** wrapper script under `scripts/` for this tool: use **`python -m hal.tools.multi_stream_display`** from the image’s Python (as in the `docker run` examples below) or from an activated venv on the host that has `krabby-hal-tools` installed.

- **`--no-display`**: prints declared sensors and example `build_pipeline(...)` strings only.
- **Jetson + display**: uses the **configured front observation driver** from the catalog (default **`zed`**, i.e. ZED SDK / `pyzed`); shows live **RGB + depth** from that device. See [ZED 2i on Jetson: default front camera example](#zed-2i-on-jetson-default-front-camera-example) for install, Docker/USB, and NEURAL model notes. Which streams the tool actually opens is defined in **`hal/tools/multi_stream_display.py`** (the live path targets the front RGB-depth pair; it does not automatically fan out to every sensor **`list_sensors()`** might report).
- **Isaac + display**: starts **Isaac Lab** with **`ZedLikeSceneCfg`** (`front_rgb` + `front_camera` depth). Pass the same **`AppLauncher`** flags as other Isaac scripts (see `scripts/run_isaac_front_camera_capture.py`). Optional: **`--num_envs`** (default 1).

Assumes **Linux with X11** for display mode: allow Docker to use the host display and mount the X11 socket. If `DISPLAY` is unset, use `:0`.

```bash
export DISPLAY="${DISPLAY:-:0}"
xhost +local:docker 2>/dev/null || true
```

**Jetson** (locomotion image; build with `make build-locomotion-image`):

```bash
IMAGE="${KRABBY_LOCOMOTION_IMAGE:-krabby-locomotion:latest}"
docker run --rm --runtime=nvidia \
  -e "DISPLAY=${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/video0 \
  --entrypoint python3 \
  "$IMAGE" \
  -m hal.tools.multi_stream_display --backend jetson
```

Add **`--no-display`** to skip the OpenCV window. Mount or pass through **USB devices** as needed so the ZED is visible inside the container.

**Isaac Sim** image (build with `make build-isaacsim-image`):

```bash
IMAGE="${KRABBY_ISAACSIM_IMAGE:-krabby-isaacsim:latest}"
docker run --rm --gpus all \
  -e "DISPLAY=${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --entrypoint /workspace/testenv/bin/python \
  "$IMAGE" \
  -m hal.tools.multi_stream_display --backend isaac
```

Append Isaac Lab launcher flags as needed (e.g. headless off, device). Use **`--no-display`** if you only want stdout (no Isaac window, no OpenCV).

**Display** depends on **`opencv-python`** (e.g. `pip install 'krabby-hal-tools[display]'`). **Jetson** display also requires **ZED** runtime/SDK in the environment.

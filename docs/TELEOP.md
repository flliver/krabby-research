# Teleop (WebRTC remote viewing)

Jetson or Isaac HAL with **`--teleop-ip HOST`** runs an **outbound** WebSocket client to **`ws://HOST:9000/ws/robot`** (built by **`build_robot_signaling_ws_url`** in **`teleop/edge/robot_settings.py`**) and answers **SDP** with **HAL-backed** video. **`hal.server.teleop_portal_signaling`** runs inside the HAL process: a **`HalClient`** poll thread samples **`rgbd_by_catalog_id`** for WebRTC video, and (when commands are enabled) **`ControlLoop` (`INPUT_CONTROLLER_WEBRTC`)** maps browser gamepad state to joint commands. The reference **operator server** is **`krabby-teleop-portal`** (wheel root: **`teleop/portal/`**): **HTTP** UI, **`GET /api/teleop-config`**, FIFO relay **`/ws/browser`** ↔ **`/ws/robot`**. After **offer/answer**, media is **browser ↔ robot** (ICE/STUN/TURN as negotiated).

See also **[controller/cli/README.md](../controller/cli/README.md)** (local **`krabby-uno`** / **`krabby-uno-sim`** vs portal teleop).

**Two packages:** **`krabby-teleop-edge`** (wheel root: **`teleop/edge/`**) on robots only; **`krabby-teleop-portal`** (wheel root: **`teleop/portal/`**) on the operator host. Robot code calls **`teleop.edge.robot_settings.build_teleop_edge_settings()`** and **`portal_client_loop`** / **`run_robot_signaling_loop`**. The portal calls **`teleop.portal.settings.build_portal_auth_settings()`**, **`teleop.portal.ice_config.build_browser_ice_config()`**, and **`teleop.portal.relay.create_portal_app`**. Dev helpers: **`scripts/run_teleop_portal_x86_docker.sh`** (portal in Docker) and HAL **`--teleop-ip`** on Jetson or Isaac; unit tests under **`tests/unit/teleop/`**.

---

## Purpose and core functionality

**Goal:** remote operators view **live** video and sensor streams over **WebRTC**. The robot runs an agent that **connects outbound** to a **remote teleop server** for signaling and answers with **HAL-backed** video when **`--teleop-ip`** is set; **HAL** continues to expose **`get_observations()`** and the rest of the stack for autonomy and logging alongside teleop.

**What ships:**

1. **`krabby-teleop-portal`** — the **remote teleop server** reference app: operator **HTTP** page, **`GET /api/teleop-config`** (STUN/TURN / ICE helpers for the browser), **`/ws/browser`**, and **`/ws/robot`**. A small FIFO relay pairs one browser socket with one robot socket and forwards JSON text unchanged.
2. **`teleop.edge`** (used from Jetson / Isaac HAL **`--teleop-ip`**) — **outbound** WebSocket signaling and **aiortc** answers; **video tracks** read RGB from the teleop **`HalClient`** subscription via **`HalRgbSnapshotVideoTrack`**.
3. **WebRTC media** is **browser ↔ robot** after signaling completes (ICE/STUN/TURN as usual), unless you add a separate media relay.

---

## WebRTC stack decision (aiortc vs webrtcbin)

### Decision

Use **`aiortc`** for the live robot-side WebRTC path (`teleop.edge` session/signaling flow).

### Alternatives considered

- **`aiortc`** (selected)
- **GStreamer `webrtcbin`** (not selected for current implementation)

### Rationale

- **Python-first integration:** existing signaling/session logic is already implemented in Python (`teleop/edge/portal_client.py`, `teleop/edge/signaling_session.py`, `teleop/edge/rtc_session.py`).
- **Lower implementation complexity:** avoids coupling session state management to a separate GStreamer WebRTC graph lifecycle.
- **Testing fit:** current unit tests in `tests/unit/teleop/` validate signaling/session behavior directly around Python boundaries.
- **Requirement fit:** current scope is reliable remote viewing from HAL-backed RGB streams, which is satisfied by the `aiortc` path.

### Tradeoffs and revisit triggers

- **Tradeoff:** `webrtcbin` can be preferable for deeper end-to-end GStreamer-native media control.
- Revisit this choice if we need:
  - tighter hardware-encoding control directly in live teleop media,
  - materially higher stream-count scaling than current targets,
  - or measured latency/performance goals that `aiortc` cannot meet within acceptable complexity.

---

## How components connect

### Responsibility split

| Piece | Role |
|--------|------|
| **`JetsonHalServer`** / **`IsaacSimHalServer`** | Cameras / sim sensors, **`get_observations()`**, publishes observations on the HAL **PUB** socket |
| **`hal.server.teleop_portal_signaling`** | Outbound **`teleop.edge`** signaling, **`HalRgbSnapshotVideoTrack`**, and (when **`send_hal_commands`**) browser control: **`WebRTCInputController`** + **`ControlLoop` (`INPUT_CONTROLLER_WEBRTC`)** → **`GamepadToKrabbyHALMapper`** → **`HalClient.put_joint_command`**. Uses a dedicated **`HalClient`** poll thread for latest RGB (catalog ids from the portal **`catalog_ids`** offer field, bootstrapped from the primary HAL catalog id until set). Command and observation clients share the HAL **`transport_context`** (inproc on Jetson portal/inference; TCP on Isaac when **`--joystick`** is used). |
| **`controller.control_loop.ControlLoop`** | Wired only for **browser teleop** inside **`teleop_portal_signaling`** (`INPUT_CONTROLLER_WEBRTC`). Local gamepad clients use **`krabby-uno`** / **`krabby-uno-sim`** in a separate process (`INPUT_CONTROLLER_KRABBY` / `INPUT_CONTROLLER_ISAACSIM`). |
| **`krabby-teleop-portal`** | Remote server: UI + config + relay **`/ws/browser`** ↔ **`/ws/robot`** |
| **Browser** (`teleop_session.js` / portal viewer) | Loads UI from the **portal** origin, **`/ws/browser`**, WebRTC **offer** / **answer**, re-offer for stream count |

### Topology (only supported path)

```text
  Operator browser ── HTTP + WSS /ws/browser (outbound) ──► Remote teleop server (portal)
  Robot agent      ── WSS …/ws/robot (outbound) ───────────► Remote teleop server (portal)
  Operator browser ◄──── WebRTC media ───────────────────► Robot (ICE; may use TURN)
```

Pass **`--teleop-ip HOST`** on Jetson or Isaac HAL entry points. The HAL builds **`ws://HOST:9000/ws/robot`** (or pass a full **`ws://`** / **`wss://`** URL). Omit **`--teleop-ip`** to disable teleop. Jetson **`--control-source portal`** requires **`--teleop-ip`**. ICE, QoS, auth token, and stream caps still come from **`teleop.edge.robot_settings`** via **`build_teleop_edge_settings(host_or_url=…)`**.

There is **no `--controller webrtc` flag** on **`krabby-uno`** or **`krabby-uno-sim`**. Browser control is selected by HAL flags and the portal UI, not by a separate controller CLI.

### Browser control vs local gamepad (`ControlLoop`)

| Goal | Jetson | Isaac Sim | `ControlLoop` mode |
|------|--------|-----------|-------------------|
| **Browser gamepad → robot** | `krabby-hal-server-jetson --control-source portal --teleop-ip <portal-host>` | Isaac HAL with **`--teleop-ip <portal-host>`**; enable **operator_override** in the portal to drive | **`INPUT_CONTROLLER_WEBRTC`** (in-process in **`teleop_portal_signaling`**) |
| **Local Pro Controller → robot** | **`krabby-uno`** with server **`--control-source gamepad`** | **`krabby-uno-sim --quad\|--hex`** with server **`--joystick`** | **`INPUT_CONTROLLER_KRABBY`** / **`INPUT_CONTROLLER_ISAACSIM`** (separate TCP client) |
| **Portal viewer / signaling** | **`krabby-teleop-portal --host 0.0.0.0 --port 9000`** on the operator host | same | (not `ControlLoop`; HTTP + WebSocket relay only) |

On Isaac with both **`--teleop-ip`** and **`--joystick`**, the portal **operator_override** checkbox gates whether browser frames reach HAL (**`ControlLoopConfig.command_send_gate`**). **`--teleop-ip`** enables WebRTC capability; override enables driving from the browser when other command sources are active.

```text
  Browser krabby-control-v1 @ 50 Hz
      → WebRTCInputController.update_from_payload
      → ControlLoop (INPUT_CONTROLLER_WEBRTC)
      → GamepadToKrabbyHALMapper
      → HalClient.put_joint_command → HAL server
```

Parallel in the same teleop thread: **`HalClient.poll`** → RGB frames → **`HalRgbSnapshotVideoTrack`** → WebRTC video.

### Typical flow

1. Operator opens the **portal** HTTP origin.
2. **`GET /api/teleop-config`** → **`iceServers`** for the browser.
3. Browser **`WebSocket`** to **`/ws/browser`**; robot **`WebSocket`** to **`/ws/robot`** (outbound).
4. **`offer`** / **`answer`** (non-trickle SDP) over the relayed JSON path.
5. **Re-offer** on the same robot socket replaces the previous **`RTCPeerConnection`**.

### Viewer: which HAL cameras (catalog ids)

The portal page (**`teleop_session.js`**) can send optional **`catalog_ids`** on **`hello`** and on each **`offer`**: a JSON array of strings (HAL **`rgbd_by_catalog_id`** keys), in the same order as the browser’s recvonly video lines. If omitted, the robot keeps polling its **bootstrap** list (Jetson **`main.py`** seeds that to the **primary** catalog id only). Send **`"catalog_ids": []`** to revert to that bootstrap after a prior selection. The list is capped by **`MAX_VIDEO_M_LINES`** in **`robot_settings.py`**.

### Cockpit motion HUD (robot → browser)

When HAL fills **`base_quat_w`**, **`base_ang_vel_b`**, and **`base_lin_vel_b`** (Jetson: primary ZED IMU + tracking; Isaac Sim: **`isaac_primary_rgbd_base_state`** using the same **`front_rgbd`** mount transforms), **`hal.server.teleop_portal_signaling`** publishes JSON on a robot-originated WebRTC data channel **`krabby-telemetry-v1`** (~20 Hz). The portal viewer overlays a **cockpit HUD** on the video stage: ground speed, a **compass rose** (heading from yaw), a **3D attitude** cube (quaternion), a 2D artificial horizon, roll/pitch/yaw readout, and body-frame linear/angular velocity. Payload shape is built in **`teleop.edge.telemetry.build_telemetry_payload`**. Browser → robot gamepad input remains on **`krabby-control-v1`**.

---

## Configuration

### Robot (Jetson / Isaac HAL **`--teleop-ip`**)

**Portal host (CLI):** **`--teleop-ip HOST`** on **`hal.server.jetson.main`** or **`hal.server.isaac.main`**. Examples:

```bash
# Jetson portal control
python -m hal.server.jetson.main --control-source portal --teleop-ip 10.0.0.130

# Isaac sim with local portal
./scripts/run_isaac_hal_server.sh --teleop-ip 127.0.0.1
```

**`ControlLoop` WEBRTC wiring** (implemented in **`hal/server/teleop_portal_signaling.py`** when **`send_hal_commands=True`**):

| `ControlLoopConfig` field | Teleop usage |
|---------------------------|--------------|
| **`mode=INPUT_CONTROLLER_WEBRTC`** | Browser path only; uses **`GamepadToKrabbyHALMapper`** (same leg/joint mapping as local **`krabby-uno`**, default mapper scales). |
| **`webrtc_input_controller`** | Shared **`WebRTCInputController`**; signaling calls **`update_from_payload`** on each control frame. |
| **`hal_client_config`** | Same observation/command endpoints as the teleop poll client. |
| **`hal_transport_context`** | Shared ZMQ context with the HAL server (required for inproc). |
| **`krabby_gamepad_robot_definition`** | Must match HAL **`--robot`** topology. |
| **`command_send_gate`** | Portal **`operator_override`** on Isaac; Jetson portal mode sends when override is on in the control JSON. |
| **`input_controller_update_rate_hz`** | **50** (matches browser **`teleop_session.js`** interval). |

**Module settings** in **`teleop.edge.robot_settings`** (checked into the repo; override per deployment or image layer):

| Constant | Role |
|----------|------|
| **`SERVER_SIGNALING_WS_URL`** | Default URL when **`build_teleop_edge_settings()`** is called without **`host_or_url`** (tests, custom entry points). HAL **`--teleop-ip`** overrides this. |
| **`TELEOP_EDGE_MODE`** | **`"off"`** or **`"agent"`** for module-only builds. HAL **`--teleop-ip`** forces **`agent`**. |
| **`SERVER_RECONNECT_S`** | Reconnect backoff after dial-out errors. |
| **`MAX_VIDEO_M_LINES`** | Cap on recvonly video **`m=`** lines per offer (clamped 1–32). |
| **`QOS_ENABLED`** | When true, robot adapts teleop video under bandwidth pressure (see **QoS and degradation**). |
| **`QOS_KBPS_BUDGET_PER_STREAM`** | Nominal per-stream bitrate budget (kbps) for the degradation ladder (default **120**, tuned for aiortc snapshot teleop ~100 kbps/stream; raise for high-bitrate GStreamer tails). |
| **`STUN_TURN_SERVERS`** | ICE list for the **robot’s** WebRTC answers. Keep aligned with **`teleop.portal.ice_config.STUN_TURN_SERVERS`** on the portal host so browser **`GET /api/teleop-config`** and robot use the same bootstrap. If empty or invalid, **`build_teleop_edge_settings`** uses **`BUILTIN_STUN_SERVERS`**. |
| **`HTTP_AUTH_TOKEN`** | If non-empty, appended as **`?token=`** on the robot’s outbound signaling WebSocket (must match **`teleop.portal.settings.HTTP_TOKEN`** when the portal requires auth). |

### Remote server (`krabby-teleop-portal`)

Optional HTTP auth: **`teleop.portal.settings`** (**`HTTP_TOKEN`**). Browser ICE defaults: **`teleop.portal.ice_config`**.

| Service | Default bind | Routes |
|---------|----------------|--------|
| Portal | **`0.0.0.0:9000`** in Docker examples | **`/`**, **`/api/teleop-config`**, **`/static/`**, **`/ws/browser`**, **`/ws/robot`** |

Terminate **TLS** in front of the portal in production; preserve **WebSocket Upgrade**; keep **`/api/teleop-config`** on the **same origin** as the UI.

---

## Critical low-level details

### Signaling (v1, JSON over WebSocket)

- **Robot path:** outbound client to **`…/ws/robot`**; JSON messages: **`hello`**, **`ping`**, **`offer`**, **`answer`**, **`error`**.
- **Non-trickle:** gather ICE to **complete** before sending the **offer** (bundled JS listens for **`icegatheringstatechange`**).
- **Multiple video lines:** **N** recvonly video transceivers → **N** sender tracks if within **`robot_settings.MAX_VIDEO_M_LINES`**.
- **Congestion / QoS:** robot-side degradation controller in **`teleop.edge.qos`** (see **QoS and degradation**).

### Data channels

**Browser → robot (`krabby-control-v1`)**

- Browser creates a WebRTC data channel named **`krabby-control-v1`**.
- Robot accepts that channel and consumes JSON control messages:
  - `{"type":"control","sent_browser_ms":<number>,"state":{...}}`
  - `state` mirrors `ControllerState` keys:
    - buttons: `LT`, `LB`, `LS`, `RS`, `RT`, `RB` (booleans)
    - axes: `LX`, `LY`, `RX`, `RY` (normalized floats in `[-1, 1]`)
- Browser sends control at **50 Hz** (`20ms` interval) from either a Gamepad API joystick or the on-page virtual joystick/buttons.
- Robot path:
  - data channel JSON -> `WebRTCInputController.update_from_payload` -> **`ControlLoop` (`INPUT_CONTROLLER_WEBRTC`)** -> `GamepadToKrabbyHALMapper` -> `HalClient.put_joint_command` (gated by portal **`operator_override`** via `ControlLoopConfig.command_send_gate` on Isaac; always on when driving in Jetson portal mode).
- Robot logs receiver-side control latency from `sent_browser_ms` every ~5s with **p50**, **p95**, **max**, and latest samples. This is a browser wall-clock to robot wall-clock delta, so keep clocks synced for meaningful one-way latency.
- Invalid/non-JSON control payloads are rejected with warning logs; malformed fields are rejected by parser without tearing down media.

**Robot → browser (`krabby-telemetry-v1`)**

- Robot creates an outbound data channel **`krabby-telemetry-v1`** (~20 Hz JSON).
- Payload from **`teleop.edge.telemetry.build_telemetry_payload`** (`base_quat_w`, `base_ang_vel_b`, `base_lin_vel_b` on **`HardwareObservations`**).
- Portal viewer cockpit HUD: speed, compass, 3D attitude, horizon, body-frame velocity (see **Cockpit motion HUD** above).

### HAL vs WebRTC

Inference uses **`get_observations()`**. Viewer depth previews (if shown) are for humans only; raw depth for models stays on HAL. Encoding helpers live in **`hal/server/gstreamer_runtime.py`**.

### GStreamer sensor interface vs live media

**`SensorInterface`** (`list_sensors`, `get_gstreamer_handle`, `build_pipeline`) is the shared Jetson/Isaac API for sensor discovery and GStreamer pipeline strings (encode, tooling, tests). **Live teleop** uses that API for **`available_catalog_ids`**, offer validation, and offer-time **`build_pipeline(..., fakesink)`** preflight per selected sensor.

**Live video** is **`HalClient`** → **`HardwareObservations.rgbd_by_catalog_id`** → **`HalRgbSnapshotVideoTrack`** → **aiortc** (same path on Jetson and Isaac). Policy, the data collector, and teleop all subscribe to that observation bus: one capture per sensor, latest-only frames, time-aligned RGB/depth/telemetry, and a single WebRTC encode at the network edge. Encoded GStreamer tails from **`build_pipeline`** are used by **`hal.server.streaming_map`** and **`hal/tools/multi_stream_display`**.

### Pipelines and codecs

**Live teleop** encodes on the robot with **`aiortc`** from **`HalRgbSnapshotVideoTrack`** frames; the browser negotiates **VP8** or **H.264** on the peer connection.

### Latency

**Targets (p95 over 60 s steady state):**

| Streams | Glass-to-glass budget |
|---------|------------------------|
| 1 | **&lt; 300 ms** |
| 4 | **&lt; 500 ms** |

Constants: **`teleop.edge.qos.G2G_TARGET_MS_SINGLE_STREAM`**, **`G2G_TARGET_MS_FOUR_STREAMS`**.

The UI reports two different latency signals:

- **RTT** comes from WebSocket **ping** / **pong** and measures signaling round-trip time, not media glass-to-glass latency.
- **g2g** is an estimated capture-to-render value for the first selected stream when robot capture timestamps are available. It uses ping/pong wall-clock offset estimation, so treat it as an approximation and keep browser/robot clocks synced.

#### Glass-to-glass measurement method

1. **Clock sync:** run NTP (or chrony) on the robot and operator host. Offset estimation in the portal adds tens of milliseconds of error when clocks drift.
2. **Setup:** open the portal viewer with **1** or **4** recvonly video lines (catalog ids in offer order). Wait until ICE is connected and video is rendering (~10 s warmup).
3. **Sample:** every **2 s** the portal sends **ping**; **pong** includes **`capture_timestamps_ns`** per active catalog id and the HUD updates **g2g~** ms for the first selected stream.
4. **Record:** collect **30** g2g samples (~60 s). Compute **p50** and **p95** (helpers in **`teleop.edge.latency.summarize_g2g_samples`**).
5. **Pass criteria:** p95 ≤ target for the stream count under test.

**Formula** (implemented in **`teleop.edge.latency.estimate_g2g_ms`**):

- `offset_ms` ≈ robot wall-clock minus browser wall-clock (EMA-smoothed from ping/pong **`t_wall_ms`** / **`server_ms`**).
- `g2g_ms` = `browser_now_ms` − (`capture_timestamp_ns` / 1e6 − `offset_ms`).

RTT and g2g measure different paths; use g2g for the latency requirement above.

### QoS and degradation

When **`QOS_ENABLED`** is true, **`TeleopQosController`** polls WebRTC **`getStats()`** on the robot (~1 Hz), derives outbound video kbps and packet-loss fraction, and drives **`HalRgbSnapshotVideoTrack`**:

1. **Lower target fps** (30 → 24 → 15 → 10 → 5) while all streams remain active.
2. **Drop lowest-priority streams** (highest track index / last catalog id in the offer) down to **one** active stream; inactive tracks emit black at **1 fps** to keep the peer connection alive.

**Budget model:** total nominal budget = `stream_count × QOS_KBPS_BUDGET_PER_STREAM` (default **120 kbps** per stream for **`HalRgbSnapshotVideoTrack`** + aiortc). Each stats sample compares measured outbound to the budget for the **current** degradation level (active streams × target fps), so a single surviving stream is not judged against the full multi-camera budget. After offer renegotiation, escalation is suppressed for **10 s** while encoders ramp. Degradation steps trigger on outbound bitrate falling below budget fractions and/or rising packet loss; recovery uses **2** consecutive healthy samples (hysteresis).

Robot logs **`teleop qos:`** every level change and ~5 s while degraded. **pong** includes **`qos`** snapshot (`level`, `target_fps`, `active_stream_count`, `outbound_kbps`, `packet_loss_fraction`).

**Characterization tests:** **`tests/unit/teleop/test_qos.py`** documents the degradation ladder (budget and loss triggers, fps, active stream count). Run:

```bash
pytest tests/unit/teleop/test_qos.py -v
```

### Troubleshooting

| Symptom | Check |
|---------|--------|
| **`--teleop-ip` but no video** | Portal running at **`HOST:9000`**; FIFO pair (one browser, one robot); HAL cameras initialized. |
| **403** on portal | **`teleop.portal.settings.HTTP_TOKEN`** on the portal host and same value in **`robot_settings.HTTP_AUTH_TOKEN`** (query **`?token=`** on robot WS URL). |
| **ICE failed** | **TURN** entries in **`teleop.portal.ice_config`** (browser) and matching **`robot_settings.STUN_TURN_SERVERS`** (robot); verify in browser devtools. |
| **too many recvonly video m-lines** | Lower stream count or raise **`robot_settings.MAX_VIDEO_M_LINES`**. |
| **Video works but robot does not move (Isaac)** | Enable **operator_override** in the portal when **`--joystick`** or inference is also active; check HAL logs for **`teleop control latency`**. |
| **Video works but robot does not move (Jetson portal)** | Use **`--control-source portal --teleop-ip`**; connect WebRTC; drive with portal gamepad/virtual controls and override on. |

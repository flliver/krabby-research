# Teleop (WebRTC remote viewing)

Jetson HAL with **`--teleop`** runs an **outbound** WebSocket client to the URL in **`teleop.edge.robot_settings`** (source: **`teleop/edge/robot_settings.py`**) and answers **SDP** with **HAL-backed** video. A **second `HalClient`** (same ZMQ endpoints as inference) subscribes to **`HardwareObservations`** and samples **`rgbd_by_catalog_id`** for each viewer line. The reference **operator server** is **`krabby-teleop-portal`** (wheel root: **`teleop/portal/`**): **HTTP** UI, **`GET /api/teleop-config`**, FIFO relay **`/ws/browser`** ↔ **`/ws/robot`**. After **offer/answer**, media is **browser ↔ robot** (ICE/STUN/TURN as negotiated).

**Two packages:** **`krabby-teleop-edge`** (wheel root: **`teleop/edge/`**) on robots only; **`krabby-teleop-portal`** (wheel root: **`teleop/portal/`**) on the operator host. Robot code calls **`teleop.edge.robot_settings.build_teleop_edge_settings()`** and **`portal_client_loop`** / **`run_robot_signaling_loop`**. The portal calls **`teleop.portal.settings.build_portal_auth_settings()`**, **`teleop.portal.ice_config.build_browser_ice_config()`**, and **`teleop.portal.relay.create_portal_app`**. Optional dev-only script: **`scripts/teleop_smoke.py`** — **`signaling`** (dial-out WebSocket to portal ``/ws/robot``, no HAL) or **`http`** (minimal **`/`** / **`/api/teleop-config`** HTTP listener).

---

## Purpose and core functionality

**Goal:** remote operators view **live** video and sensor streams over **WebRTC**. The robot runs an agent that **connects outbound** to a **remote teleop server** for signaling and answers with **HAL-backed** video where **`--teleop`** is enabled; **HAL** continues to expose **`get_observations()`** and the rest of the stack for autonomy and logging alongside teleop.

**What ships:**

1. **`krabby-teleop-portal`** — the **remote teleop server** reference app: operator **HTTP** page, **`GET /api/teleop-config`** (STUN/TURN / ICE helpers for the browser), **`/ws/browser`**, and **`/ws/robot`**. A small FIFO relay pairs one browser socket with one robot socket and forwards JSON text unchanged.
2. **`teleop.edge`** (used from Jetson HAL **`--teleop`**) — **outbound** WebSocket signaling and **aiortc** answers; **video tracks** read RGB from the teleop **`HalClient`** subscription via **`HalRgbSnapshotVideoTrack`**.
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
| **`hal.server.teleop_portal_signaling`** | Dedicated **`HalClient`** poll thread (latest RGB for catalog ids chosen by the **portal viewer** over signaling, bootstrapped from the primary HAL catalog id until **`catalog_ids`** is sent) plus **`teleop.edge`** outbound signaling and **`HalRgbSnapshotVideoTrack`** (shared by Jetson and Isaac; not Jetson-specific) |
| **`krabby-teleop-portal`** | Remote server: UI + config + relay **`/ws/browser`** ↔ **`/ws/robot`** |
| **Browser** (`teleop_session.js` / portal viewer) | Loads UI from the **portal** origin, **`/ws/browser`**, WebRTC **offer** / **answer**, re-offer for stream count |

### Topology (only supported path)

```text
  Operator browser ── HTTP + WSS /ws/browser (outbound) ──► Remote teleop server (portal)
  Robot agent      ── WSS …/ws/robot (outbound) ───────────► Remote teleop server (portal)
  Operator browser ◄──── WebRTC media ───────────────────► Robot (ICE; may use TURN)
```

Configure the robot by editing **`teleop.edge.robot_settings`**: set **`TELEOP_EDGE_MODE`** to **`"agent"`** and **`SERVER_SIGNALING_WS_URL`** to your portal’s **`/ws/robot`** URL. With Jetson **`--teleop`**, signaling starts when a freshly built **`TeleopEdgeSettings`** has **`agent_enabled`** true (agent mode plus non-empty URL).

### Typical flow

1. Operator opens the **portal** HTTP origin.
2. **`GET /api/teleop-config`** → **`iceServers`** for the browser.
3. Browser **`WebSocket`** to **`/ws/browser`**; robot **`WebSocket`** to **`/ws/robot`** (outbound).
4. **`offer`** / **`answer`** (non-trickle SDP) over the relayed JSON path.
5. **Re-offer** on the same robot socket replaces the previous **`RTCPeerConnection`**.

### Viewer: which HAL cameras (catalog ids)

The portal page (**`teleop_session.js`**) can send optional **`catalog_ids`** on **`hello`** and on each **`offer`**: a JSON array of strings (HAL **`rgbd_by_catalog_id`** keys), in the same order as the browser’s recvonly video lines. If omitted, the robot keeps polling its **bootstrap** list (Jetson **`main.py`** seeds that to the **primary** catalog id only). Send **`"catalog_ids": []`** to revert to that bootstrap after a prior selection. The list is capped by **`MAX_VIDEO_M_LINES`** in **`robot_settings.py`**.

---

## Configuration

### Robot (Jetson HAL **`--teleop`**)

Edit **`teleop.edge.robot_settings`** (checked into the repo; override values per deployment or image layer):

| Constant | Role |
|----------|------|
| **`SERVER_SIGNALING_WS_URL`** | WebSocket on the teleop server (e.g. **`wss://host/ws/robot`**). Required for agent mode. |
| **`TELEOP_EDGE_MODE`** | **`"off"`** or **`"agent"`**. **`"agent"`** without a non-empty URL is treated as **`off`**. |
| **`SERVER_RECONNECT_S`** | Reconnect backoff after dial-out errors. |
| **`MAX_VIDEO_M_LINES`** | Cap on recvonly video **`m=`** lines per offer (clamped 1–32). |
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
- **Congestion:** standard WebRTC; no custom algorithm in-repo.

### Control data channel (v1)

- Browser creates a WebRTC data channel named **`krabby-control-v1`**.
- Robot accepts that channel and consumes JSON control messages:
  - `{"type":"control","sent_browser_ms":<number>,"state":{...}}`
  - `state` mirrors `ControllerState` keys:
    - buttons: `LT`, `LB`, `LS`, `RS`, `RT`, `RB` (booleans)
    - axes: `LX`, `LY`, `RX`, `RY` (normalized floats in `[-1, 1]`)
- Browser sends control at **50 Hz** (`20ms` interval) from either a Gamepad API joystick or the on-page virtual joystick/buttons.
- Robot path:
  - data channel JSON -> `WebRTCInputController` -> `GamepadToKrabbyHALMapper` -> `HalClient.put_joint_command`.
- Robot logs receiver-side control latency from `sent_browser_ms` every ~5s with **p50**, **p95**, **max**, and latest samples. This is a browser wall-clock to robot wall-clock delta, so keep clocks synced for meaningful one-way latency.
- Invalid/non-JSON control payloads are rejected with warning logs; malformed fields are rejected by parser without tearing down media.

### HAL vs WebRTC

Inference uses **`get_observations()`**. Viewer depth previews (if shown) are for humans only; raw depth for models stays on HAL. Encoding helpers live in **`hal/server/gstreamer_runtime.py`**.

### Pipelines and codecs

**`hal.server.streaming_map`** maps sensors to **`build_pipeline`** for encoded tails (e.g. files, **`fakesink`**, **`appsink`**). That path is separate from live teleop.

**Live teleop** uses **`aiortc`**; the browser negotiates **VP8** or **H.264** on the peer connection. HAL **`build_pipeline(..., encoding='h264'|'h265')`** targets recorded / headless encode checks, not the **`teleop_edge`** WebRTC session.

### Latency

The UI reports two different latency signals:

- **RTT** comes from WebSocket **ping** / **pong** and measures signaling round-trip time, not media glass-to-glass latency.
- **g2g** is an estimated capture-to-render value for the first selected stream when robot capture timestamps are available. It uses ping/pong wall-clock offset estimation, so treat it as an approximation and keep browser/robot clocks synced.

### Troubleshooting

| Symptom | Check |
|---------|--------|
| **`--teleop` but no video** | **`robot_settings.SERVER_SIGNALING_WS_URL`** and agent mode; portal running; FIFO pair (one browser, one robot). |
| **403** on portal | **`teleop.portal.settings.HTTP_TOKEN`** on the portal host and same value in **`robot_settings.HTTP_AUTH_TOKEN`** (query **`?token=`** on robot WS URL). |
| **ICE failed** | **TURN** entries in **`teleop.portal.ice_config`** (browser) and matching **`robot_settings.STUN_TURN_SERVERS`** (robot); verify in browser devtools. |
| **too many recvonly video m-lines** | Lower stream count or raise **`robot_settings.MAX_VIDEO_M_LINES`**. |

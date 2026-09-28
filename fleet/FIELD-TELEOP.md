# Field teleop (enrolled robots)

## Requirement

Any **enrolled, powered** Krabby must accept **Open teleop** from the fleet portal
without an operator SSH session or ad-hoc Docker commands.

## End-to-end path

```text
Operator browser → fleet portal / service (Cognito)
  → MQTT teleop/{thingName}/signaling/in|out
  → krabby agent (ws://127.0.0.1:9000/ws/robot)
  → HAL teleop edge (--teleop-ip 127.0.0.1)
  → WebRTC media (ICE / coturn)
```

Cloud signaling is shared across the fleet ([`SETUP-FLEET.md`](SETUP-FLEET.md)). Each
robot uses one MQTT client (`krabby agent`) and per-thing IoT policy.

## Robot processes

| Process | systemd | Role |
|---------|---------|------|
| **`krabby agent`** | `krabby-agent.service` | Shadow, Secure Tunnel notify, teleop MQTT ↔ local WebSocket on **:9000** |
| **Locomotion HAL** | `krabby-locomotion.service` | `krabby run` → Docker **`krabby`** → `hal.server.jetson.main` with fleet teleop flags |

Both units are **enabled on enroll**, **`Restart=always`**. Locomotion starts
**after** `docker.service` and **`krabby-agent.service`**.

Teleop **video and WebRTC** require HAL running with **`--teleop-ip 127.0.0.1`**. The
agent alone only carries **signaling** JSON.

## Kit config: `/etc/krabby/locomotion.json`

Written at **`krabby enroll`** (optional overrides on the enroll command). Read by
**`krabby run`** when **`/etc/krabby/iot/`** exists.

| Field | Meaning |
|-------|---------|
| `control_source` | `portal` (browser commands) or `inference` (policy + optional remote view) |
| `robot` | `hex` or `go2` |
| `teleop_ip` | Always **`127.0.0.1`** for fleet (dial agent shim, not a LAN portal) |
| `checkpoint` | Container path; required when `control_source` is `inference` |
| `checkpoint_host_dir` | Host dir mounted at `/workspace/checkpoints` |
| `zed_resources_host` / `zed_settings_host` | Host paths for ZED cache (defaults under `~/zed-resources/`) |

Example enroll:

```bash
sudo -E env PATH="$PATH" krabby enroll --thing-name <thing-name> \
  --locomotion-control-source portal \
  --locomotion-robot hex
```

Inference kit:

```bash
sudo -E env PATH="$PATH" krabby enroll --thing-name <thing-name> \
  --locomotion-control-source inference \
  --locomotion-robot go2 \
  --locomotion-checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
  --locomotion-checkpoint-host-dir /path/on/host/checkpoints
```

Step-by-step identity: [`ENROLL.md`](ENROLL.md).

## `krabby run` behavior

| Host | Command | Container behavior |
|------|---------|-------------------|
| **Enrolled** (`/etc/krabby/iot/`) | `krabby run` | HAL argv from `locomotion.json` + **`--teleop-ip 127.0.0.1`** (no gamepad client) |
| **Enrolled** | `krabby run --gamepad-only` | Gamepad stack (local dev only; not fleet teleop) |
| **Not enrolled** | `krabby run` | Gamepad stack |
| **Not enrolled** | `krabby run -- --checkpoint …` | Inference via image entrypoint |

**`krabby install`** sets up the host and enables **`krabby-locomotion.service`**
with **`ExecStart=… krabby run`**. On enrolled hosts, that single **`ExecStart`**
resolves to fleet HAL at runtime (no separate unit template).

Do not run a second HAL container alongside **`krabby`**.

## Cold start

If signaling arrives at the agent before HAL connects to **:9000**, the teleop shim
rate-limits **`systemctl start krabby-locomotion.service`**. Primary mode is still
locomotion **up on boot** after enroll.

## Shadow

**`teleop_edge_connected`** in `state.reported` is **true** when HAL’s WebSocket is
attached to the agent shim (see **`krabby get telemetry`**).

## Gamepad vs fleet

**`krabby install`** on a machine **without** enroll may autostart gamepad **`krabby run`**
for local bring-up. **After enroll**, default **`krabby run`** is fleet teleop, not
gamepad.

## Out of scope

- **`--teleop-control-echo`** — automated tests only ([`BENCH-TELEOP.md`](BENCH-TELEOP.md)).
- **`--teleop-ip <lan-host>`** without the agent — legacy LAN portal dev ([`docs/TELEOP.md`](../docs/TELEOP.md)).
- A second MQTT client on the robot.

## Verify (reference Orin)

After enroll and reboot, without SSH:

1. **`systemctl is-active krabby-agent krabby-locomotion`**
2. **`ss -tlnp | grep 9000`** — agent shim
3. Portal **Open teleop** → viewer reaches **Playing** (cameras / ZED permitting)

Bench CI: [`BENCH-TELEOP.md`](BENCH-TELEOP.md) (same locomotion path; add
**`--teleop-control-echo`** for pytest via `krabby run -- --teleop-control-echo`).

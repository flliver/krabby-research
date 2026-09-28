# Fleet setup and operations

Operator guide for the Krabby fleet: AWS IoT control plane, device
onboarding, Secure Tunneling SSH, telemetry, and the fleet service stack.

Infra deploy scripts: [`infra/`](infra/README.md). Device CLI: [`krabby/README.md`](../krabby/README.md).
Index: [`README.md`](README.md).

## Deploy sequence (fresh account)

1. Deploy the control plane (IoT thing type, per-thing policy, Fleet Indexing,
   reported-images bucket, `krabby-enroll` IAM user):

   ```bash
   cd fleet/infra
   ./scripts/setup-venv.sh
   source .venv/bin/activate
   ./scripts/deploy-control-plane.sh
   ```

   Then create an enroll access key once (CDK does not create keys):
   `aws iam create-access-key --user-name krabby-enroll --output json`

2. Deploy the fleet service stack (EC2, Cognito, tunnel API — required `-c`
   keys including GitHub OIDC trust: see
   [`infra/fleet-service.md`](infra/fleet-service.md)):

   ```bash
   ./scripts/deploy-fleet-service.sh \
     -c domainName=... -c hostedZoneName=... \
     -c githubOwner=... -c githubRepo=... -c githubBranch=... \
     -c githubOidcProviderArn=... -c cdkBootstrapQualifier=...
   ```

3. On each robot Orin, enroll and start the agent
   ([`ENROLL.md`](ENROLL.md)).

4. Verify shadow telemetry / portal / CLI as needed.

## Enroll and SSH

- Enroll one Orin: [`ENROLL.md`](ENROLL.md) — needs
  **`krabby-launcher` ≥ 0.1.22** (`enroll` / `agent`, fleet teleop signaling, bench `teleop_control_echo`)
- One SSH source → one Orin: [`SSH-TUNNEL.md`](SSH-TUNNEL.md)
- Cognito operators (CLI + Console): [`OPERATORS.md`](OPERATORS.md)

Later, with the fleet service up: `krabby-fleet ssh` / portal Open SSH
([`cli/README.md`](cli/README.md)).

**Scale path (document only):** when per-device enroll-time AWS creds no
longer scale, switch onboarding to AWS IoT Fleet Provisioning by claim —
see [`ENROLL.md` § Scale path](ENROLL.md#scale-path-not-implemented).

## MQTT topic scheme

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `$aws/things/{thingName}/shadow/update` | device → cloud | 1/min fleet telemetry in `state.reported` |
| `teleop/{thingName}/signaling/in` | cloud → device | WebRTC signaling |
| `teleop/{thingName}/signaling/out` | device → cloud | WebRTC signaling |
| `$aws/things/{thingName}/tunnels/notify` | cloud → device | Secure Tunneling destination token (reserved) |

Per-thing isolation is enforced by `KrabDevicePolicy` using
`${iot:Connection.Thing.ThingName}` — a device cert cannot publish or
subscribe on another device's topics.

On the robot, `krabby agent` bridges those teleop topics to
`ws://127.0.0.1:9000/ws/robot`. Run the HAL edge with
`--teleop-ip 127.0.0.1` so it dials the shim (message shape unchanged).

On the fleet host, `krabby-fleet-service` holds one persistent SigV4 MQTT
connection and bridges browser WebSockets at
`/api/devices/{thingName}/teleop/signaling` (Cognito operator token via
`Authorization: Bearer` or `?token=`) to the same topics. JSON signaling
shape is unchanged from the existing teleop stack.

## Bench control-plane E2E

Automated control-plane checks against a permanently enrolled bench Orin
(default thing name is configurable; see the e2e README): connectivity +
shadow via Fleet Indexing / `GetThingShadow`, per-thing isolation with a
scratch cert, and Secure Tunneling TCP through source `localproxy`. See
[`infra/tests_e2e/README.md`](infra/tests_e2e/README.md).

```bash
export BENCH_E2E=1
cd fleet/infra && source .venv/bin/activate
pip install -r tests_e2e/requirements.txt
pytest tests_e2e/ -q
```

## Runtime (EC2)

One host runs four systemd units (fleet service, portal, Caddy, coturn), all
logging to journald:

| Unit | Bind | Role |
|------|------|------|
| `krabby-fleet-service` | `127.0.0.1:8080` | Fleet REST API + teleop signaling / ICE |
| `krabby-fleet-portal` | `127.0.0.1:3000` | Next.js operator UI + NextAuth |
| `caddy` | `:443` / `:80` | TLS + reverse proxy |
| `krabby-coturn` | `:3478` UDP/TCP + relay `49152-65535/udp` | STUN/TURN for WebRTC |

Caddy routing (`fleet/service/deploy/Caddyfile`):

- `/api/auth*` → portal (NextAuth)
- `/api/*` → fleet service (strip `/api` prefix)
- everything else → portal

```bash
journalctl -u krabby-fleet-service -f
journalctl -u krabby-fleet-portal -f
journalctl -u caddy -f
journalctl -u krabby-coturn -f
```

Deploy apps + coturn with `fleet/infra/scripts/deploy-fleet-service.sh` (CDK + SSM).

## Fleet service API

Caddy serves the fleet service under `/api/*` (prefix stripped), except
`/api/auth*` which is reverse-proxied to the Next.js portal (NextAuth).
All fleet-service routes except `/healthz` require a Cognito access token
for a user in the `operator` group.

| Method | Path | Backend |
|--------|------|---------|
| `GET` | `/devices` | `iot:SearchIndex` (`thingTypeName:Krab`) — connectivity + shadow `reported` |
| `GET` | `/devices/{thingName}` | `iot:DescribeThing` + `iot:GetThingShadow` + SearchIndex connectivity |
| `POST` | `/devices/{thingName}/ssh-tunnel` | `iot:OpenTunnel` |
| `DELETE` | `/devices/{thingName}/ssh-tunnel/{tunnelId}` | `iot:CloseTunnel` |
| `GET` | `/teleop/ice-servers` | STUN + short-lived coturn TURN credentials (`iceServers`) |
| `WS` | `/devices/{thingName}/teleop/signaling` | MQTT bridge to `teleop/{thing}/signaling/in` and `.../out` |

## coturn (STUN/TURN)

`FleetServiceStack` opens:

| Port | Proto | Purpose |
|------|-------|---------|
| 3478 | UDP + TCP | STUN/TURN |
| 5349 | UDP + TCP | Reserved for TURNS (TLS/DTLS; not enabled in the default conf) |
| 49152–65535 | UDP | TURN relay allocations |

coturn uses the TURN REST API shared secret (`/krabby/fleet/turn-auth-secret` in
Secrets Manager). The fleet service mints per-session credentials
(`username = expiry:sub`, `credential = base64(hmac-sha1(secret, username))`,
default TTL 1h) and returns them from `GET /api/teleop/ice-servers` together
with Google public STUN.

### Force-relay verification

To confirm TURN works when direct ICE fails:

1. Open teleop for a robot and fetch ICE servers (portal/CLI will call
   `GET /api/teleop/ice-servers`).
2. In the browser console, create the peer connection with relay-only policy:

```js
const cfg = await (await fetch('/api/teleop/ice-servers', {
  headers: { Authorization: 'Bearer ' + accessToken }
})).json();
const pc = new RTCPeerConnection({
  iceServers: cfg.iceServers,
  iceTransportPolicy: 'relay',
});
```

3. Complete signaling as usual. The session should still connect; `chrome://webrtc-internals`
   (or Firefox `about:webrtc`) should show selected candidate type `relay`.

## Field teleop readiness

Enrolled robots: what must stay up for remote **Open teleop** to work, and how that
differs from default **`krabby run`** gamepad boot — see [`FIELD-TELEOP.md`](FIELD-TELEOP.md).
Bench CI uses the same signaling path with extra test flags: [`BENCH-TELEOP.md`](BENCH-TELEOP.md).

## Dual-robot teleop (public internet)

Two enrolled Orins must work concurrently from an operator SSH source on the
public internet — no VPN, no per-robot signaling servers.

### One-time per robot (Orin)

```bash
# Enroll once (device cert + krabby-agent.service) — see sections 3–6
sudo -E env PATH="$PATH" krabby enroll --thing-name <thing-name>

# Locomotion + WebRTC edge pointed at the local agent teleop shim
# (agent already bridges MQTT ↔ ws://127.0.0.1:9000/ws/robot)
python -m hal.server.jetson.main --control-source portal --teleop-ip 127.0.0.1
```

For the **bench Orin** used in CI, use the always-on agent + HAL procedure in
[`BENCH-TELEOP.md`](BENCH-TELEOP.md) (detached Docker, `--teleop-control-echo`).

No extra fleet registration steps: enroll attaches the thing; the portal
lists it via Fleet Indexing.

### Operator SSH source

Install and configure the CLI on an operator machine (not the Orin):
[`cli/README.md`](cli/README.md).

```bash
# from krabby-research repo root
pip install ./fleet/cli
# ~/.config/krabby-fleet/config.toml → service_url + cognito IDs
krabby-fleet list
krabby-fleet teleop <thing-name-a>   # browser tab 1
krabby-fleet teleop <thing-name-b>   # browser tab 2 (concurrent)
```

Or use **Open teleop** on the portal device list for each robot (new tab).
Signaling is per-thing MQTT (`teleop/{thing}/signaling/*`); media is WebRTC
(P2P / coturn). Two sessions = two browser tabs, one fleet MQTT bridge.

### Playwright E2E (bench only)

Automated teleop checks against a bench Orin (manual until CI is wired):
see [`service/tests_e2e/README.md`](service/tests_e2e/README.md).

```bash
export BENCH_E2E=1
export FLEET_PORTAL_URL=https://<fleet-domain>
export FLEET_SERVICE_URL=https://<fleet-domain>/api
# + Cognito IDs + AWS creds
cd fleet/service && pip install -e ".[e2e]" && playwright install chromium
pytest tests_e2e/test_teleop_e2e.py -q
```

The bench's HAL server also needs `--teleop-control-echo` added to its launch
command (alongside `--teleop-ip 127.0.0.1` above) -- it echoes the last
applied control state onto the telemetry channel so the test can confirm a
motion-safe control message was actually received by the real HAL, not just
sent by the browser. Off by default everywhere else (not a checked-in
`robot_settings.py` constant, since it's a per-run instrumentation flag, not
persistent per-robot deployment config); no operator-facing feature reads it.

## Portal + CLI

- Portal: [`portal/`](portal/README.md) — Cognito sign-in, device list/detail,
  **Open teleop** (existing teleop viewer via fleet signaling + ICE), Open SSH.
- CLI: `krabby-fleet list` prints online/last-seen + telemetry summary;
  `krabby-fleet teleop <robot>` opens the same teleop URL in a browser;
  `krabby-fleet ssh <robot>` opens Secure Tunneling SSH.
## Telemetry

### Topic and rate

`krabby agent` publishes once per minute to the device's Classic Shadow:

```json
{"state": {"reported": { ... }}}
```

The fleet portal and `krabby-fleet list` read the latest document via Fleet
Indexing (`iot:SearchIndex`) or `iot:GetThingShadow`.

### Schema (open-ended)

There is **no fixed JSON Schema** for `state.reported`. The canonical source
of truth on the device is:

```bash
krabby get telemetry
```

Whatever that command prints is what `krabby agent` uploads to the shadow on
the next 1/min tick. New fields may appear as the locomotion stack and host
expose more signals; consumers should treat unknown keys as opaque.

### Typical top-level keys

These fields are populated when the underlying source is available:

| Key | Source | Notes |
|-----|--------|-------|
| `timestamp` | wall clock | Unix epoch seconds |
| `reported_image` | `~/.config/krabby/state.json` | Running locomotion image ref from `krabby install` / `krabby update`. May later become an S3 object reference for a camera snapshot (`KrabReportedImages` bucket). |
| `health` | host | `locomotion_container`, `krabby_agent`, `krabby_locomotion` systemd states, `mcu_present` |
| `imu` | HAL (locomotion container) | `base_quat_w`, `base_ang_vel_b`, `base_lin_vel_b` when the HAL server is publishing observations |
| `pose` | HAL | `joint_positions`, `joint_velocities` |
| `power` | sysfs | `supplies[]` with `capacity_percent`, `voltage_v`, `current_a` when exposed by the board |
| `red_flags` | derived | Strings such as `agent_not_running`, `locomotion_container_down`, `hal_no_observation`, `mcu_missing` |

Example (fields vary by runtime state):

```json
{
  "timestamp": 1710000000,
  "reported_image": "public.ecr.aws/t7t7b3i3/krabby-locomotion:release-latest",
  "health": {
    "locomotion_container": "running",
    "krabby_agent": "active",
    "krabby_locomotion": "active",
    "mcu_present": true
  },
  "imu": {
    "base_quat_w": [0.0, 0.0, 0.0, 1.0],
    "base_ang_vel_b": [0.0, 0.0, 0.0],
    "base_lin_vel_b": [0.0, 0.0, 0.0]
  },
  "pose": {
    "joint_positions": [0.1, 0.2],
    "joint_velocities": [0.0, 0.0]
  },
  "power": {
    "supplies": [{"name": "BAT0", "capacity_percent": 87}]
  },
  "red_flags": []
}
```

### `reported_image`

Today this is the installed locomotion image tag/ref. Camera frame uploads to
the `KrabReportedImages` S3 bucket (via the `KrabImageRoleAlias` credentials
provider) are infrastructure-ready in `ControlPlaneStack` but not yet wired
into the agent — when added, `reported_image` will hold an S3 URI instead of
inline image bytes (shadow documents must stay small).

## Device services

| Unit | Started by | Role |
|------|------------|------|
| `krabby-locomotion.service` | `krabby install` | Runs `krabby run` (HAL + controller container) |
| `krabby-agent.service` | `krabby enroll` | Runs `krabby agent` (MQTT shadow + tunnel notify + teleop shim) |

Logs:

```bash
journalctl -u krabby-agent -f
journalctl -u krabby-locomotion -f
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Shadow `timestamp` stale | `systemctl status krabby-agent`; `journalctl -u krabby-agent` |
| Empty `imu` / `pose` | `docker ps` shows locomotion running; HAL publishing on `:6001` |
| `red_flags` includes `mcu_missing` | USB hub + `/dev/ttyACM0` present |
| Enroll fails on policy | Deploy `ControlPlaneStack` first; use `krabby-enroll` keys |
| Apt missing `localproxy` / enroll issues | See [`ENROLL.md`](ENROLL.md) |
| Tunnel / SSH issues | See [`SSH-TUNNEL.md`](SSH-TUNNEL.md) |

## Rollback

Redeploy CDK at a prior git SHA:

```bash
git checkout <sha>
cd fleet/infra && source .venv/bin/activate
./scripts/deploy-control-plane.sh
./scripts/deploy-fleet-service.sh \
  -c domainName=... -c hostedZoneName=... \
  -c githubOwner=... -c githubRepo=... -c githubBranch=... \
  -c githubOidcProviderArn=... -c cdkBootstrapQualifier=...
```

SSM restart on the fleet EC2 picks up new service artifacts after
`fleet-deploy.yml` runs.

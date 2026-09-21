# Fleet service live E2E (SSH + teleop)

Tests against a **deployed** fleet host and an enrolled bench Orin. Non-secret
settings come from committed [`../config/fleet.toml`](../config/fleet.toml)
(see [`../config/README.md`](../config/README.md)).

## When tests run

| Context | Behavior |
|---------|----------|
| Local `pytest tests_e2e/` (default) | **Skipped** — unit runs stay green |
| `BENCH_E2E=1` or GitHub Actions | **Pass/fail** — missing config, password, tools, or bench → red job |

## Config vs secrets

| Source | What |
|--------|------|
| `fleet/config/fleet.toml` | URLs, region, Cognito pool/client IDs, bench thing name, CI operator email |
| GitHub secret `COGNITO_CI_PASSWORD` | CI operator password only |
| GitHub secret `BENCH_CI_SSH_PRIVATE_KEY` | Ed25519 private key for SSH login as `operator` on the bench (see [`../../BENCH-SSH.md`](../../BENCH-SSH.md)) |
| Env vars | Optional overrides of any committed value |

## CI scope

**Happy path only**: list devices, SSH tunnel, teleop signaling/video/control, authed
ICE servers. Uses the persistent CI operator (`[ci].operator_username` +
`COGNITO_CI_PASSWORD`) — no scratch Cognito users, no negative-auth cases.

Teleop also needs runner AWS creds with `iot:DescribeEndpoint` and MQTT SigV4 on
`teleop/*/signaling/*` (signaling sniffer).

Bench preconditions for teleop (always-on setup on the Orin):
[`../../BENCH-TELEOP.md`](../../BENCH-TELEOP.md).

Playwright opens the viewer with **`?e2e=1`**: one recvonly video line and matching
``catalog_ids`` (HAL rejects m-line / catalog length mismatch). ICE uses normal
STUN+TURN from ``/api/teleop/ice-servers`` (not ``?ice=relay`` — relay-only stalls
headless Chromium). Deployed portal must ship current `teleop_session.js`.

* `krabby-agent.service` — MQTT + teleop shim on **`127.0.0.1:9000`**
* HAL in **portal** mode with **`--teleop-ip 127.0.0.1`**, **`--teleop-control-echo`**
  (persistent Docker — not **`krabby run`** / gamepad container **`krabby`**)

## Run

```bash
cd fleet/service
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../config -e ".[e2e]"
playwright install --with-deps chromium

export COGNITO_CI_PASSWORD='…'   # CI operator password
export BENCH_E2E=1

pytest tests_e2e/ -q
```

SSH round-trip also needs `krabby-fleet` CLI, `localproxy` on PATH, and pubkey
auth as `operator` on the bench — setup: [`../../BENCH-SSH.md`](../../BENCH-SSH.md).

## Coverage

| Test | Checks |
|------|--------|
| `test_open_and_close_tunnel_happy_path` | Operator opens/closes SSH tunnel via REST |
| `test_get_devices_*` | List + get device shadow for bench |
| `test_krabby_fleet_ssh_runs_command_end_to_end` | CLI SSH echo through Secure Tunnel |
| `test_teleop_signaling_control_and_video` | Portal viewer → live session (ICE + control DC); control echo + video; MQTT idle after close |
| `test_teleop_ice_servers_authed` | ICE endpoint returns STUN with operator token |

# Bench control-plane E2E

Live integration tests against an enrolled bench Orin and real IoT Core
(`ControlPlaneStack`). Skipped locally unless `BENCH_E2E=1` or GitHub Actions;
when enabled, missing config, AWS access, tools, or bench → **fail**.

Non-secret settings come from committed [`../config/fleet.toml`](../config/fleet.toml).

## Enable

```bash
export BENCH_E2E=1
# optional overrides:
export SHADOW_MAX_AGE_SECS=180
export TUNNEL_E2E_DEBUG=1   # optional: live stderr during waits; failures always include poll trace
```

AWS credentials must allow: `iot:SearchIndex`, `iot:GetThingShadow`,
`iot:DescribeThing` / `CreateThing` / `DeleteThing`, cert lifecycle,
`iot:AttachPolicy` / `DetachPolicy`, `iot:OpenTunnel` / `CloseTunnel`, plus
MQTT data-plane via device certs created in-test.

## Run

```bash
cd fleet/infra
source .venv/bin/activate   # from ./scripts/setup-venv.sh
pip install -e ../config -r tests_e2e/requirements.txt
pytest tests_e2e/ -q
```

## Coverage

| Test | Checks |
|------|--------|
| `test_control_plane_prereqs` | Persistent stack: `Krab` thing type + `KrabDevicePolicy` exist |
| `test_bench_connectivity_and_shadow_index` | `SearchIndex`: connected + recent `reported.timestamp` |
| `test_bench_get_thing_shadow_schema` | `GetThingShadow` has expected `reported` keys |
| `test_scratch_cert_cannot_update_bench_shadow` | Second-device cert cannot write bench shadow |
| `test_secure_tunnel_source_proxy_reaches_ssh` | `OpenTunnel` + source `localproxy` + SSH banner over TCP |

Bench offline or stale shadow → **red** when bench E2E is enabled.

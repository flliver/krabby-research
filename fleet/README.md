# Fleet

AWS IoT control plane, device enroll, Secure Tunneling SSH, fleet service,
portal, and CLI.

| Doc | Purpose |
|-----|---------|
| [`config/`](config/README.md) | Committed `fleet.toml` + Python loader (CI, CLI, E2E) |
| [`ENROLL.md`](ENROLL.md) | Enroll one Orin |
| [`FIELD-TELEOP.md`](FIELD-TELEOP.md) | Field robots — teleop-ready vs agent-only |
| [`BENCH-TELEOP.md`](BENCH-TELEOP.md) | Bench Orin always-on agent + HAL for teleop CI |
| [`BENCH-SSH.md`](BENCH-SSH.md) | Bench `operator` user + CI SSH pubkey |
| [`SSH-TUNNEL.md`](SSH-TUNNEL.md) | One SSH source → one Orin |
| [`OPERATORS.md`](OPERATORS.md) | Add Cognito operator users (CLI + Console) |
| [`SETUP-FLEET.md`](SETUP-FLEET.md) | Deploy sequence, MQTT, telemetry, ops |
| [`infra/`](infra/README.md) | CDK stacks |
| [`cli/`](cli/README.md) | `krabby-fleet` install + usage |
| [`portal/`](portal/README.md) | Operator UI |
| [`service/`](service/README.md) | Fleet REST API |

## See enrolled Orins (AWS Console)

Same region as the control plane. Enrolled robots are IoT things of type
**`Krab`** (there is no separate “Orins” page).

- **IoT Core → Manage → All devices → Things** — open a thing for certs,
  shadow, connectivity.
- **IoT Core → Manage → Fleet indexing** (or Things search) — query
  `thingTypeName:Krab` or `thingName:<thing-name>` for connectivity /
  last seen.

CLI: `aws iot search-index --query-string 'thingTypeName:Krab'`. Portal /
`krabby-fleet list` once the fleet service is up.

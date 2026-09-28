# krabby-fleet-service

Cognito-authenticated REST API that lets operators open/close AWS IoT Secure
Tunneling SSH sessions to fleet robots, without the browser or CLI ever
seeing AWS credentials. Runs on the `FleetServiceStack` EC2 host, using that
instance's own IAM role -- no static AWS credentials anywhere in this
service.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | none | Liveness + signaling diagnostics (`mqttConnected`, `signalingOutSubscribed`, `activeSessionsByThing`) |
| `GET` | `/devices` | operator group | Lists enrolled robots via Fleet Indexing (`SearchIndex`, `thingTypeName:Krab`). Returns `[{thingName, connected, connectivityTimestamp, reported}]` where `reported` is the latest classic-shadow `state.reported` document. |
| `GET` | `/devices/{thingName}` | operator group | Device detail: `DescribeThing` metadata + connectivity from SearchIndex + full `state.reported` from `GetThingShadow`. Returns `{thingName, thingTypeName, attributes, connected, connectivityTimestamp, reported}`. |
| `POST` | `/devices/{thingName}/ssh-tunnel` | operator group | Opens a Secure Tunnel (`OpenTunnel`, `services=SSH`) to the device. Returns `{tunnelId, sourceAccessToken, region}` -- the *destination* token goes straight to the device over MQTT (`krabby agent`, not through this service). |
| `DELETE` | `/devices/{thingName}/ssh-tunnel/{tunnelId}` | operator group | Force-closes the tunnel (`CloseTunnel`, `delete=True`). |
| `GET` | `/teleop/ice-servers` | operator group | Returns `{"version":1,"iceServers":[...],"ttlSeconds":N}` — Google STUN plus short-lived coturn TURN credentials (HMAC REST API). Same `iceServers` shape as the existing teleop stack's `GET /api/teleop-config`. |
| `WS` | `/devices/{thingName}/teleop/signaling` | operator group | Bidirectional WebRTC signaling bridge: browser JSON ↔ IoT MQTT `teleop/{thingName}/signaling/in` and `.../out`. Message shape is unchanged from the existing teleop stack. |

`deploy/Caddyfile` reverse-proxies `/api/auth*` to the Next.js portal
(`:3000`), other `/api/*` to this service (stripping the `/api` prefix), and
everything else to the portal. The routes above are what this app itself
serves, on `127.0.0.1:8080`. Externally the signaling socket is
`wss://<fleet-host>/api/devices/{thingName}/teleop/signaling`.

## Auth

Every non-`/healthz` route requires `Authorization: Bearer <Cognito access
token>`. The token is verified against the Cognito user pool's JWKS
(`_auth.py`): signature, issuer, `token_use=access`, and `client_id` must
all check out (401 otherwise), and the token's `cognito:groups` claim must
include `operator` (403 otherwise). Group membership is granted manually by
an admin -- see [`../OPERATORS.md`](../OPERATORS.md) (CLI + Console).
`FleetServiceStack` creates the empty `operator` group but doesn't populate it.

The teleop WebSocket also accepts `?token=<access-token>` (browsers often
cannot set an Authorization header on `WebSocket`).

## Configuration

Reads values from SSM Parameter Store at startup (published by
`FleetServiceStack`; the instance role has read access):

| SSM parameter | Purpose |
|---|---|
| `/krabby/fleet/cognito-user-pool-id` | Which Cognito user pool to validate tokens against |
| `/krabby/fleet/cognito-app-client-id` | Expected `client_id` claim |
| `/krabby/fleet/iot-ats-endpoint` | IoT Core ATS data endpoint for the signaling MQTT client |

TURN settings come from `/etc/krabby-fleet/service.env` (written by deploy from
Secrets Manager `/krabby/fleet/turn-auth-secret` + the fleet DNS name):

| Env var | Purpose |
|---|---|
| `KRABBY_FLEET_TURN_AUTH_SECRET` | coturn `static-auth-secret` (HMAC key) |
| `KRABBY_FLEET_TURN_HOST` | Hostname in `turn:` URLs (fleet domain) |
| `KRABBY_FLEET_TURN_TTL_SECS` | Credential lifetime (default 3600) |
| `KRABBY_FLEET_LOG_LEVEL` | App log level (default `INFO`) |
| `KRABBY_FLEET_SIGNALING_TRACE` | Set to `1` to log every teleop/MQTT frame at DEBUG (temporary triage only) |

For local dev/tests, set `KRABBY_FLEET_COGNITO_USER_POOL_ID` /
`KRABBY_FLEET_COGNITO_APP_CLIENT_ID` / `KRABBY_FLEET_IOT_ATS_ENDPOINT` /
`KRABBY_FLEET_TURN_AUTH_SECRET` / `KRABBY_FLEET_TURN_HOST` env vars instead.
`AWS_REGION` (or `AWS_DEFAULT_REGION`) controls Cognito issuer, Secure
Tunneling, and MQTT SigV4 signing; defaults to `us-east-1`.

## Teleop signaling bridge

On startup the service opens **one** persistent MQTT connection to IoT Core
(SigV4 over websockets, instance-role credentials) and subscribes to
`teleop/+/signaling/out`. Each authenticated browser session:

* publishes WS text frames to `teleop/{thing}/signaling/in` (cloud → robot)
* receives MQTT frames from `teleop/{thing}/signaling/out` as WS text (robot → cloud)

The robot side is `krabby agent`'s localhost shim (`--teleop-ip 127.0.0.1`).

**Production logging:** at default `INFO`, the journal records startup, MQTT
connect/subscribe/resume, teleop session open/close, stalls, and errors — not
every `hello`/`ping` (those are DEBUG). Enable
`KRABBY_FLEET_SIGNALING_TRACE=1` in `/etc/krabby-fleet/service.env` only while
debugging, then remove it.

### Teleop bridge triage (one deploy)

After deploy + `systemctl restart krabby-fleet-service` and `systemctl reload caddy`,
open portal teleop once and map **journalctl** lines to the failure:

| Log line | Meaning |
|---|---|
| `fleet startup region=… iot_ats=…` | Config loaded; verify region matches IoT console |
| `fleet MQTT connected …` | CRT MQTT path OK at startup |
| `fleet MQTT subscribed filter=teleop/+/signaling/out` | Return path armed |
| `teleop signaling session open thing=…` | Browser reached Uvicorn (auth is DEBUG unless tracing) |
| `teleop signaling stall … no text frame in 30s` | Caddy/proxy or browser not delivering WS payloads |
| `signaling browser→mqtt …` / `fleet MQTT publish ack …` | DEBUG — enable `KRABBY_FLEET_SIGNALING_TRACE=1` |
| `publish signaling/in failed` / `fleet MQTT publish failed` | IAM, disconnect, or wrong endpoint — full stack trace in journal |
| `signaling mqtt→browser …` but `dropped … no active WebSocket` | Robot `out` arrived with no portal session |
| `fleet MQTT interrupted` / `connection failure` | Long-lived bridge issue; resubscribe runs on resume |

**Before** blaming the bridge, run MQTT smoke on the instance (same CRT client as the service):

```bash
sudo -u krabby-fleet AWS_REGION=us-east-2 AWS_DEFAULT_REGION=us-east-2 \
  python3 /opt/krabby-fleet-service/src/scripts/signaling_mqtt_smoke.py --thing bench-krabby-ci
```

If smoke **publish ack** appears on IoT `…/signaling/in` but portal does not, the bug is WebSocket/Caddy. If smoke **fails**, fix IAM/endpoint before retesting teleop.

```bash
curl -sS http://127.0.0.1:8080/healthz | jq .
sudo journalctl -u krabby-fleet-service --since "5 minutes ago" | grep -E 'signaling|fleet MQTT|teleop'
```

## ICE / TURN

`GET /teleop/ice-servers` returns Google STUN plus coturn TURN URLs on the
fleet host (`turn:<domain>:3478` UDP and TCP) with short-lived REST-API
credentials. coturn runs as `krabby-coturn.service` on the same EC2 (see
`deploy/coturn.service` and `deploy/turnserver.conf.in`).

## Run locally

```bash
cd fleet/service
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export KRABBY_FLEET_COGNITO_USER_POOL_ID={cognito-user-pool-id}
export KRABBY_FLEET_COGNITO_APP_CLIENT_ID={cognito-app-client-id}
export KRABBY_FLEET_IOT_ATS_ENDPOINT={iot-ats-endpoint}
export KRABBY_FLEET_TURN_AUTH_SECRET={turn-shared-secret}
export KRABBY_FLEET_TURN_HOST={fleet-domain}
export AWS_REGION={aws-region}
krabby-fleet-service   # binds 127.0.0.1:8080
```

Opening a real tunnel or signaling MQTT still needs real AWS credentials
with the FleetServiceStack instance-role permissions (`iot:OpenTunnel` /
`CloseTunnel` / `DescribeTunnel` plus `iot:Connect`/`Publish`/`Subscribe`/
`Receive` on `teleop/*/signaling/*`).
For local testing, whatever `boto3` / the CRT default credential chain
resolves from your environment.

## Tests

```bash
cd fleet/service
pip install -e ".[dev]"
pytest tests/
```

No real AWS or Cognito access needed -- `test_auth.py` signs test JWTs with
a locally generated RSA keypair and mocks JWKS resolution; `test_tunnels.py`
and `test_devices.py` mock boto3 clients; `test_signaling.py` uses a fake
MQTT client; `test_ice.py` checks TURN HMAC minting and the ICE endpoint.

## Deploy artifacts

`systemd/krabby-fleet-service.service` runs `krabby-fleet-service` as a
dedicated `krabby-fleet` system user (not root -- this service needs no
special privileges). `deploy/Caddyfile` reverse-proxies `/api/auth*` to the
portal on `:3000`, other `/api/*` to this service (stripping `/api`), and
everything else to the portal. `deploy/caddy.service` runs Caddy as its own
dedicated `caddy` system user. `fleet/infra/scripts/deploy-fleet-service.sh`
pushes this directory **and** `fleet/portal` onto the `FleetServiceStack`
instance and (re)starts `krabby-fleet-service`, `krabby-fleet-portal`, and
`caddy` on every deploy -- see `fleet/infra/fleet-service.md`.

# SSH over Secure Tunneling (one source → one Orin)

Open an SSH session from one **SSH source** (operator machine) to one
enrolled Orin. No fleet portal or `krabby-fleet` required — only AWS IoT
Secure Tunneling + `localproxy` on both ends.

Placeholders: `<region>`, `<thing-name>`, `<ssh-user>` (a real login on the
Orin). Tip: `export AWS_PAGER=""`.

```
SSH source                         Orin
  open-tunnel ──────────────────►  krabby-agent (MQTT tunnels/notify)
  localproxy (source :5555) ◄───►  localproxy (destination → sshd :22)
  ssh -p 5555 user@localhost
```

## Prerequisites

- Orin enrolled, agent running, destination `localproxy` on PATH —
  [`ENROLL.md`](ENROLL.md).
- SSH source: AWS CLI creds (can open tunnels), `localproxy`, `ssh`, `jq`.

Agent must be up **before** you open a tunnel (notify is delivered once).

### SSH source: install `localproxy`

Apt if available, else Docker extract (same as Orin in [`ENROLL.md`](ENROLL.md),
use `amd64-latest` on x86_64). Confirm: `command -v localproxy`.

## Connect

**Terminal A (SSH source)** — open the tunnel and keep `localproxy` in the
foreground:

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=<region>
TUNNEL_JSON=$(aws iotsecuretunneling open-tunnel --destination-config thingName=<thing-name>,services=SSH --timeout-config maxLifetimeTimeoutMinutes=30 --output json)
export SOURCE_TOKEN=$(echo "$TUNNEL_JSON" | jq -r .sourceAccessToken)
export TUNNEL_ID=$(echo "$TUNNEL_JSON" | jq -r .tunnelId)
localproxy -s 5555 -t "$SOURCE_TOKEN" -r "$AWS_DEFAULT_REGION" -c /etc/ssl/certs
```

Always pass `-c /etc/ssl/certs`. Omitting it can fail SSL handshake
(`unregistered scheme`). Destination mode in `krabby agent` already passes it.

**Terminal B (SSH source)** — after `localproxy` is listening in Terminal A:

```bash
ssh -o StrictHostKeyChecking=no -p 5555 <ssh-user>@localhost
```

## Disconnect

In Terminal B, exit the SSH session. In Terminal A, stop `localproxy`
(Ctrl-C), then:

```bash
aws iotsecuretunneling close-tunnel --tunnel-id "$TUNNEL_ID" --delete
```

(`TUNNEL_ID` is still set in Terminal A. If that shell is gone:
`aws iotsecuretunneling list-tunnels` / close by id.)

## Reopen after fixes

Tunnel notify is **one-shot** at `OpenTunnel`. After changing agent or
`localproxy` on the Orin, close the tunnel and open a **new** one. Do not
reuse an old `SOURCE_TOKEN` (you get `403`).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No destination `localproxy` on Orin | [`ENROLL.md`](ENROLL.md); `journalctl -u krabby-agent -f --no-pager` while opening tunnel |
| SSL / `unregistered scheme` | `-c /etc/ssl/certs` on both ends |
| `403` / notify never arrives | Close + new open-tunnel; agent must be running first |
| SSH auth fails | Real Orin account as `<ssh-user>` |

Fleet service path later: `krabby-fleet ssh` ([`cli/README.md`](cli/README.md)).

# krabby-bench

Bench watchdog for the Krabby locomotion stack. Polls ECR for new `mainline-latest` digests, runs a firmware smoke test when one appears, and alerts on failure.

## Install

```bash
sudo pip3 install krabby-bench
```

Then bootstrap the systemd service as root.

### SSM mode (recommended for fleet use)

Credentials live in AWS SSM Parameter Store and are fetched at runtime. Nothing sensitive is stored in plaintext on the device.

```bash
sudo \
  BENCH_AWS_KEY_ID=AKIA... \
  BENCH_AWS_SECRET_KEY=... \
  krabby-bench install \
    --ssm-prefix /krabby/bench \
    [--ecr-tag mainline-latest] \
    [--firmware-channel release/0.2.9] \
    [--mode both]
```

`install` writes `/etc/krabby-bench/config.toml`, then enables and starts the service.

#### SSM parameter layout

Create these in AWS SSM Parameter Store before or after installing. The service starts without them and logs a single warning; it picks them up automatically within one `credentials_refresh_interval` (default: 3600 s) once they exist.

| Path | Type | Description |
|---|---|---|
| `/krabby/bench/smtp-host` | String | SMTP server hostname |
| `/krabby/bench/smtp-port` | String | SMTP port (default `587`) |
| `/krabby/bench/smtp-user` | String | SMTP login username |
| `/krabby/bench/smtp-password` | SecureString | SMTP login password |
| `/krabby/bench/smtp-from` | String | From address |
| `/krabby/bench/smtp-to` | String | Alert recipient address |
| `/krabby/bench/github-repo` | String | `owner/repo` to open issues against |
| `/krabby/bench/github-token` | SecureString | Fine-grained PAT with Issues write scope |

#### IAM policy

The IAM user whose access key is passed to `install` needs only:

```json
{
  "Effect": "Allow",
  "Action": "ssm:GetParametersByPath",
  "Resource": "arn:aws:ssm:*:*:parameter/krabby/bench/*"
}
```

#### Credential rotation

Update values in SSM. Devices pick up the new credentials within one poll interval — no SSH required.

To rotate the AWS access key, re-run `install` with the new key:

```bash
sudo BENCH_AWS_KEY_ID=AKIANEW... BENCH_AWS_SECRET_KEY=... \
  krabby-bench install --ssm-prefix /krabby/bench
```

### Legacy mode

Pass credentials via environment variables. Written to `/etc/krabby-bench/smtp.env` (mode 600) and loaded by the systemd unit.

```bash
sudo \
  BENCH_SMTP_HOST=smtp.example.com \
  BENCH_SMTP_PORT=587 \
  BENCH_SMTP_USER=krabby-errors@example.com \
  BENCH_SMTP_PASSWORD=secret \
  BENCH_SMTP_FROM=krabby-errors@example.com \
  BENCH_SMTP_TO=krabby-errors@example.com \
  BENCH_GITHUB_REPO=owner/krabby-research \
  BENCH_GITHUB_TOKEN=ghp_... \
  krabby-bench install [--ecr-tag mainline-latest] [--firmware-channel release/0.2.9] [--mode both]
```

#### Legacy environment variables

| Variable | Required for | Description |
|---|---|---|
| `BENCH_SMTP_HOST` | email alerts | SMTP server hostname |
| `BENCH_SMTP_PORT` | email alerts | SMTP port (default `587`) |
| `BENCH_SMTP_USER` | email alerts | SMTP login username |
| `BENCH_SMTP_PASSWORD` | email alerts | SMTP login password |
| `BENCH_SMTP_FROM` | email alerts | From address |
| `BENCH_SMTP_TO` | email alerts | Alert recipient address |
| `BENCH_GITHUB_REPO` | GitHub alerts | `owner/repo` to open issues against |
| `BENCH_GITHUB_TOKEN` | GitHub alerts | Fine-grained PAT with Issues write scope |

## Config

Non-secret fields only — credentials come from SSM or the env vars above.

Default path: `/etc/krabby-bench/config.toml`

```toml
[ecr]
repo = "public.ecr.aws/t7t7b3i3/krabby-locomotion"
tag = "mainline-latest"
poll_interval = 60          # seconds

[smoke]
firmware_channel = "release/0.2.9"
run_hal_check = false

[alert]
mode = "both"               # "email" | "github" | "both"
dedup_window = 3600         # suppress repeat alerts for the same failure (seconds)

[github]
repo = "owner/krabby-research"

[ssm]
prefix = "/krabby/bench"
credentials_refresh_interval = 3600   # how often to re-fetch from SSM (seconds)
```

## Smoke test

For each new digest the watchdog:

1. Runs `krabby firmware show` to discover attached board ports.
2. Runs `krabby firmware update <channel> <port>` for each port.
3. Runs `krabby firmware show` again and parses the version strings.
4. Asserts all three boards report the same version.
5. Fetches `https://krabby-firmware-public.s3.amazonaws.com/<channel>/latest.json` and checks the version matches the S3 manifest.

## Monitor

```bash
journalctl -fu krabby-bench
```

## Force a failure (test alert path)

Unplug one Mega. Clear the state file to trigger a re-test on the next poll:

```bash
sudo bash -c 'echo "{}" > /var/lib/krabby-bench/state.json'
sudo systemctl restart krabby-bench
```

Within one poll cycle the watchdog detects the failure and fires an alert.

## State file

`/var/lib/krabby-bench/state.json` — persists the last-tested digest and last-alert metadata. Clear it to force a re-test on the next poll.

## Local development

Set `BENCH_SMTP_*` and `BENCH_GITHUB_TOKEN` env vars directly; the watchdog reads them as fallback when no SSM prefix is configured:

```bash
BENCH_SMTP_HOST=smtp.example.com ... python -m krabby_bench.watchdog
```

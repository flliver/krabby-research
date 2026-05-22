# krabby-bench

Bench watchdog for the Krabby locomotion stack. Polls ECR for new `mainline-latest` digests, runs a firmware smoke test when one appears, and alerts on failure.

## Install

```bash
sudo pip3 install krabby-bench
```

Then bootstrap the systemd service as root, passing credentials via environment variables:

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

`install` writes `/etc/krabby-bench/config.toml`, `/etc/krabby-bench/smtp.env` (mode 600), and `/etc/systemd/system/krabby-bench.service`, then enables and starts the service. Credentials live only on the device — never in source control.

### Environment variables

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

SMTP vars are written to `/etc/krabby-bench/smtp.env` and loaded by the systemd unit at runtime. They can also be set directly in the environment when testing without the service.

## Config

Non-secret fields only — credentials come from the env vars above.

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
token = ""                  # leave blank — set BENCH_GITHUB_TOKEN instead
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

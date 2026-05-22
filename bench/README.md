# krabby-bench

Bench watchdog for the Krabby locomotion stack. Polls ECR for new `mainline-latest` digests, runs a firmware smoke test when one appears, and alerts on failure.

## Install

```bash
pip install krabby-bench
```

## Config

Default config path: `/etc/krabby-bench/config.toml`

```toml
[ecr]
repo = "public.ecr.aws/t7t7b3i3/krabby-locomotion"
tag = "mainline-latest"
poll_interval = 60          # seconds

[smoke]
firmware_channel = "mainline"
run_hal_check = false       # set true to also start/stop container and check telemetry

[alert]
mode = "email"              # "email" | "github" | "both"
dedup_window = 3600         # suppress repeat alerts for the same failure within this window (seconds)

[smtp]
host = "smtp.example.com"
port = 587
user = "user@example.com"
password = "secret"
from = "krabby-bench@example.com"
to = "oncall@example.com"

[github]
repo = "owner/krabby-research"
token = "ghp_..."
```

## Smoke test

For each new digest the watchdog:

1. Runs `krabby firmware update <channel>` (flashes all three boards).
2. Runs `krabby firmware show` and parses the version strings.
3. Asserts all three boards report the same version.
4. Fetches `https://krabby-firmware-public.s3.amazonaws.com/<channel>/latest.json` and checks the reported version matches the S3 manifest.

## systemd

Copy the unit and timer files, then enable:

```bash
sudo cp bench/systemd/krabby-bench.service /etc/systemd/system/
sudo cp bench/systemd/krabby-bench.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now krabby-bench
```

Monitor:

```bash
journalctl -fu krabby-bench
```

## Force a failure (test alert path)

Unplug one Mega, then push a dummy commit to `mainline` to trigger a new ECR digest. Within one poll cycle the watchdog should detect the new digest, run the smoke test, and fire an alert.

To reset: replug the board, let the next poll cycle pass, confirm no new alert.

## State file

`/var/lib/krabby-bench/state.json` — persists the last-tested digest and last-alert metadata. Delete it to force a re-test on the next poll.

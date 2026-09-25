# krabby-bench

Watches ECR for new locomotion images, updates the stack when one appears, runs a firmware smoke test, and alerts on failure.

## Install

**Prerequisite:** `krabby-launcher` must be installed and `krabby install` run before krabby-bench — the watchdog calls the `krabby` CLI to pull images and flash firmware. See the [top-level README](../README.md) for the full setup sequence.

```bash
sudo pip3 install krabby-launcher krabby-bench
sudo krabby install
```

Then run the install command as root to configure and start the watchdog service:

```bash
sudo krabby-bench install [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--ssm-prefix` | `/krabby/bench` | SSM parameter path prefix. When set, credentials are loaded from AWS SSM. |
| `--ecr-tag` | `mainline-latest` | ECR image tag to watch. |
| `--firmware-channel` | `release/0.2.9` | Firmware channel for smoke tests. |
| `--error-alert-type` | `both` | Alert delivery type: `email`, `github`, or `both`. |
| `--github-repo` | _(none)_ | `owner/repo` to open issues against (legacy mode only). |

#### SSM mode (recommended for fleet use)

Pass IAM access keys at install time; the service fetches credentials from SSM at runtime and re-reads them automatically on each refresh interval.

```bash
sudo \
  BENCH_AWS_KEY_ID=AKIA... \
  BENCH_AWS_SECRET_KEY=... \
  krabby-bench install \
    [--ssm-prefix /krabby/bench] \
    [--ecr-tag mainline-latest] \
    [--firmware-channel release/0.2.9] \
    [--error-alert-type both]
```

`install` writes `/etc/krabby-bench/config.toml`, then enables and starts the service.

##### SSM parameter layout

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

##### IAM policy

The IAM user whose access key is passed to `install` needs only:

```json
{
  "Effect": "Allow",
  "Action": "ssm:GetParametersByPath",
  "Resource": "arn:aws:ssm:*:*:parameter/krabby/bench/*"
}
```

##### Credential rotation

Update values in SSM. Devices pick up the new credentials within one poll interval — no SSH required.

To rotate the AWS access key, re-run `install` with the new key:

```bash
sudo BENCH_AWS_KEY_ID=AKIANEW... BENCH_AWS_SECRET_KEY=... \
  krabby-bench install --ssm-prefix /krabby/bench
```

#### Legacy mode

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
  krabby-bench install [--ecr-tag mainline-latest] [--firmware-channel release/0.2.9] [--error-alert-type both]
```

##### Legacy environment variables

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

For verbose output on a one-off manual run:

```bash
krabby-bench --log-level DEBUG
```

When running as a systemd service, the log level must be set in the unit file — systemd launches the process with a fixed command line, so flags can't be passed after the fact. Edit `/etc/systemd/system/krabby-bench.service`:

```
ExecStart=/usr/local/bin/krabby-bench --log-level DEBUG --config /etc/krabby-bench/config.toml
```

Then apply and restart:

```bash
sudo systemctl daemon-reload && sudo systemctl restart krabby-bench
```

Valid levels: `DEBUG`, `INFO` (default), `WARNING`, `ERROR`.

## Force a recheck

To make the watchdog re-run the update + smoke test immediately — regardless of whether the ECR digest has changed — run:

```bash
sudo krabby-bench force-recheck
```

This signals the running service to clear its last-tested digest. The recheck runs on the next poll cycle (within one `poll_interval`).

## Force a failure (test alert path)

Unplug one Mega. Clear the state file to trigger a re-test on the next poll:

```bash
sudo bash -c 'echo "{}" > /var/lib/krabby-bench/state.json'
sudo systemctl restart krabby-bench
```

Within one poll cycle the watchdog detects the failure and fires an alert.

## State file

`/var/lib/krabby-bench/state.json` — persists the last-tested digest and last-alert metadata. Clear it to force a re-test on the next poll.

## Four-stage harness (install → flash → bringup → motion)

One-shot CI path for cold-start verification. Extends this package rather than a second bench
service. Dual-use Orin: uses `~/.venv-krabby-bench` and
`scripts/jetson/bench-reset.sh` (never deletes `~/.venv-krabby`).

```bash
# From cold-start or any venv that has this package (editable ok):
cd /path/to/krabby-research
source ~/.venv-krabby/bin/activate   # or bench venv after first install
pip install -e ./bench

# Full run (reset + fresh Install + Flash + Bringup + Motion).
# Requires sudo for krabby install; keep one leg free to move.
krabby-bench harness --repo-root "$(pwd)"

# After Install already proved — skip wipe/reinstall:
krabby-bench harness --skip-install --no-reset --repo-root "$(pwd)"
```

| Stage | What it does |
|---|---|
| **install** | `bench-reset.sh` → create `~/.venv-krabby-bench` → `pip install krabby-launcher` (+ local `./bench`) → `sudo … krabby install --no-launch-on-startup` |
| **flash** | Existing `run_smoke`: flash all boards, assert versions match S3 |
| **bringup** | `krabby run --gamepad-only`; wait for container + MCU connected in logs |
| **motion** | Stop container; jog RLKL, RRKL, FLKL, FRHL, FRKL (override with `--joint …`) via **host** repo `firmware/` (PIN_REV=2 is local-testing only / `dev-local` — not S3; will not be used going forward); assert pot or hall delta per joint |

Motion stops the stack temporarily because Jetson HAL observations do not carry
real pot/hall (KNOWN-ISSUES #6/#7). Failure output names the stage and the
command that failed.

Useful flags: `--skip-flash`, `--skip-bringup`, `--skip-motion`, `--firmware-channel`,
`--joint`, `--rmi` (pass through to reset), `--no-discord`, `--run-url`, `--commit`.

Discord: set `DISCORD_WEBHOOK_URL` to post PASS/FAIL after the harness (skipped if unset).
In GitHub Actions the same secret name is used.

### Flash note (PIN_REV=2 / local only)

**`PIN_REV=2` is for local Uno v0.1 testing only and will not be used going
forward.** Those boards report `dev-local`, so the harness **does not take the
published S3 firmware** (wrong pin map) and **skips S3 overwrite**. Future CI
should use the published pin revision / S3 artifact path.

## CI: artifact health (hosted runners)

Workflow: [`.github/workflows/artifact-health.yml`](../.github/workflows/artifact-health.yml)

Runs on push to `mainline` / `release/**` (and `workflow_dispatch`). No Orin required.

| Job | Check |
|---|---|
| packages | `pip install` + import for `krabby-launcher`, `krabby-firmware`, `krabby-bench` |
| locomotion-image | pull `mainline-latest` + `release-latest`, `docker run … --help` (arm64 via QEMU) |
| firmware-artifact | S3 `latest.json` + manifest `ver_string` + HEX HEAD for `release/0.2.15` |
| notify | Discord summary (skipped if `DISCORD_WEBHOOK_URL` secret unset) |

## CI: self-hosted Orin runner (deferred)

Workflow: [`.github/workflows/bench-harness.yml`](../.github/workflows/bench-harness.yml)

The hardware harness job is **gated** on repo variable `BENCH_RUNNER_ENABLED=true`
so it does not queue forever before a runner exists. Until then a stub job
explains the skip.

When you have repo admin permission:

1. On the Orin, install the GitHub Actions runner for **linux-arm64** from
   Settings → Actions → Runners → New self-hosted runner.
2. Configure with labels: `self-hosted`, `krabby-bench` (default `self-hosted`
   is fine; add `krabby-bench` explicitly).
3. Install as a service; ensure the runner user is in `dialout`, can use Docker,
   and can `sudo` the same `krabby install` path the harness uses
   (`sudo -E env PATH=…`).
4. Repo **Actions variable**: `BENCH_RUNNER_ENABLED` = `true`.
5. Repo **secret** (optional): `DISCORD_WEBHOOK_URL` = Discord channel webhook.
6. Keep dual-venv reset: harness uses `scripts/jetson/bench-reset.sh` and must
   **never** delete `~/.venv-krabby`.

Until the runner is registered, exercise hardware locally:

```bash
krabby-bench harness --skip-install --no-reset --repo-root "$(pwd)"
# optional: export DISCORD_WEBHOOK_URL=… to prove notifications
```

## Local development

Set `BENCH_SMTP_*` and `BENCH_GITHUB_TOKEN` env vars directly; the watchdog reads them as fallback when no SSM prefix is configured:

```bash
BENCH_SMTP_HOST=smtp.example.com ... python -m krabby_bench.watchdog
```

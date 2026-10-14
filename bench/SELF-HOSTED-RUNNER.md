# Self-hosted Orin runner (bench harness)

Reusable setup for any Jetson Orin used as the **GitHub Actions** target for
[`.github/workflows/bench-harness.yml`](../.github/workflows/bench-harness.yml).

Also indexed from [`README.md`](README.md) (§ CI: self-hosted Orin runner).

## Requirements

| Need | Notes |
|------|--------|
| Machine | linux **aarch64** Orin (`uname -m` → `aarch64`) |
| Repo access | Someone with **admin** on the GitHub repo must mint the registration token |
| Labels | `self-hosted`, `krabby-bench` (workflow `runs-on`) |
| Repo variable | `BENCH_RUNNER_ENABLED=true` (otherwise harness job is stubbed) |
| Repo secret | `DISCORD_WEBHOOK_URL` (optional; harness/CI notify) |
| Host perms | Runner user in `docker` + `dialout`; can `sudo -E env PATH="$PATH" "$(which krabby)" …` |

## Admin: mint a registration token

1. Open **Settings → Actions → Runners → New self-hosted runner**.
2. OS **Linux**, architecture **ARM64** (not x64).
3. Copy the **token** from that page (and the download URL if useful).
4. Hand the token to the Orin operator **immediately**.

**Token lifetime:** ~**1 hour**. Expired or reused tokens often fail as
`404` on `POST …/actions/runner-registration`. Do **not** paste tokens into
chat, tickets, or commit logs.

## Operator: install on the Orin

Do **not** unpack under a git working tree (e.g. not inside `krabby-research/`).
Use a home directory path such as `~/actions-runner`.

**Do not check in `actions-runner`.** It is machine-local only (binaries, config, and
credentials). Never add it to the repo, commit it, or leave it under a tracked tree.

```bash
uname -m   # must be aarch64

mkdir -p ~/actions-runner && cd ~/actions-runner

# Prefer the exact curl + sha256 lines from the New runner page (ARM64).
# Example version — bump to match the page:
curl -o actions-runner-linux-arm64-2.337.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-arm64-2.337.0.tar.gz
# echo "<sha256 from page>  actions-runner-linux-arm64-2.337.0.tar.gz" | shasum -a 256 -c

tar xzf ./actions-runner-linux-arm64-*.tar.gz

./config.sh --url https://github.com/<org>/<repo> \
  --token <PASTE_FRESH_TOKEN> \
  --name <orin-hostname>-bench \
  --labels self-hosted,krabby-bench

# Only after config succeeds (creates svc.sh):
sudo ./svc.sh install
sudo ./svc.sh start
./svc.sh status
```

Confirm the runner shows **Idle** under Settings → Actions → Runners.

### Common failures

| Symptom | Cause | Fix |
|---------|--------|-----|
| `cannot execute binary file: Exec format error` | Downloaded **linux-x64** on Orin | Delete tree; install **linux-arm64** |
| `404` on `runner-registration` | Token expired (~1h), reused, or wrong repo | Admin mints a **new** token; re-run `config.sh` promptly |
| `sudo: ./svc.sh: command not found` | `config.sh` never succeeded | Fix registration first; `svc.sh` appears after a good config |

## Enable the harness workflow

1. Repo **Settings → Secrets and variables → Actions → Variables**  
   `BENCH_RUNNER_ENABLED` = `true`
2. Secret `DISCORD_WEBHOOK_URL` if not already set (same name local harness uses).
3. Trigger: push to `main` / `release/**`, or **Actions → Bench harness → Run workflow**.

Until the variable is `true`, [`bench-harness.yml`](../.github/workflows/bench-harness.yml)
runs only the stub job (stays green, no Orin queue).

## Dual-use Orin (cold-start + bench)

If this machine is also used for manual bring-up:

- Use two venvs: `~/.venv-krabby` (bring-up, durable) and `~/.venv-krabby-bench` (CI install stage).
- Harness / `scripts/jetson/bench-reset.sh` may wipe **`~/.venv-krabby-bench`** only.
- **Never** delete `~/.venv-krabby` from CI reset.
- Prefer runner **offline** (or no overlapping jobs) while doing interactive bring-up.

## Manual harness until runner is live

```bash
cd /path/to/krabby-research
source ~/.venv-krabby/bin/activate   # or ~/.venv-krabby-bench after Install stage
pip install -e ./bench               # if needed
export DISCORD_WEBHOOK_URL=…         # optional local notify

# Full: reset + install + flash + bringup + motion
krabby-bench harness --repo-root "$(pwd)"

# Continue after a proven Install (no wipe / no reinstall):
krabby-bench harness --skip-install --no-reset --repo-root "$(pwd)"

# Motion only (boards already flashed / stack already exercised):
krabby-bench harness --skip-install --no-reset --skip-flash --skip-bringup \
  --joint RRKL FLKL FRHL FRKL --repo-root "$(pwd)"
```

| Flag | Effect |
|------|--------|
| `--skip-install` | Skip reset venv + `pip install` + `krabby install` |
| `--no-reset` | Do not run `bench-reset.sh` before Install |
| `--skip-flash` | Skip MCU flash / version check |
| `--skip-bringup` | Skip `krabby run` bring-up wait |
| `--skip-motion` | Skip jog / pot-hall assert |
| `--joint …` | Override default joints for motion |
| `--rmi` | Pass `--rmi` through to `bench-reset.sh` |
| `--no-discord` | Skip Discord even if `DISCORD_WEBHOOK_URL` is set |
| `--firmware-channel` | S3 channel for flash (when not skipped) |
| `--run-url` / `--commit` | Optional Discord metadata |

More detail: [`README.md`](README.md) (§ Four-stage harness).

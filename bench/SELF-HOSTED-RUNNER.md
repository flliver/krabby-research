# Self-hosted Orin runner (bench harness)

How to turn a **new** Jetson Orin into the GitHub Actions target for
[`.github/workflows/bench-harness.yml`](../.github/workflows/bench-harness.yml).

Also indexed from [`README.md`](README.md) (§ CI: self-hosted Orin runner).

> **Validated** on a test Orin named **orin1**
> ([E2E PASS 39980867146](https://github.com/flliver/krabby-research/actions/runs/39980867146)).
> A **dedicated** bench Orin is still needed for ongoing CI; follow this guide
> on that machine (its own hostname and `$(hostname)-bench` runner).

Two phases: **(A)** register the GitHub runner, **(B)** enable the workflow and
prove install → flash → bringup → motion on this machine.

---

## Phase A — Register the Actions runner

### A1. Machine prerequisites

| Need | Check |
|------|--------|
| linux **aarch64** | `uname -m` → `aarch64` |
| Network + GitHub reachability | Can `curl -I https://github.com` |
| User in `docker` and `dialout` | `groups` lists both; `docker ps` works without sudo tricks |
| Can sudo with venv PATH | Later: `sudo -E env PATH="$PATH" "$(which krabby)" …` |

### A2. Admin: mint a registration token (~1 hour life)

1. Repo **Settings → Actions → Runners → New self-hosted runner**.
2. OS **Linux**, architecture **ARM64** (not x64).
3. Copy the **token** (and prefer the page’s exact download / sha256 lines).
4. Give it to the Orin operator **immediately**.

Do **not** paste tokens into chat, tickets, or commits. Expired/reused tokens
usually fail as `404` on `runner-registration`.

### A3. Operator: install under `~/actions-runner`

Do **not** unpack inside a git tree (e.g. not under `krabby-research/`).
**Never** commit `actions-runner` (binaries + credentials).

```bash
uname -m   # must be aarch64

mkdir -p ~/actions-runner && cd ~/actions-runner

# Prefer the exact curl + sha256 from the New runner page (ARM64).
# Example only — bump version to match the page:
curl -o actions-runner-linux-arm64-2.337.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-linux-arm64-2.337.0.tar.gz
# echo "<sha256 from page>  actions-runner-linux-arm64-2.337.0.tar.gz" | shasum -a 256 -c

tar xzf ./actions-runner-linux-arm64-*.tar.gz

# REAL URL only — never paste <org>/<repo> (bash treats <org> as a redirect).
# Quote the token. Use this machine's hostname in the runner name.
./config.sh --url https://github.com/flliver/krabby-research \
  --token '<PASTE_FRESH_TOKEN>' \
  --name "$(hostname)-bench" \
  --labels self-hosted,krabby-bench
```

`config.sh` may prompt interactively:

- Runner group → **Enter** (Default)
- Work folder → **Enter** (`_work`)

Success looks like: **Connected to GitHub**, **Runner successfully added**,
**Settings Saved.** Only then:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status    # needs sudo; expect Active: active (running)
```

### A4. Confirm registration

| Where | Expect |
|-------|--------|
| GitHub → Settings → Actions → Runners | Runner **`$(hostname)-bench`**, status **Idle** |
| On Orin | `journalctl -u 'actions.runner.*.service' -n 20 --no-pager` shows **Listening for Jobs** (runner may self-update once; that is normal) |

**Phase A done** when the runner is Idle/Listening. The token is no longer needed.

### How the runner runs on the Orin

`sudo ./svc.sh install` creates a **systemd** unit (name like
`actions.runner.<org>-<repo>.<hostname>-bench.service`), **enabled** for
`multi-user.target`. On every boot, systemd starts it after the network is up:

`systemd` → `~/actions-runner/runsvc.sh` → Runner listener → **Listening for Jobs**.

It does **not** watch your git remotes. Flow when someone pushes to `main` /
`release/**` (or uses **Run workflow**):

1. GitHub creates the Bench harness run.
2. If `BENCH_RUNNER_ENABLED=true`, GitHub looks for an online runner with labels
   `self-hosted`,`krabby-bench`.
3. This Orin’s listener is already connected; GitHub **assigns** the job.
4. The runner checks out the repo and runs install → flash → bringup → motion.

If the Orin is **powered off** or the service is stopped, the harness job
**queues** (does not fail immediately) until a matching runner comes online or
someone cancels the run. The job’s 90‑minute timeout applies only **after** a
runner has started it.

Control locally:

```bash
cd ~/actions-runner
sudo ./svc.sh status   # Active: active (running)?
sudo ./svc.sh stop     # offline for GitHub until start again
sudo ./svc.sh start
journalctl -u 'actions.runner.*.service' -f
```

### Phase A failures

| Symptom | Cause | Fix |
|---------|--------|-----|
| `Exec format error` | Downloaded **linux-x64** | Wipe tree; install **linux-arm64** |
| `404` on registration | Token expired/reused/wrong repo | New admin token; re-run `config.sh` promptly |
| `bash: org: No such file or directory` | Literal `<org>` in URL | Real `https://github.com/flliver/krabby-research` |
| `svc.sh: command not found` | `config.sh` never succeeded | Fix registration; `svc.sh` appears only after success |
| `Must run as sudo` on status | Forgot sudo | `sudo ./svc.sh status` |

---

## Phase B — Enable harness and validate end-to-end

### B1. Hardware for a real PASS

Before expecting motion green:

| Need | Why |
|------|-----|
| MCUs attached / powered; roles visible | Flash + bringup need boards (`krabby firmware show`) |
| **≥1 actuator** free to move | Motion asserts pot/hall change |
| **12 V / EN** on | Without EN, motion FAILS with `en=[0,0]` and pot Δ≈0 |
| Prefer no overlapping manual `krabby run` | CI owns the `/krabby` container name |

**Firmware:** the harness normally reflashes from the **published** artifact and
checks versions. Some lab boards already carry local firmware and skip that
overwrite — expect published flash on the dedicated bench Orin.

### B2. Warm the locomotion image (strongly recommended)

Cold `docker pull` of `release-latest` is ~**19 GB** and can eat most of the
install stage’s **15 min** timeout. Warm once as the runner user:

```bash
docker pull public.ecr.aws/t7t7b3i3/krabby-locomotion:release-latest
```

If a prior CI install **timed out**, `docker pull` / `krabby install` may still
be running as orphans — wait until they exit (`pgrep -af 'docker pull|krabby install'`)
before re-running the workflow, or the next install races the same pull.

### B3. Admin: turn on the workflow

1. **Settings → Secrets and variables → Actions → Variables**  
   `BENCH_RUNNER_ENABLED` = `true` (exact string; this is a **variable**, not a secret).
2. **Secrets** → `DISCORD_WEBHOOK_URL` if you want PASS/FAIL notify (recommended).

The workflow always defines **two** jobs in
[`.github/workflows/bench-harness.yml`](../.github/workflows/bench-harness.yml):

| Job name in the UI | Workflow key | Runs when |
|--------------------|--------------|-----------|
| **install → flash → bringup → motion** | `harness` | `BENCH_RUNNER_ENABLED == 'true'` → real Orin |
| **Runner not enabled (stub)** | `stub-status` | `BENCH_RUNNER_ENABLED != 'true'` → hosted ubuntu, no Orin |

Defined as jobs `harness` and `stub-status` (stub starts around line 51). The stub
exists so pushes stay green before any Orin is registered — it only prints that
the variable is off.

**How to read a run:**

- Variable **on:** harness runs on your Orin; stub shows **Skipped**. That skipped
  row still appears in the Actions UI with the name “Runner not enabled (stub)” —
  that is expected, not a failure.
- Variable **off:** stub is **Success** (green) and harness is **Skipped** — nothing
  ran on the Orin.

### B4. Smoke: Run workflow

1. **Actions → Bench harness → Run workflow** (branch `main`, or push to `main` /
   `release/**`).
2. Confirm **install → flash → bringup → motion** ran on **`$(hostname)-bench`**
   with conclusion **success** (or a real stage fail). The stub should be
   **Skipped**, not the only green job.
3. On the Orin (optional live watch):

   ```bash
   journalctl -u 'actions.runner.*.service' -f
   # Expect: Running job: install → flash → bringup → motion
   ```

4. **Success:** Actions green; Discord (if set) PASS; summary shows all four
   stages PASS. Typical warm-image run is on the order of a few minutes.
5. **Failure:** Discord/Actions should name the real stage (`install` / `flash` /
   `bringup` / `motion`) with a detail line — not a collapsed
   `flash+bringup+motion: child exit 1`.

### Phase B failures

| Symptom | Cause | Fix |
|---------|--------|-----|
| Stub **Success**, harness **Skipped** | `BENCH_RUNNER_ENABLED` not `true` | Set variable; re-run |
| Stub **Skipped**, harness green/red | Variable on — stub name is cosmetic | Ignore skipped stub; read harness job |
| Job queued forever / wrong host | Labels wrong or old runner still online | Labels must be `self-hosted`,`krabby-bench`; offline unused runners |
| Install timeout / stuck on `docker pull` | Cold ~19 GB pull | Warm image (B2); install timeout is 15 min; wait for orphaned pull |
| Motion `No module named 'serial'` | Stale bench without `pyserial` | Current `krabby-bench` declares `pyserial`; ensure CI checked out that commit / reinstall bench into venv |
| Motion `en=[0,0]` / pot Δ≈0 | No 12 V or EN | Power/EN the H-bridges; re-run |
| Discord opaque `child exit 1` | Old harness before summary parse | Use current `main` harness (child `--no-discord` + parent parses summary) |

---

## What the CI harness does

Checkout → temp venv `pip install -e ./bench` →
`krabby-bench harness --repo-root $GITHUB_WORKSPACE …`.

| Stage | Notes |
|-------|--------|
| **install** | `bench-reset.sh` → recreate `~/.venv-krabby-bench` → `pip install krabby-launcher` + `./bench` (pulls **pyserial**) → `krabby install --no-launch-on-startup`. Timeout **15 min**. |
| **flash** | Reflash from the published artifact and check versions (lab boards with local firmware may skip overwrite). |
| **bringup** | `krabby run --gamepad-only`; wait for MCU / gamepad-ready. |
| **motion** | Stop container; host jog via repo `firmware/`; assert pot/hall. Needs **12 V / EN**. |

After install, the harness **re-execs** under `~/.venv-krabby-bench` for
flash/bringup/motion. Parent parses the child summary for Discord/Actions
(child uses `--no-discord`).

---

## Dual-use Orin (cold-start + bench)

If this machine is also used for manual bring-up:

- `~/.venv-krabby` — durable bring-up (CI must **never** delete it).
- `~/.venv-krabby-bench` — CI install stage only (`bench-reset.sh` may wipe it).
- Prefer runner **offline** (or no overlapping jobs) during interactive bring-up.

---

## Manual harness (debug on the Orin)

```bash
cd /path/to/krabby-research
source ~/.venv-krabby/bin/activate   # or ~/.venv-krabby-bench after Install
pip install -e ./bench
export DISCORD_WEBHOOK_URL=…         # optional

krabby-bench harness --repo-root "$(pwd)"

# After a proven Install:
krabby-bench harness --skip-install --no-reset --repo-root "$(pwd)"

# Motion only:
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
| `--no-discord` | Skip Discord even if webhook is set |
| `--firmware-channel` | S3 channel for flash (when not skipped) |
| `--run-url` / `--commit` | Optional Discord metadata |

More detail: [`README.md`](README.md) (§ Four-stage harness).

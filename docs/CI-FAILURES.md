# CI failures — what to do

Short playbook for engineers when a GitHub Actions run goes red (or Discord reports FAIL).

**Actions:** https://github.com/flliver/krabby-research/actions

Discord FAIL messages link here as **CI failures playbook**. Use that link, then the **View run** link in the same message.

## What runs on a push to `main` (or `release/*`)

| Workflow | When | What it means if red |
| --- | --- | --- |
| **Unit tests** (`.github/workflows/test-unit.yml`) | Push / PR | Full Python suite under `tests/unit/` failed (wheels installed first so imports match a real install). Does not publish artifacts. |
| **Firmware tests** (`.github/workflows/test-firmware.yml`) | Push / PR | Native + Python firmware tests failed. Does not upload HEX (publish-firmware still runs its own test step before S3). |
| **Publish firmware to S3** | Push (+ daily schedule on newest `release/*`) | Firmware tests failed or HEX was not uploaded. Dev channel is S3 `main/`; stable is `release/*`. |
| **Build and push locomotion image to ECR** | Push (+ daily schedule) | Image was not published. Dev tag `main-latest`; stable `release-latest`. |
| **Artifact health** | **After** locomotion publish **succeeds** (`workflow_run`) | Pull/smoke of published artifacts failed (packages, image, or firmware S3). |
| **Bench harness** | Push (Orin runner) | Often skipped until a self-hosted runner is enabled — see `bench/README.md`. When enabled: install → flash → bringup → motion failed. |
| **Fleet CI** / **Fleet Deploy** | Push / PR to `main` | Fleet/portal path; separate from kit firmware/image publish. No Discord. |

### Also (not ordinary branch pushes)

| Workflow | When | What it means if red |
| --- | --- | --- |
| **Publish packages to PyPI** (`.github/workflows/publish-packages.yml`) | **Tag only** (`*-v*`) | Package tests or `twine` upload failed. See [PUBLISHING.md](PUBLISHING.md). |

## Discord

When `DISCORD_WEBHOOK_URL` is set, CI notify uses **FAIL always** and **PASS only on recovery** (`collapse_consecutive_pass`) — consecutive greens do not spam Discord:

| Source | Discord title / note |
| --- | --- |
| **Unit tests** | `unit-tests` |
| **Firmware tests** | `firmware-tests` |
| **Publish firmware** | `publish-firmware` |
| **Publish locomotion** | `publish-locomotion` (Artifact health does not start until this is green) |
| **Publish packages** | `publish-packages` |
| **Artifact health** | `artifact-health` (per-stage results in the message) |
| **Bench harness** | via `krabby-bench` when the Orin runner is enabled; stub skip posts nothing |

Typical FAIL fields:

| Field | Meaning |
| --- | --- |
| Commit | Short SHA + first line of the commit message |
| Branch / event | e.g. `main` `(push)` |
| Failed stage | Workflow job name (or Artifact health stage) |
| Failed steps | `job › step` when the Actions API can resolve it |
| View run | Open the red Actions run / log |
| CI failures playbook | This document |
| Hint / Jobs | Short cause hint, or per-stage results for Artifact health |

Open the Discord link (or Actions) and check which **stage** failed: for Artifact health that is `packages`, `locomotion-image`, or `firmware-artifact`.

## Steps when something fails

1. Open **View run** from Discord (or [Actions](https://github.com/flliver/krabby-research/actions)) for the failed commit.
2. Open the red job → failed step (match **Failed steps** in Discord when present) → read the log (not only the annotation summary).
3. Fix forward on a new commit to `main` (or the relevant `release/*` branch / PR / tag). Prefer fixing the cause over blind re-runs.
4. **Re-run** a failed job only for flakes (network, runner blip, registry rate limit). Re-run does not help if tests or the build are wrong.
5. Confirm the follow-up:
   - Firmware: S3 `https://krabby-firmware-public.s3.amazonaws.com/<channel>/latest.json` (e.g. channel `main`).
   - Image: `docker pull public.ecr.aws/t7t7b3i3/krabby-locomotion:main-latest` (or `release-latest`).
   - Artifact health: should start automatically after the next **green** locomotion publish; or **Actions → Artifact health → Run workflow**.
   - PyPI: package page / `pip index versions <pkg>` for the tag you published.

## Common cases

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Discord `FAIL: unit-tests` | A test under `tests/unit/` failed, or wheel build / install before pytest failed | Open **View run** → job `test`; fix the failing test or package install; push / update the PR. Workflow: `.github/workflows/test-unit.yml`. |
| Discord `FAIL: firmware-tests` | Native C++ firmware tests, coverage gate, or Python firmware tests failed | Open **View run** → job `test`; fix firmware tests. Workflow: `.github/workflows/test-firmware.yml`. (Publish-firmware runs a similar suite before S3 upload.) |
| Discord `FAIL: publish-firmware` | Firmware tests, compile, or S3 upload failed | Open **View run**; fix the named step; push again. Scheduled daily builds use `release/*`, so they can stay green while `main` is red. |
| Discord `FAIL: publish-locomotion` | Docker build/push, OIDC, or smoke (`--teleop-ip`) failed | Fix Dockerfile/pins or AWS role; push again. Artifact health will not run until this is green. |
| Discord `FAIL: publish-packages` | Package tests or PyPI/`twine` upload failed for a `*-v*` tag | Open **View run**; fix tests or trusted-publisher / version; retag or push a new tag per [PUBLISHING.md](PUBLISHING.md). Workflow: `.github/workflows/publish-packages.yml`. |
| Discord `FAIL: artifact-health` with `locomotion-image` and hint mentions **rate limit** / `toomanyrequests` | ECR Public (or Docker) pull quota exhausted despite OIDC login — rare shared-runner flake | **Re-run** the Artifact health job (Actions → Re-run failed jobs). Not an image bug. |
| Discord `FAIL: artifact-health` with `locomotion-image` (other) | ECR pull or `--help` / `--teleop-ip` smoke failed | Confirm the tag exists in ECR; inspect `--help` in the log; fix image or smoke expectation. |
| Discord `FAIL: artifact-health` with `packages` | `pip install` / import failed for a cold-start package | Check PyPI / package version on that channel; fix packaging or the smoke import. |
| Discord `FAIL: artifact-health` with `firmware-artifact` | S3 `latest.json` / manifest / HEX HEAD failed | Confirm the channel artifact on S3; fix firmware publish or the channel the check uses. |
| Artifact health never started after your push | Locomotion publish still running or failed | Wait for publish, or fix publish first (expect Discord from publish on failure). |
| Discord FAIL from **bench harness** (Orin) | One of install / flash / bringup / motion stages failed | Open **View run** on the self-hosted job; see `bench/README.md`. If only the stub job ran, the Orin runner is not enabled (`BENCH_RUNNER_ENABLED`). |

## Related docs

- [bench/README.md](../bench/README.md) — Artifact health trigger and jobs; Orin bench harness.
- [images/locomotion/README.md](../images/locomotion/README.md) — ECR tags and publish flow.
- [firmware/SETUP.md](../firmware/SETUP.md) — S3 firmware store layout.
- [PUBLISHING.md](PUBLISHING.md) — PyPI tag publish.

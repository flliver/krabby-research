# CI failures — what to do

Short playbook for engineers when a GitHub Actions run goes red (or Discord reports FAIL).

**Actions:** https://github.com/flliver/krabby-research/actions

## What runs on a push to `main` (or `release/*`)

| Workflow | When | What it means if red |
| --- | --- | --- |
| **Publish firmware to S3** | Push | Firmware tests failed or HEX was not uploaded. Dev channel is S3 `main/`; stable is `release/*`. |
| **Build and push locomotion image to ECR** | Push | Image was not published. Dev tag `main-latest`; stable `release-latest`. |
| **Artifact health** | **After** locomotion publish **succeeds** (`workflow_run`) | Pull/smoke of published artifacts failed (packages, image, or firmware S3). |
| **Bench harness** | Push (Orin runner) | Often skipped until a self-hosted runner is enabled — see `bench/README.md`. |
| **Fleet CI** / **Fleet Deploy** | Push to `main` | Fleet/portal path; separate from kit firmware/image publish. |

PyPI package publish is **tag-only** (`*-v*`) — see [PUBLISHING.md](PUBLISHING.md). It does not run on ordinary branch pushes.

## Discord

- **Artifact health** can post PASS/FAIL to Discord when `DISCORD_WEBHOOK_URL` is set.
- That notify runs only when Artifact health itself runs (after a **successful** locomotion publish, or a manual Artifact health dispatch).
- If **locomotion publish fails**, Artifact health (and its Discord) **do not** run. Publish firmware / locomotion failures are **silent** unless you watch the Actions tab (or email/GitHub from the Orin bench watchdog).

**TODO:** Discord is **not** sent when Artifact health is not triggered (e.g. locomotion publish failed, or never completed). Engineers who rely on Discord alone will miss those failures. Until fixed: always check the Actions tab for publish jobs.

Open the Discord link (or Actions) and check which **stage** failed: `packages`, `locomotion-image`, or `firmware-artifact`.

## Steps when something fails

1. Open [Actions](https://github.com/flliver/krabby-research/actions) and find the failed run for the commit.
2. Open the red job → failed step → read the log (not only the annotation summary).
3. Fix forward on a new commit to `main` (or the relevant `release/*` branch). Prefer fixing the cause over blind re-runs.
4. **Re-run** a failed job only for flakes (network, runner blip). Re-run does not help if tests or the build are wrong.
5. Confirm the follow-up:
   - Firmware: S3 `https://krabby-firmware-public.s3.amazonaws.com/<channel>/latest.json` (e.g. channel `main`).
   - Image: `docker pull public.ecr.aws/t7t7b3i3/krabby-locomotion:main-latest` (or `release-latest`).
   - Artifact health: should start automatically after the next **green** locomotion publish; or **Actions → Artifact health → Run workflow**.

## Common cases

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Publish firmware red at “Run native and Python firmware tests” | Unit/native test failure on that branch | Fix the test or code; push again. Scheduled daily builds use `release/*`, so they can stay green while `main` is red. |
| Publish locomotion red | Docker build/push / OIDC / smoke `--teleop-ip` | Fix Dockerfile/pins or AWS role; push again. Artifact health will not run until this is green. |
| Artifact health `locomotion-image` red | Pull or `--help` / `--teleop-ip` check failed | Confirm the tag exists in ECR; inspect `--help` in the log; fix image or smoke expectation. |
| Artifact health never started after your push | Locomotion publish still running or failed | Wait for publish, or fix publish first. |
| Discord FAIL but you expected a notify on publish failure | Discord only runs inside Artifact health; publish failures do not trigger it | Watch Actions until publish notify is added. |

## Related docs

- [bench/README.md](../bench/README.md) — Artifact health trigger and jobs; Orin bench harness.
- [images/locomotion/README.md](../images/locomotion/README.md) — ECR tags and publish flow.
- [firmware/SETUP.md](../firmware/SETUP.md) — S3 firmware store layout.
- [PUBLISHING.md](PUBLISHING.md) — PyPI tag publish.

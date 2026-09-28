# Publishing to PyPI

This document is a short checklist for publishing **krabby‑*** packages to PyPI using the GitHub Actions workflow. The workflow runs when you push a version tag; it builds the package, runs that package’s tests, then uploads the wheel to PyPI.

**Workflow:** [.github/workflows/publish-packages.yml](../.github/workflows/publish-packages.yml)

## Tag patterns and package names

| Tag pattern | PyPI package |
| :-- | :-- |
| `krabby-v*` | krabby-launcher |
| `firmware-v*` | krabby-firmware |
| `hal-client-v*` | krabby-hal-client |
| `hal-server-v*` | krabby-hal-server |
| `compute-parkour-v*` | krabby-compute-parkour |
| `controller-v*` | krabby-controller |
| `hal-tools-v*` | krabby-hal-tools |
| `hal-server-isaac-v*` | krabby-hal-server-isaac |
| `hal-server-jetson-v*` | krabby-hal-server-jetson |
| `teleop-edge-v*` | krabby-teleop-edge |
| `bench-v*` | krabby-bench |

> **Note:** More specific patterns override generic ones. For example, `hal-server-isaac-v0.1.0` → `krabby-hal-server-isaac`, not `krabby-hal-server`.

## Account

- Create an account at [pypi.org](https://pypi.org) if you do not have one.

## Token

- In PyPI: Account → API tokens.
- Create a token; scope it to the project(s) you publish or to the whole account.
- Keep the token secret; do not commit it.

## GitHub secrets

- Repo → Settings → Secrets and variables → Actions.
- Add a secret: `PYPI_API_TOKEN` with the PyPI API token value.
- The workflow uses `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=${{ secrets.PYPI_API_TOKEN }}`.

## Trusted Publishing (optional)

- Link the GitHub repo to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
- After one-time setup, the workflow can upload without a long-lived token.

## Publishing

- Push a version tag to trigger the workflow, e.g.:
  - `git tag krabby-v0.1.12 && git push origin krabby-v0.1.12`
- **The version comes entirely from the tag.** CI writes it into the package's
  `pyproject.toml` before building, and `krabby --version` reads it from the installed
  package metadata — you do **not** hand-edit any version in source to cut a release.
  (Bumping `pyproject` in source is optional hygiene so the repo matches the latest tag.)
- CI builds the package, runs its tests, then uploads to PyPI.
- **Publish in dependency order** so dependents can install from PyPI:
  1. `hal-client-v*`, `hal-server-v*`, `firmware-v*`, `krabby-v*`, `bench-v*` (no internal deps)
  2. `compute-parkour-v*`, `controller-v*`, `hal-tools-v*`
  3. `hal-server-isaac-v*`, `hal-server-jetson-v*`, `teleop-edge-v*` (required in the locomotion image for fleet `--teleop-ip`)

### Before you push a tag (docs and pins)

CI takes the **version from the tag**, but several **human-facing** files still need to match that version **before** you tag and push. PyPI renders each package’s **`readme`** from source at upload time (for `krabby-launcher`, that is [`krabby/README.md`](../krabby/README.md)); stale `pip install '…>=0.1.N'` lines there show up on the project page even when the wheel metadata is newer.

**Every `krabby-v*` release (`krabby-launcher`):**

1. Set `version` in [`krabby/pyproject.toml`](../krabby/pyproject.toml) to the tag version (repo hygiene; CI also overwrites it during the job).
2. Update the Install example in [`krabby/README.md`](../krabby/README.md) to `pip install 'krabby-launcher>=<tag-version>'`.
3. Update fleet operator docs that copy the same pin, e.g. [`fleet/ENROLL.md`](../fleet/ENROLL.md), [`fleet/SETUP-FLEET.md`](../fleet/SETUP-FLEET.md), [`fleet/BENCH-TELEOP.md`](../fleet/BENCH-TELEOP.md) (minimum version and install blocks).
4. If the release changes HAL CLI or fleet teleop behavior, plan the **locomotion image** separately: publish [`krabby-hal-server-jetson`](../hal/server/jetson/) if needed, bump [`images/locomotion/requirements.release.txt`](../images/locomotion/requirements.release.txt), and push a tracked branch so ECR `release-latest` updates — see [`images/locomotion/README.md`](../images/locomotion/README.md). Launcher and locomotion image releases are not the same tag.

**Other PyPI packages:** bump `version` in that package’s `pyproject.toml` and any README install line that names a specific version.

**Quick check** from repo root (replace `0.1.21` with the version you are about to ship):

```bash
rg "krabby-launcher>=0\.1\.(20|21)|≥ 0\.1\.(20|21)" krabby fleet docs
```

Fix or intentionally keep any hit before pushing the tag.

## Locomotion image (`release-latest` on ECR)

Production robots pull **`public.ecr.aws/t7t7b3i3/krabby-locomotion:release-latest`**, built from [`images/locomotion/Dockerfile.release`](../images/locomotion/Dockerfile.release) when you push a **`release/*`** branch (see [`images/locomotion/README.md`](../images/locomotion/README.md)). **`main` pushes do not move `release-latest`.**

**Do not reuse or fast-forward an old `release/x.y.z` branch** for a new field release. Each locomotion refresh gets a **new branch** cut from current **`main`** (e.g. `release/0.2.12`, then later `release/0.2.13`).

**Order (every locomotion release):**

1. On **`main`**, set the full pin bundle in [`images/locomotion/requirements.release.txt`](../images/locomotion/requirements.release.txt) (all `krabby-*==…` lines must be compatible — pre-push smoke imports HAL).
2. **Publish PyPI packages first** in dependency order (e.g. `firmware-v*` before rebuilding the image if HAL imports firmware). Wait until the **Publish packages to PyPI** workflow is green and `pip index versions <package>` lists each pin.
3. Run locally: build full [`Dockerfile.release`](../images/locomotion/Dockerfile.release) with `--load`, then `docker run --rm --network host … --help | grep teleop-ip` (same as CI pre-push gate).
4. **Cut a new release branch** from **`main`**: `git checkout -b release/0.2.N main` (pick the next patch version **N**; use a **new N** for each attempt).
5. `git push -u origin release/0.2.N` — locomotion CI builds the **full** release image once for pre-push smoke, then pushes to ECR (GHA cache should reuse layers); **`release-latest`** moves only if this branch is the **highest** `release/*` version.
6. After CI is green, on kits: `krabby update --image release-latest` and restart `krabby-locomotion`.

Leave older `release/*` branches on the remote as history; do not merge **`main`** into them for the next field push.

## Testing locally (same as CI)

To run the same build-and-test steps as the publish workflow locally (no tag or PyPI):

1. **One-time setup:** From the repo root, create a venv, activate it, and install build and test deps:
   ```bash
   python3 -m venv testenv && source testenv/bin/activate   # Windows: testenv\Scripts\activate
   pip install --upgrade pip build pytest pytest-cov keyboard pyserial torch scipy
   ```

2. **Test a single package** (e.g. before pushing a tag):
   ```bash
   ./scripts/test-publish-job.sh <package-key>
   ```
   Or via Make (with venv active or `testenv` present): `make test-publish-job PKG=<package-key>`.

   `<package-key>` is one of: `hal-client`, `hal-server`, `compute-parkour`, `controller`, `hal-tools`, `hal-server-isaac`, `hal-server-jetson`, `firmware`. (`krabby-launcher` and `krabby-bench` aren't wired into this helper yet — run their tests directly, e.g. `pytest tests/unit/krabby/` or `pytest tests/unit/bench/`.)

3. **Test all eight packages:**
   ```bash
   ./scripts/test-publish-job.sh all
   ```
   Or: `make test-publish-job PKG=all`.

This mirrors the workflow’s build and test steps only; it does not upload to PyPI.

## First-time: reserve package names

- Reserve each name on PyPI so no one else can use it.
- Either create a minimal release (e.g. push a tag and let CI publish) or use the PyPI web UI to create the project.
- Package names to reserve: `krabby-launcher`, `krabby-firmware`, `krabby-hal-client`, `krabby-hal-server`, `krabby-compute-parkour`, `krabby-controller`, `krabby-hal-tools`, `krabby-hal-server-isaac`, `krabby-hal-server-jetson`, `krabby-bench`.

---

## Appendix

### (a) How this works

The workflow runs **only when you push a tag** (not on every branch push). After you push a tag that matches a pattern (e.g. `controller-v0.1.0`):

1. GitHub Actions checks out the repo and parses the tag to pick the package (path, dependency list, test path).
2. It builds and installs any internal krabby-* dependency wheels from the repo (in order).
3. It builds the package wheel, installs it and pytest, then runs that package’s tests.
4. If tests pass, it uploads the wheel to PyPI with `twine` using `PYPI_API_TOKEN`.

More specific tag patterns (e.g. `hal-server-isaac-v*`) are matched before generic ones (e.g. `hal-server-v*`) so the right package is chosen.

### (b) How to create the tag

Create the tag locally, then push it:

```bash
git tag <tag-name>              # e.g. git tag controller-v0.1.0
git push origin <tag-name>     # e.g. git push origin controller-v0.1.0
```

To tag a specific commit: `git tag <tag-name> <commit-hash>`. List tags: `git tag` or `git tag -l 'controller-v*'`.


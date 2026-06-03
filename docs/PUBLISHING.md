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
  3. `hal-server-isaac-v*`, `hal-server-jetson-v*`

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


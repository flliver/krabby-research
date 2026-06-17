# Image-build provenance — EPI-SCN-FLEET-IMAGE-DEPLOY

Durable, git-tracked copies of the artifacts the 2026-06-16 fleet-image
audit surfaced, so nothing depends on a single host's disk. The
authoritative originals live on dbeeprz at
`/home/jeremy/preserve/EPI-SCN-FLEET-IMAGE-DEPLOY/` (persistent nvme,
not tmpfs); these are the small text artifacts copied here for
git-level durability.

## What's here

| File | What |
|------|------|
| `MAtCha-tracked-vs-b119fd9.patch` | dbeeprz `/home/jeremy/sc38/MAtCha` working-tree delta vs upstream `b119fd9` (28 files) |
| `MAtCha-v2-tracked-vs-b119fd9.patch` | `/home/jeremy/sc38/MAtCha-v2` working-tree delta vs `b119fd9` (28 files) |
| `MAtCha-status-porcelain.txt` | `git status --porcelain` of the MAtCha dev tree |
| `tmpfs-rescue.sha256` | sha256 manifest of the rescued `/tmp/{matcha,da3}-build` + `tools/` recipes |

## Key finding (STO-SCN-154 / STO-SCN-155) — these are PROVENANCE, not build inputs

The matcha + da3 images are **already reproducible from committed
source** — the dev-tree patches above are **not** applied at build time:

- The committed `images/matcha/Dockerfile` (sha256 `68be3a8f…`) is
  **byte-identical** to the recipe that built `matcha:0.2.2-selfcontained`
  (the rescued tmpfs Dockerfile), per `tmpfs-rescue.sha256`. Same for
  `images/matcha/requirements.txt` (`356add6b…`) and
  `images/da3/Dockerfile` (`7512f18e…`).
- The image builds MAtCha from **pinned upstream** (`ARG MATCHA_SHA=b119fd9…`,
  `git checkout ${MATCHA_SHA}`) plus the **5 committed** `patch_matcha_*.py`
  scripts — it does **not** COPY the `sc38/MAtCha` dev tree.
- da3 is likewise pinned (`DA3_SHA`, `GSPLAT_SHA`) + committed `krabby-tools/`.

So the `sc38/MAtCha` 28-file delta is the operator's **dev working tree**,
mostly trivial 1-line edits — a deliberate superset of the 5 build patches,
preserved here only so the working tree can't be lost.

## MAtCha vs MAtCha-v2

The two dev trees differ by exactly **2 lines** — a `weights_only=False`
paren-placement bug. **MAtCha-v2 is canonical** (correct placement:
`torch.load(os.path.join(...), weights_only=False)`); the v1 copy
mis-passes `weights_only` into `os.path.join(...)`. The image is
unaffected either way (it uses `patch_matcha_torch_load.py`).

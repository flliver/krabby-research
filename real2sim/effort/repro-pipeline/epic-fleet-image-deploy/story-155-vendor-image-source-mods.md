---
xid: STO-SCN-155
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-16
depends-on: []
bd-id: krabby-edx8
assignee: krabby
shipped: 2026-06-16
tasks: 5
complete: 5
---

# Vendor MAtCha/DA3 source customizations as committed, reproducible patches

## Summary

The krabby modifications baked into `krabby-matcha`/`krabby-da3` are
captured as version-controlled patches (or a krabby fork) so the
images are rebuildable from committed source — not from a dirty
working tree on dbeeprz.

## Context

The audit found, on dbeeprz's persistent `/home/jeremy/sc38/MAtCha`
(+ `MAtCha-v2`):

- remote = `github.com/Anttwo/MAtCha` (**upstream only — no krabby
  fork to push to**); last commit is upstream's `b119fd9` (2025-04-07).
- **28 modified files, all uncommitted** — the krabby customizations
  baked into the images: `2d-gaussian-splatting/train.py`,
  `extract_mesh*.py`, `mast3r/**`, `Depth-Anything-V2/**`,
  `matcha/pointmap/*`.

So the image-defining source changes are version-controlled **nowhere
durable** — they exist only as a dirty working tree + inside the image
layers. The images are not reproducible from committed source today.

## Problem

If that working tree is lost (disk, `git checkout`, host reset), the
exact source that produced matcha/da3 is gone and the images become
black boxes. We need the 28-file delta captured durably and a
documented path to rebuild the image from it.

## Design

### Approach

Decide the durable form (operator call): **(a)** capture as a patch
series under `images/matcha/patches/` + `images/da3/patches/` applied
at build time over a pinned upstream SHA (matches the existing
`patch_*.py` convention), or **(b)** stand up a krabby fork of MAtCha
and pin the image `git clone` to it. (a) is lighter and consistent
with how the Dockerfiles already patch upstream; (b) is cleaner if the
delta is large/structural. Recommend (a) unless the diff resists
clean patch extraction.

### Changes

| File | Change |
|------|--------|
| `images/matcha/patches/` | add extracted patch series for the 28-file MAtCha delta |
| `images/da3/patches/` | same for any DA3-side source delta |
| `images/matcha/Dockerfile`, `images/da3/Dockerfile` | pin upstream SHA + apply the patch series (or clone the fork) |
| `images/*/NOTES.md` | document upstream SHA + patch provenance |

## Definition of Done

- [x] The 28-file MAtCha delta is captured durably (preserved on persistent nvme + git provenance).
- [x] matcha/da3 Dockerfiles pin a known upstream SHA and apply the committed delta.
- [x] A clean rebuild from the committed recipe reproduces a functionally-equivalent image — proof path is the **periodic re-image** half of the dev-loop (the metric-equivalence check happens at each re-image, not a one-off cold build).
- [x] No image-defining source change exists only as an uncommitted working tree.
- [x] **Dev-loop established** (operator reframe): fast bind-mount iteration + periodic re-image, with the `+dev` identity guardrail.

## Out of scope

- The build *recipe* (Dockerfile/patches/tools) tmpfs rescue → STO-SCN-154.
- Re-tagging/pushing rebuilt images to the fleet → handled by 157/159 once reproducibility is established.

## Implementation Notes

### What Changed (2026-06-16) — premise corrected (T-001)

Investigation flipped the audit's framing: **the matcha + da3 images are
already reproducible from committed source** — the "uncommitted 28-file
MAtCha delta" is NOT the build input:

- `images/matcha/Dockerfile` builds MAtCha from **pinned upstream**
  (`ARG MATCHA_SHA=b119fd96…`, `git checkout ${MATCHA_SHA}`; STO-SCN-038)
  plus the **5 committed** `patch_matcha_*.py` scripts. It does **not** COPY
  the `sc38/MAtCha` dev tree. da3 likewise pins `DA3_SHA` + `GSPLAT_SHA` and
  bakes committed `krabby-tools/`. SHA-pinning was already in place — no gap.
- The deployed `matcha:0.2.2-selfcontained` was built from this exact
  committed recipe (Dockerfile sha `68be3a8f…` matches the tmpfs build).
- The `sc38/MAtCha` 28-file delta is dev working-tree scratch (a superset of
  the 5 build patches; mostly trivial 1-line edits). Preserved for durability:
  `dbeeprz:/home/jeremy/preserve/…/matcha-patches/` + git copies under this
  epic's `provenance/`. **MAtCha-v2 is canonical** (a `weights_only=False`
  paren-fix vs v1; image unaffected — it uses `patch_matcha_torch_load.py`).

### Dev-loop established (2026-06-16, operator reframe + greenlight)

Reframed from "vendor source mods" to **"establish the dev-loop"**: fast
iteration + periodic re-image. Three pieces, all now in place:

1. **Fast iter — `KRABBY_DEV_TOOLS=1`** (`v4exec.dev_tools_mount`): stages the
   live `real2sim/` tools to the engine host and **per-file bind-mounts** them
   over the baked `/opt/krabby-tools` in the **fastmap + da3** containers
   (per-file so image-local files like `run_fastmap.sh` survive; matcha is
   self-contained — unaffected). Edit `real2sim/<tool>.py` → rerun in seconds,
   no rebuild.
2. **Guardrail — `+dev` identity salt** (`v4core.identity_hash`): when
   `KRABBY_DEV_TOOLS` is set, the algo is salted `+dev`, so unprovenanced dev
   results land in a separate identity namespace and **can never overwrite a
   canonical store node** (honors STO-SCN-093 D). Verified: same inputs/settings
   → distinct canonical vs dev identities. Stages invoked with explicit
   canonical upstream refs (`covis --solve <id>`) still **reuse** that upstream
   — only the recomputed stage is isolated, so iteration stays fast.
3. **Periodic re-image** (provenanced): `images/fastmap/sync-tools.sh` (the
   STO-SCN-157 guard) → rebuild → push → one-command fan-out pull
   (STO-SCN-158) → bump the `v4exec` image constant. The guard guarantees the
   dev-mounted code == the baked code, so re-imaging is drift-free and IS the
   metric-equivalence proof.

Covered stages: covis (`covis_graph`/`validity_gate`) + da3 (`da3_infer_gs`).
The script-driven solve path (`run_fastmap.sh`) is a future extension.

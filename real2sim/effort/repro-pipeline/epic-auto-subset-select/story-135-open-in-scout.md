---
xid: STO-SCN-135
parent: ./epic.md
kind: story
effort: scn
size: M
status: done
date: 2026-06-15
depends-on: []
bd-id: krabby-taqu
assignee: krabby
---

# Open in Scout: from the Rank UI Manifest, open a variant's mesh + camera spine (utilized subset highlighted) in the scout viewer

## Summary

An **"Open in Scout"** button in the Rank UI **Manifest** area opens the focused variant in a
scout-style 3D viewer that shows three things together: **(1)** the reconstruction's **mesh**,
**(2)** the full **camera spine** (every posed camera) as frustums, and **(3)** the **utilized
subset** of cameras (the N this variant was built from) **highlighted**. Lets the operator
*see*, in 3D, which views produced a given mesh and how they sit in the space.

## Context

The scout verify surface (STO-SCN-095) already renders a gaussian splat + proposed-N frustums
with gravity-up + a level grid (`verify_viewer/build_verify.py` → `viewer.html`). This story
generalizes it to **an arbitrary reconstruction mesh**: instead of the scout splat, load the
variant's `mesh.ply`; show the whole spine; highlight the variant's camera subset (the same set
the Manifest's "Camera Subset" section, STO-SCN-134, lists). Triggered from the runoff so the
operator can inspect any ranked variant in 3D.

## Design

### Approach (reuses verify_viewer — T-013)
A builder resolves, for a variant (scene + mesh identity):
- **mesh** = the variant's `…/meshify/<m>/<id>/mesh.ply` (canonical/oriented gauge).
- **spine** = all posed cameras from the solve's `sparse/0` (`build_frustums` over the full pool).
- **subset** = the variant's camera-subset frames (scout views for da3-scout / subset members
  otherwise — same source as STO-SCN-134) → marked `proposed` so the viewer **highlights** them.
- **gauge** = the variant's `canonical_gauge` `oriented.json` (R, z_shift). The mesh is oriented;
  the frustums come out of `build_frustums` in the **solve** gauge, so **apply the orient xform
  to each frustum** (`_apply_xform`, scale 1) → both land in the oriented gauge (gravity = +z).

Output: a served viewer (`mesh.ply` + `frustums.json` + a viewer html) — `viewer.html` already
has a mesh layer (PLYLoader) + frustum rendering + highlight; drive it in mesh-mode.

### Trigger
- **Rank UI**: an **"Open in Scout"** button in the Manifest panel header. On click → hit a
  studio endpoint `/api/scout/<scene>/<variant>` which builds + serves the viewer and returns
  its URL; the button opens it in a new tab.

### Changes
| File | Change |
|------|--------|
| `real2sim/verify_viewer/build_scout_mesh.py` (new) or extend `build_verify.py` | variant → mesh + spine frustums (orient-applied) + highlighted subset → served viewer |
| `real2sim/studio/server.py` (+ `rate_renders`) | `/api/scout/<scene>/<variant>` → build + serve, return URL |
| `real2sim/rate_renders/static/{index.html, app.js}` | "Open in Scout" button in the Manifest header; open the returned URL |

## Definition of Done

- [x] "Open in Scout" in the Manifest opens a viewer for the focused variant.
- [x] Viewer shows the **mesh** + the **full camera spine** + the **utilized subset highlighted**.
- [x] Mesh and frustums are **gauge-aligned** (orient applied — they overlay correctly).
- [x] Works for DA3-scout (24 highlighted of the spine); meshes above the WebGL per-draw index
      limit (10M tris — matcha-15 tetra/tsdf, large historical) are **gated**: the button is
      disabled with a tooltip naming the limit (a renderer chunk/pre-decimate is deferred follow-up).
- [x] Operator-verified in `/rank` → Scout (T-020) — **DA3-24 opens and looks good** (operator, 2026-06-15).

## Out of scope

- Editing the subset from the viewer (this is inspection; selection editing is STO-SCN-095's surface).
- The gaussian splat layer (this is the mesh view; the splat verify stays STO-SCN-095).

## Implementation Notes

**Built + serving-verified 2026-06-15.**
- **`verify_viewer/build_scout_mesh.py`** (new): variant mesh-id → resolves `mesh.ply` + the
  **solve dir from the rep's `canonical_gauge`** (parent-pool aware, so matcha-15's FINAL-N
  subset resolves to the parent solve) → builds the **full spine** frustums from `sparse/0`,
  carries each through the orient `(R, z_shift)` via `_apply_xform` (→ oriented gauge, `up=[0,0,1]`),
  marks the **utilized subset** (`subset_stems`: scout views for da3-scout / subset members
  otherwise) as `proposed`. Reuses `viewer.html` (mesh layer `scene.ply`, default-visible;
  splat load wrapped in try/catch — matcha has no splat). Mesh **symlinked** (not copied).
- **`studio/server.py`**: `GET /api/scout/<scene>/<variant>` builds (via a `uv … --with numpy`
  subprocess; studio python has no numpy) + returns `{url}`; `GET /scout/<scene>/<variant>/<file>`
  serves it (streams the `.ply` symlink).
- **Rank UI**: "🔭 Scout" button in the Manifest header → `openInScout()` → `/api/scout` →
  `window.open` the viewer.
- **Verified**: matcha-15 → 539 spine / **15** highlighted; DA3-24 → 539 / **24**; viewer +
  frustums.json + streamed mesh all served on `:8091`. **Operator visual verify pending (T-020)**
  — the one risk is `viewer.html` rendering the overlay *without* a splat scene (mesh-only);
  reload `/rank`, focus a render, click 🔭 Scout.

### WebGL per-draw index limit + size-gate (2026-06-15)
First operator test surfaced a hard browser limit: WebGL draws **at most 30M indices = 10M
triangles per draw call**; a mesh above that **silently won't render** (matcha TSDF/tetra ≈
17–19M tris, DA3-24 = 4.6M renders fine). On-the-fly quadric decimation was tried and
**rejected by the operator** ("decimating on-the-fly is not an acceptable solution"). Interim
decision: **gate the button, don't mutate the mesh.**
- `build_scout_mesh.py` serves the mesh **as-is** (symlink, no copy/decimate).
- The scene payload already carries `mesh.faces` per variant (`_ply_stats`, header-only).
- Rank UI (`app.js` `updateScoutButton` + `openInScout` guard, `SCOUT_MAX_TRIS = 10_000_000`):
  **disables 🔭 Scout** when `faces ≥ 10M`, with a tooltip naming the limit; `style.css`
  `#open-scout:disabled`. Confirmed against the live payload: DA3-24 (4.6M) enabled; all matcha
  variants + the large historical meshes (11–27M) disabled.
- **Follow-up (deferred):** to make large meshes inspectable, the viewer needs to either
  **chunk** the mesh across multiple draws or load a **pre-decimated** sidecar produced by the
  recon pipeline (NOT at request time). File as a separate story if/when needed.

_(Operator request 2026-06-15. Reuses `verify_viewer` build_frustums/viewer; the new bit is
variant→inputs resolution + orient-applied spine frustums + the Rank-UI button/endpoint.)_

---
xid: STO-SCN-151
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-16
depends-on: [STO-SCN-150]
bd-id: krabby-ybbe
assignee: scout
---

# Scout in the Scenes tab — view gaussian (default) + define Render Views

## Summary

Show the scene's **scout** in the config center view (the DA3 **gaussian by default**), and let the
operator **define Render Views** (1…N) from within the tab. Reuses the `verify_viewer` scout viewer.

## Context

Creation steps **10–11**. The scout surface (`build_verify.py` → `verify_viewer/`) renders the
registered gaussian + posed frustums; render views are the named virtual cameras the renderer uses.
Full spec: **EPI-SCN-SCENE-MANAGER § Creation flow**.

## Design / scope
- **Scout center view**: embed the scout viewer showing the registered gaussian (default) +
  frustums; reuse `build_verify.py` to assemble the serve payload behind an endpoint.
- **Define Render Views**: position a virtual camera and **save it as a named Render View** (reuse
  the `/camera-save` flow / v4 graph writer that already materializes comparison cameras); list +
  manage the scene's render views.
- Render views feed the Rank tab (the renderer already consumes views).

## Definition of Done
- [ ] Scout view renders the registered gaussian (default) + frustums for the scene, in-tab.
- [ ] Operator defines + names ≥1 Render View from the tab; it persists and is usable by the renderer.
- [ ] Reuses `build_verify.py`/`verify_viewer` + the existing camera-save/graph-writer path.

## Out of scope
- MEASURE / Normalize Units (STO-SCN-152) — a sibling scout action.
- Mesh viewing / culling (cull/condition epics).

## Build notes (2026-06-16)
- **Backend** `scout_serve.py` (numpy-FREE; shells the heavy work to a numpy
  python): `numpy_python()` discovery (env `KRABBY_NUMPY_PYTHON` → scan, found
  py3.10), `resolve_scout()` (newest subset/solve/scout with `scout.gs.ply`),
  `build_serve()` (runs `verify_viewer/build_verify.py` → `<scene>/verify-serve/`
  in ~6 s — scout gaussian + frustums + de-warped frames + viewer.html +
  match.html), `list_views()` / `author_overview()` (reuse the
  `views/<slot>/view.json` convention via `author_overview_view.py`).
- **Endpoints** (`rate_renders/server.py`): `POST …/scout-build` (threaded +
  `scout_build_status.json`), `GET …/scout-status`, `GET …/verify/<path>`
  (serves the verify-serve dir, path-clamped, mime per ext), `GET …/views`,
  `POST …/view-author`.
- **Frontend** `static/scenes-scout.js` (`window.scenesViews.scout`, 5th view
  tab): embeds `verify/viewer.html` (gaussian + frustums) when built, else a
  **Build scout view** button (+ poll); a **Render views** panel (list + **+
  Overview view**).
- **Verified:** `scout_serve` driver + HTTP e2e on real 001-patio — build_serve
  produced the serve dir (viewer.html 200, frustums.json 200, **scout.gs.ply 33
  MB served**), traversal → 404, views listed. The 3D scout interaction +
  interactive render-view *positioning* (vs. the one-click overview) are
  operator-verified (T-020).

## Definition of Done (status)
- [x] Scout view renders the registered gaussian (default) + frustums for the scene, in-tab (embed of the built verify surface).
- [x] Operator defines + names ≥1 Render View from the tab (overview author); it persists (`views/<name>/view.json`) and is usable by the renderer.
- [x] Reuses `build_verify.py`/`verify_viewer` + the `views/*/view.json` writer.
- [ ] **Operator-verified (T-020):** Scenes → Scout → Build → confirm the gaussian renders; author an overview view + confirm it persists. (Interactive camera-pose-capture-from-viewer deferred — overview author shipped.)

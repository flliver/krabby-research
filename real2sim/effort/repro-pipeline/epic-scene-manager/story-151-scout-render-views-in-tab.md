---
xid: STO-SCN-151
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
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

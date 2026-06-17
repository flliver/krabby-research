---
xid: STO-SCN-138
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-15
depends-on: [STO-SCN-136]
bd-id: krabby-4ev6
assignee: krabby
---

# Rank UI: surface cull/condition settings + make condition nodes first-class variants

## Summary

The Rank tool shows each variant's transform settings in the Manifest panel — but a **culled
(condition) node's own settings** (`min_views` / `max_dist_from_cluster` / `floor_z_min` /
`cambox_expand`) are **not** surfaced: they get folded under the *represent* algo and dropped from
the one-line description. This story makes the condition node carry its **own** `algo` + settings
so the cull knobs render in the Manifest (and the description), letting the operator compare a raw
mesh vs its culls by the exact cull parameters.

## Context

STO-SCN-136 added the `cull-mesh@0` condition node; the rank UI already **enumerates** condition
nodes as variants (`rate_renders/server.py` `_scene_payload`, `v4core.scan_scene` `conditioned`),
renders them, and the frontend (`app.js` `renderManifest`) already renders *every* transform's
parameters generically. The gap is purely in **what the server attaches** to a condition node's
manifest.

## Problem

In `rate_renders/server.py` `_scene_payload` the condition node's manifest is built as:

```python
"transforms": {rep["algo"] or rep["kind"]: {
    "parameters": {**rep["settings"], **m.get("settings", {})}}}
```

— so the cull settings are (a) merged under the **represent** algo (mislabeled), and (b) invisible
in `describe_render`, whose `_SALIENT` tuple has no cull keys. And `v4core.scan_scene`'s
`conditioned` entries omit `algo`, so the server can't even name the cull task.

## Design

### Changes
| File | Change |
|------|--------|
| `real2sim/v4core.py` (`scan_scene`) | conditioned entries gain `"algo": cmd_.get("algo")` (+ keep `settings`/`renders`) |
| `real2sim/rate_renders/server.py` (`_scene_payload`) | a condition node's `transforms` = an ordered **chain** `{<cull algo>: {cull settings}, <represent algo>: {represent settings}}` — cull first so `describe_render` summarizes the cull; base meshes unchanged |
| `real2sim/rate_renders/server.py` (`_SALIENT`, `_setting_tok`) | add `min_views` / `max_dist_from_cluster` / `floor_z_min` / `cambox_expand` tokens (skip disabled: `max_dist=0`, `min_views=0`, `cambox=-1`) |

Frontend needs **no change** — `renderManifest` already loops over `transforms` and prints all
parameters; once the server labels the cull transform with its real algo + settings, they render.

### Backwards-compat
Pure additive read-side change — no taskdef/graph/store mutation, base-mesh manifests unchanged.
Store-identity rule unaffected (canonical: STO-SCN-136 § "Backwards compatibility — store
identity").

## Definition of Done

- [ ] A culled variant's Manifest panel shows a distinct `cull-mesh@0` transform with its
      `min_views`/`max_dist_from_cluster`/`floor_z_min`/`cambox_expand` values.
- [ ] The one-line description includes the salient cull knobs (e.g. `cull-mesh@0 · ≤3.0m · …`).
- [ ] Condition nodes remain deep-linkable (`?scene&view&variant=<cull id>`) and rankable.
- [ ] Base-mesh manifests are unchanged (no regression).
- [ ] **Operator-verified (T-020):** open a culled variant in the Rank tool, confirm the cull
      settings are visible and correct.

## Testing
- [ ] `scan_scene` conditioned entries include `algo`.
- [ ] `_scene_payload` emits the cull transform with cull settings for a condition node; base mesh
      unchanged.
- [ ] `_setting_tok` formats cull knobs; disabled values (`max_dist=0`/`min_views=0`/`cambox=-1`) omit.

## Out of scope
- Editing cull settings from the UI (this surfaces them read-only; a "re-cull with knobs" launcher
  is a later story if wanted).
- The Scout 3D viewer (STO-SCN-135) — unaffected.

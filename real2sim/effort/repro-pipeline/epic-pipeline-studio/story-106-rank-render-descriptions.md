---
xid: STO-SCN-106
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-15
depends-on: []
bd-id: krabby-pvz
assignee: krabby
---

# Rank UI: per-render Description (how it was built) + backfill

## Summary

Each render in the rank UI carries a **Description** — an ultra-succinct narrative of *how
that render was built* (the pipeline that produced it) — so the operator ranks with the
provenance in view, not just the pixels. All currently-visible renders are backfilled.

## Context

The rank surface (`rate_renders` / the studio rank tab, STO-SCN-074) shows N comparison
renders per view and asks the operator to rank them. Today the variants are distinguished
only by a terse variant id + the settings sidecar — the operator can't *see* why render A
differs from render B without cross-referencing. Ranking is a judgment about the *pipeline*,
not the image alone; the "how it was built" narrative belongs next to the image.

The provenance already exists: every render is content-addressed in the v4 store
(`represent/<kind>/<RID>/…/meshify/<method>/<MID>/…/condition/<CID>/renders/<REND>/`) with a
`metadata.json` per identity, and pre-v4 renders carry a `<view>.json` settings sidecar
(engine/resolution/mesh-source + the run's transform parameters). The Description is a
**generated narrative from that lineage** — no new data capture, just synthesis + display.

## Problem

Operator ranking is provenance-blind: the UI doesn't say, per render, what pipeline made it
(e.g. *"matcha@1 posed · dense-strong · TSDF default · oriented"* vs *"da3@1 · TSDF · ICP-ref"*).
The operator asked for a per-render **Description** in an ultra-succinct narrative style, and
for **all visible renders to be backfilled** with it.

## Design

### Approach

- **Narrative synthesis** — walk the render's ancestry (v4 metadata chain: render → condition
  → meshify → represent → select/solve → ingest; or the pre-v4 `<view>.json` sidecar +
  transform params) and emit one ultra-succinct line per render: the representation
  (matcha@/da3@/posed|unposed), the mesh method + key config (TSDF default/lowmem, tetra),
  conditioning, orient, and any distinguishing setting (dense-regul, n-views, selector).
  Style: telegraphic, no filler — *"da3@1 posed · 24 voxel-views · TSDF · ICP-ref to matcha"*.
- **Display** — a Description block per render card in the rank UI, beside/under the image.
- **Backfill** — render the Description for every render currently surfaced by the rank UI
  (all scenes/views/variants the aggregator lists), so nothing shows blank.

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/server.py` (and/or `studio/`) | derive per-render description from the v4 ancestry / settings sidecar; expose via the render/manifest API |
| `real2sim/rate_renders/static/app.js` + markup | render the Description block per render card |
| (helper) | a `describe_render(render_identity\|sidecar)` narrative synthesizer (reused for backfill) |
| backfill | generate descriptions for all visible renders (no orphaned blanks) |

## Definition of Done

- [x] Every render card in the rank UI shows a **Description** of how it was built.
      (Big grid tile: visible `.desc` under the label; small ranking card: hover tooltip.
      Wired in `rate_renders/static/app.js` — inherited by the studio rank tab at :8091.)
      *Operator visual sign-off pending (T-020).*
- [x] The Description is **ultra-succinct narrative** (one telegraphic line), derived from the
      render's actual provenance — not a static label. (`describe_render(manifest)` in
      `rate_renders/server.py`: v4 algo+settings or the legacy transform chain → e.g.
      `matcha@0 · dense-strong · align-strong · tetra · 11.4M tris`.)
- [x] **All currently-visible renders are backfilled** — no blanks. (Read-derived: the
      manifest is rebuilt each `/api/scene` request, so every visible render gets a
      description automatically; verified 4/4 on 001-patio.)
- [x] Description is read-derived (no new stored field) — survives a server restart by
      construction.
- [x] Tests: the synthesizer produces the expected narrative for matcha + da3 lineages
      (`tests/test_describe_render.py`, 6/6).

## Testing

### Unit / fixture tests

- [ ] `describe_render` on a known matcha@1 lineage → expected ultra-succinct string.
- [ ] `describe_render` on a da3@1 lineage → expected string (distinguishes from matcha).
- [ ] Missing/partial provenance → graceful degraded narrative (never crashes/blank).

### Integration

- [ ] Rank UI for a real scene: each visible render shows its Description; the strings
      distinguish the variants the operator is actually comparing.

## Out of scope

- Editing/authoring descriptions by hand (they are derived from provenance).
- Changing the ranking math or the leaderboard aggregation.
- The persistence-of-rankings work (that's STO-SCN-107).

## Implementation Notes

**Built (2026-06-15).** `describe_render(manifest)` in `rate_renders/server.py` synthesizes an
ultra-succinct, dot-joined narrative from the render's manifest — works on both the **v4**
shape (`{algo: {parameters}}` + the `[mesh-method]` label + ply stats + deliverable/align
flags) and the **legacy** transform chain (`{transform-NN: {kind, parameters}}`). Salient
settings get telegraphic tokens (`posed`, `dense-strong`, `voxel-select`, `24v`, `504px`,
`align-strong`); defaults are dropped; tokens dedup keeping order; it never raises (degrades
to the variant name). Attached as `manifest["description"]` in both `_scene_payload` paths.

**Read-derived ⇒ backfill is automatic** — the manifest (hence the description) is rebuilt on
every `/api/scene` request, so all visible renders are covered with no separate backfill job
and no stored field.

**Display** (`rate_renders/static/app.js` + `style.css`): big grid tile shows a visible
`.desc` block under the label; the small ranking cards (≤88×50px) carry it as a `title`
tooltip to avoid clutter.

**Surfaces it reaches.** The studio app (`:8091`) **subclasses the rate_renders handler** and
embeds the rate_renders static app at `/rank` (one implementation, T-023) — so the change
flows to the operator's `:8091` rank tab AND standalone `rate_renders` (`:8090`). Studio
restarted to pick up the server change; assets served at `/static/{app.js,style.css}`.

**Verified.** `tests/test_describe_render.py` 6/6; real 001-patio payload → 4/4 variants
described, distinguishing dense-strong / tetra-vs-tsdf / tri-count. Live at
`http://localhost:8091/` (rank tab) — **operator visual sign-off (T-020) is the close gate.**

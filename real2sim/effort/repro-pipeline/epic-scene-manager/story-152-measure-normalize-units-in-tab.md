---
xid: STO-SCN-152
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-16
depends-on: [STO-SCN-151]
bd-id: krabby-4p8a
assignee: scout
---

# MEASURE + Normalize Units in the Scenes tab (formalize STO-SCN-144 / datum.json)

## Summary

Surface the **MEASURE units** flow (press `M`) inside the Scenes tab's scout view and add a
**Normalize Units** action that writes the metric datum — formalizing the already-shipped
STO-SCN-144 MEASURE mode + `calibrate_datum`/`datum_frame` back-end (001-patio calibrated at
s=4.45) into the UI.

## Context

Creation step **12**. The hard part is **done**: `verify_viewer/match.html` MEASURE mode (two-view
triangulation), `metric_scale.py`, `calibrate_datum.py` (recompute + gate + `apply_to_gauge`
datum sidecar), `datum_frame.py`. This story wires them as a first-class tab action. Full spec:
**EPI-SCN-SCENE-MANAGER § Creation flow**; back-end origin: **STO-SCN-144 / STO-SCN-016**.

## Design / scope
- **MEASURE mode** in the scout view (reuse `match.html` MEASURE): `M` to enter; `[ ]` photo nav;
  click P1 in ≥1 photos → `E`; click P2 in ≥1 photos → `E`; enter the **P1:P2 distance (meters)**.
- **Normalize Units** button → POST the MEASURE export to the back-end → run `calibrate_datum`
  (re-triangulate, aggregate log-median + spread, DA3 gross-error gate) → `apply_to_gauge` writes
  the additive `datum.json` (scale + datum frame + provenance) for the scene's solve gauge.
- Surface the result: `s`, spread, weak-triangulation / anisotropy / DA3-disagreement flags;
  support multiple control distances; re-measure overwrites one number.

## Definition of Done
- [ ] MEASURE mode usable in-tab (the STO-SCN-144 flow), multi-photo picking + `E` per endpoint.
- [ ] Normalize Units writes `datum.json` via `calibrate_datum`/`apply_to_gauge`; surfaces s + spread + flags.
- [ ] Reuses the shipped MEASURE/`metric_scale`/`calibrate_datum`/`datum_frame` back-end (no new math).
- [ ] **Operator-verified (T-020):** normalize a scene in-tab and confirm the written scale matches a known dimension.

## Out of scope
- The scale math/tool itself (STO-SCN-144, shipped) and the datum strategy (STO-SCN-016).
- Consuming `datum.json` downstream (USD export 017, metric cull 145).

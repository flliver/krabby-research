---
xid: STO-SCN-152
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
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

## Build notes (2026-06-16)
- **Backend**: `normalize_datum.py` (numpy CLI) ties the three SHIPPED seams —
  `calibrate_datum.recompute(export)` → scale + gate flags →
  `datum_frame.build_datum(centers, up, scale)` →
  `calibrate_datum.apply_to_gauge(...)` → additive `datum.json`. `--dry`
  computes without writing; refuses clobber without `--force`. `scout_serve.normalize()`
  shells it out to a numpy python; endpoint `POST /api/scene/<s>/normalize`
  {export, dry, force}.
- **Frontend** `static/scenes-measure.js` (`window.scenesViews.measure`, 6th
  view tab): embeds `verify/match.html` (the STO-SCN-144 MEASURE tool) + a
  paste-the-export box + **Normalize Units** (Dry-run default) → shows
  `s` + spread + anisotropy/weak-triangulation/DA3-disagree flags; Overwrite on
  existing datum.
- **Verified:** `normalize_datum --dry` on the real 001-patio solve → **s=4.45**
  (its datum.json left untouched); temp-dir write-path writes `datum.json` with
  the datum frame + refuses clobber; HTTP e2e — `match.html` serves, normalize
  dry-run returns the scale, bad body → 400. The MEASURE workflow + the real
  datum WRITE on a new scene are operator-verified (T-020).

## Definition of Done (status)
- [x] MEASURE mode usable in-tab (embeds the STO-SCN-144 match.html), multi-photo picking + E per endpoint.
- [x] Normalize Units writes `datum.json` via `calibrate_datum`/`apply_to_gauge`; surfaces s + spread + flags.
- [x] Reuses the shipped MEASURE/`metric_scale`/`calibrate_datum`/`datum_frame` back-end (no new math).
- [ ] **Operator-verified (T-020):** measure a known dimension in-tab, Normalize, confirm the written scale matches.

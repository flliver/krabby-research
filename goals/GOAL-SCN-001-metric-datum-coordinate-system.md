---
xid: GOAL-SCN-001
kind: goal
effort: scn
status: open
date: 2026-06-16
urgency: 0.70
importance: 0.90
delivers: [STO-SCN-016, STO-SCN-144, STO-SCN-145]
bd-id: krabby-p8lf
---

# Metric-scale datum + camera-relative coordinate system (meters for cull + USD/physics)

## What we're trying to do

Establish **one stable, camera-derived, gravity-aligned, metric coordinate frame (the datum) per
solve** — into which every generated mesh lands, and against which an operator authors **boolean
cull primitives in meters**. The cameras are the constant (we do not re-solve; the whole spine is
passed even for camera subsets), so culling is defined relative to the cameras *before any mesh
exists* and reused across the many meshes we generate. Absolute scale is recovered from a
**hand-measured control distance** and fixed **at the gauge level** so all downstream meshes + USD
export + IsaacSim physics inherit meters from a single source.

## Why now

Surfaced in the 2026-06-16 design session while planning boolean-primitive culling: a metric,
camera-relative frame is the missing foundation. **STO-SCN-016 is the ★ T2-acceptance blocker**
(the single biggest technical unknown per the manager audit), and it gates USD export + physics
spawn. Investigation killed the "free" path — DA3 `is_metric`/`scale_factor` is an unreliable prior
(measured **3.35 / 3.48 / 11.22** across three scouts of the *same* gauge), so the control-distance
tool must be built. OLAI 3d-reconstruction guidance confirmed the recipe (corpus:
`personal.research/3d-reconstruction/metric-scale-calibration/index.md`).

## The plan (cross-cutting)

**Datum recipe** (gauge-fixing — pin the SfM 7-DoF similarity):
`origin = camera centroid projected to the gravity ground plane · +Z = gravity (gauge_up) ·
+X = spine azimuth · +Y = Z×X · 1 unit = 1 m`. Metric + isotropic (a 2 m sphere is a 2 m sphere);
**not** a per-axis normalized unit cube (the cambox is vertically thin → that distorts shapes).

**Scale recipe** (STO-SCN-016 § Decisions D1–D7): primary = hand-measured control distance
(`s = D / d_solve`, applied as one scalar on the solve Sim3); DA3 demoted to prior + **1.5×
gross-error gate** + fallback; ≥2 control distances on different axes for anisotropy.

**Story DAG / dependency order:**

```
STO-SCN-144  →  STO-SCN-016  →  STO-SCN-145
(control-dist     (datum-level      (camera-relative metric
 tool, match.html) metric scale)     datum + boolean-primitive cull)
                                          ↑
STO-SCN-136 → STO-SCN-137 ─────────────────┘
(shipped)     (shipped)

then unblocks T2 export:  016 → STO-SCN-017 (mesh→USD) → STO-SCN-018 (IsaacSim spawn)
```

Build sequence: **144 → 016 → 145** (136/137 already shipped feed 145). 016 also unblocks 017/018.

## What this is NOT

- Not re-deriving gravity/up (STO-SCN-105 `gauge_up` already supplies it).
- Not a per-mesh scale transform — scale lives at the **datum/gauge**, applied once.
- Not trusting monocular metric as truth — it is a prior + gate only.
- Not the mesh-conditioning chain itself (cull/merge/verify/smooth — EPI-SCN-MESH-CONDITION);
  this GOAL is the metric frame those and USD export depend on.

## Status notes

- 2026-06-16: GOAL opened. urgency=0.70, importance=0.90. Delivers STO-SCN-016 (datum scale,
  scout-owned), STO-SCN-144 (control-distance tool, next build), STO-SCN-145 (primitive cull,
  draft). Decisions recorded in STO-SCN-016 § Decisions; corpus entry captured at OLAI.
- 2026-06-16: **ENGINEERING COMPLETE — BLOCKED ON OPERATOR VERIFICATION (T-020).** All code for
  016/144/145 built + tested (44/44) + committed (7 commits, ending `ca87855`): `metric_scale.py`,
  `calibrate_datum.py` (`apply_to_gauge` datum sidecar), `datum_frame.py`, `sdf_primitives.py`,
  `match.html` MEASURE mode, `cull-mesh@2` primitives tunable. Verify surface pre-staged +
  e2e-validated on real 001-patio (`/tmp/measure-001`). The three stories stay `in-progress`
  (NOT `shipped`) because the agent cannot self-close operator-facing surfaces (T-020).
  **OPERATOR ACTION (the only thing left, assigned to the human):**
  `python3 -m http.server 8099 --directory /tmp/measure-001` → open `match.html` → press `M` →
  measure one KNOWN real distance in two photos → export → `python3 real2sim/calibrate_datum.py
  --export <json>`. That number + the visual sign-off in Rank ship 016/144/145 and complete this
  GOAL. (Release early without measuring: `/goal clear`.)

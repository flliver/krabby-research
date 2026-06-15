---
xid: STO-SCN-131
parent: ./epic.md
kind: story
effort: scn
size: S
status: draft
date: 2026-06-15
depends-on: []
bd-id: krabby-bk4d
assignee: scout
---

# reconstruct-matcha + da3-infer: emit progress (lib_progress.sh + nanny-progress→MQTT) like solve/scout

## Summary

The long GPU reconstruct steps (`reconstruct-matcha`'s `run_in_matcha`, `reconstruct-da3`'s
infer) report **live progress** to MQTT + `nanny-progress`, the same way `solve` and `scout`
already do — so a 15–25 min weld shows phase/percent on `beeprz dash` and to any monitor,
instead of being a black box until it exits.

## Context / Problem

`run_in_matcha` runs `subprocess.run(["ssh", host, docker…], capture_output=True)` — output is
**captured, not streamed**, and `matcha.log` is written **only after the container exits**.
There is **no MQTT publish, no `nanny-progress`, no `lib_progress.sh`**. Same for the DA3 infer
path. So during a weld there's zero visibility (observed 2026-06-15 on matcha-15 — couldn't
report any % for the in-flight run).

This is also a **fleet-ops compliance gap**: the standing rule is that >30 s work on a
beeprz host wraps `nanny-progress`. `solve` (v4exec ~L462) and `scout` (`run_scout.sh` +
`lib_progress.sh`) honor it; matcha + da3-infer don't.

## Design

Mirror the `solve`/`scout` pattern:
- Stage `lib_progress.sh` next to the work on the host (already done for solve/scout).
- Wrap the matcha train + TSDF/tetra steps (and the DA3 infer) with `nanny-progress set
  <phase> <pct>` calls → MQTT, with the `trap … EXIT INT TERM` clear (fleet-ops hygiene).
- Phases for matcha: stage → train (the long pole; emit % from train.py iteration if cheap) →
  tetra-extract → tsdf-extract → gather.
- Consider streaming the container stdout (drop `capture_output=True` for a `tee`-to-log +
  live parse) so the log isn't end-of-run only.

| File | Change |
|------|--------|
| `real2sim/v4exec.py` | `run_in_matcha` (+ `cmd_da3` infer): stage `lib_progress.sh`, emit `nanny-progress`/MQTT phases, stream the log |

## Definition of Done

- [ ] A matcha weld shows phase/percent on `beeprz dash` + MQTT during the run.
- [ ] `matcha.log` is written incrementally (not only at exit).
- [ ] DA3 infer reports the same way.
- [ ] `nanny-progress` cleared on exit (success AND failure) — no stuck bar.
- [ ] Parity with `solve`/`scout` progress (same `lib_progress.sh` mechanism).

## Out of scope

- Changing the reconstruction algorithms.
- The scene-driver orchestration (STO-SCN-128) — it *consumes* this progress, doesn't provide it.

## Implementation Notes

_(Earned 2026-06-15: matcha-15's in-flight weld was un-observable. Also corrects the T3b doc,
which overclaimed "progress published to MQTT" as universal — true for solve/scout/render,
not matcha/da3-infer.)_

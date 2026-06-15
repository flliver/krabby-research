---
xid: STO-SCN-130
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-15
depends-on: []
bd-id: krabby-xy27
assignee: krabby
---

# reconstruct-matcha on a FINAL-N selection — posed from the parent solve (no re-solve)

## Summary

`reconstruct-matcha` can reconstruct a **FINAL-N selection** (a member-only subset emitted by
`select`) by **posing its members from the parent solve** — matcha@1 posed, no re-solve, no new
gauge. This is what makes the documented "point `primary` at the FINAL-N subset; reconstruct"
actually work, and unblocks **matcha-15** for 001-patio (STO-SCN-123).

## Context / Problem

`select` emits a FINAL-N subset that is **just a member list** (`subset.json`) — no `cameras/`,
no solve. But `cmd_matcha` resolves its solve from `subsets/<primary>/cameras/*/` and exits
**"no solve for primary"** when primary is a selection. Re-solving the N members separately would
mint a *new arbitrary gauge* (breaking spine consistency) — wrong. The members are already posed
in the **parent solve** (e.g. `62QEHJDAJZBI`); matcha@1 posed should reuse those poses.

**The mechanism already exists:** `cmd_matcha`'s posed path calls
`solve_to_sparse(<parent cameras.json>, staged_members)` which **restricts the solve to exactly
the staged members**. The only missing piece is *resolving the parent solve* when `primary` is a
member-only selection.

## Design

In `cmd_matcha`, introduce a **pose source** distinct from the reconstructed subset:
- `sub = primary` (the FINAL-N members — staged + counted + part of identity).
- If `subsets/sub/cameras/*` has a solve → `pose_sub, sid = sub, that solve` (unchanged behavior).
- **Else (selection):** resolve the parent via the `select` provenance — scan
  `subsets/*/cameras/*/select/*/final.json` for `final_subset == sub` → `pose_sub, sid` =
  that subset + solve. Print it loudly.
- Use **`pose_sub`** for every solve-relative read: `cameras.json`, `weld_to_solve_sim`, the
  `orient` dir. Keep **`sub`** for staging/members/`n_images`/identity `{subset: sub, cameras: sid}`.

`solve_to_sparse(<pose_sub cameras.json>, staged=sub members)` then mints a sparse/0 of just the
N members in the parent gauge — exactly matcha@1's contract.

| File | Change |
|------|--------|
| `real2sim/v4exec.py` | `cmd_matcha`: `pose_sub` resolution (parent-solve fallback for selections) via `resolve_pose_source()`; thread `pose_sub` through cameras.json / weld-sim / orient |

Depends on the parent solve having a `cameras.json` (STO-SCN-129; already present for
`62QEHJDAJZBI`).

## Definition of Done

- [ ] `reconstruct-matcha --sfm posed` on a FINAL-N selection (primary = FINAL-N) reconstructs
      the N members posed from the parent solve — no re-solve, no "no solve for primary".
- [ ] Identity is `{subset: FINAL-N, cameras: parent-solve}` (re-runs NOOP).
- [ ] The mesh grounds into a gauge consistent with the parent solve (weld→solve sim passes).
- [ ] Validated: matcha-15 on 001-patio (FINAL-15) produces a mesh + renders in the runoff.
- [ ] Operator sign-off on the rendered matcha-15 variant (T-020).

## Out of scope

- The render `cameras.json` general fix (STO-SCN-129).
- DA3 selection path (already handled — the scout posed the selection; STO-SCN-127).
- Spine (M>1) multi-segment reconstruction.

## Implementation Notes

_(Building 2026-06-15. The posed restriction mechanism `solve_to_sparse` already exists; this
is purely parent-solve resolution + path threading.)_

---
xid: STO-SCN-083
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-11
depends-on: []
bd-id: krabby-ym0
shipped: 2026-06-11
tasks: 1
complete: 1
---

# BUG: rank submit on v4 scenes crashes UI — POST returns rows[] but frontend reads d.row.submitted_at

## Summary

BUG (operator, 2026-06-11, scene 013): submitting a ranking on a v4
scene crashed the UI — `Error: can't access property "submitted_at",
d.row is undefined`.

## Root cause

The v4 POST branch (STO-SCN-074 absorption) returned
`{ok, rows[], store:"v4"}` while the frontend submit handler reads
`d.row.submitted_at` (the v2 contract). Contract drift between server
branch and client.

## Fix (shipped)

- Server: v4 response now includes BOTH `row` (v2 shape) and `rows`.
- Frontend hardened: falls back `d.row?.submitted_at → rows[0].ts →
  "now"` so a missing field can never throw.
- Verified: POST on 013 returns row.submitted_at; smoke row removed.

## Definition of Done

- [x] Reproduced, root-caused, fixed both sides, verified live.

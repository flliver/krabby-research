---
xid: STO-SCN-124
parent: ./epic.md
kind: story
effort: scn
status: shipped
shipped: 2026-06-15
date: 2026-06-15
depends-on: []
bd-id: krabby-sjzc
assignee: krabby
tasks: 1
complete: 1
---

# Discard render PWZ4S24AZ72T (invalid) from ranking

## Summary

The invalid TSDF mesh `PWZ4S24AZ72T` (the `12sharp-strong` matcha, `LQLIS7O67GHX`) is excluded
from the 001-patio runoff so it doesn't pollute the comparison.

## Resolution

**Operator-completed 2026-06-15.** The operator discarded `PWZ4S24AZ72T` from the ranking
directly. No code mechanism was built here — a general "retract/hide a variant" feature is
tracked under STO-SCN-132 (failed/invalid results as first-class, visible-in-data records).

## Definition of Done

- [x] `PWZ4S24AZ72T` no longer participates in the 001-patio runoff (operator-handled).

## Out of scope

- A general discard/retract mechanism in the rank UI / store (→ STO-SCN-132).

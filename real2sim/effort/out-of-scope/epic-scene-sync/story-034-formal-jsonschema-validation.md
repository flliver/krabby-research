---
xid: STO-SCN-034
parent: ./epic.md
kind: story
effort: scn
size: S
status: open
date: 2026-06-04
depends-on: []
bd-id: krabby-dbq
assignee: principal
---

# Formal jsonschema validation harness + fixture suite (CI gate) for the scene schemas

## Summary

A runnable validator + fixture suite that checks scene artifacts against the
committed JSON Schemas (`epic-scene-sync/schemas/*.schema.json`), wired so it runs
in CI. STO-SCN-026 authored the schemas and verified them with a *dependency-free
structural check* (no `jsonschema` installed here, and these schemas have no test
home yet); this story stands up the **formal** gate.

## Context

Split out of `STO-SCN-026` (which is closing). 026 delivered the schemas + a
structural sanity check; the formal `jsonschema` validation + the full fixture
matrix were explicitly deferred here so 026 could close on its actual deliverable
(the schema definition). Two specific fixture cases were **moved from 026's DoD**
to this story (see Definition of Done).

## Problem

The schemas are committed but nothing enforces them. Without a runnable validator,
`scene.toml`/`run.json`/`specification.json`/`results.json` files (especially the
~10 scenes `STO-SCN-033` will migrate) can drift from the schema undetected.

## Definition of Done

- [ ] `jsonschema` (Draft 2020-12) available to the validator (test extra / CI dep).
- [ ] A validator that checks a scene tree's artifacts against
      `schemas/{scene,run,specification,results}.schema.json`.
- [ ] Fixture suite — valid + deliberately-invalid fixtures — including the two
      cases **moved from STO-SCN-026**:
  - [ ] `provenance: "deduced"` with absent/`unknown` `environment.*` **validates**
        (the migration-leniency case — back-filled records aren't rejected for
        honestly lacking data).
  - [ ] A bad transform name (`transform-2-x`, unpadded ordinal / lowercase-prefix
        violations) is **rejected** by the documented regex.
- [ ] The `reference/004-sky-house/` example validates end-to-end against all schemas.
- [ ] Wired into CI (runs on change to `schemas/` or any scene artifact).

## Out of scope

- The schemas themselves (owned by `STO-SCN-026`, shipped).
- Migrating data (`STO-SCN-033`) — though this validator is what 033 runs to
  prove migrated scenes conform.

## Notes

- The structural check 026 ran already covers: conforming fixtures pass, missing
  required field fails, and the `measured`-requires-`host` conditional. This story
  formalizes those under `jsonschema` and adds the two cases above.

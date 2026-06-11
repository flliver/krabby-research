---
xid: STO-SCN-073
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: [STO-SCN-070, STO-SCN-071, STO-SCN-076]
bd-id: krabby-24v
shipped: 2026-06-11
tasks: 4
complete: 4
---

# Central run trigger: launch instance on one chosen host, capture full run record

## Summary

The "run" step: trigger a pipeline_instance centrally, execute it on
**one operator-chosen host** (host = explicit parameter, no
scheduler), and capture a run record sufficient for a third party to
re-run it — settings expansion, image digests, tool SHAs, input
hashes, logs, output locations.

## Context

Operator decision 3. Reproducibility machinery already exists
piecemeal: registry-pinned images, baked tools (`/opt/krabby-tools`),
expected-outputs hard gate + LFS-pointer input guard in the runner,
store-shape v2 `transient_data.location` stanzas. This story
composes them behind one trigger and makes the record complete.

## Problem

Runs today are launched by hand-invoked scripts per host; provenance
is captured but assembled from several places (spec JSON, results
JSON, image labels, commit log). Nothing guarantees a run record is
complete enough to reproduce from.

## Design

- Trigger: Studio (or CLI equivalent) takes (pipeline_instance, scene,
  host) → dispatches over the existing SSH + docker path the fleet
  already uses. No queueing, no multi-host, no retries-as-policy.
- Tooling-provenance policy enforced: image-baked tools only
  (`/opt/krabby-tools`); a run with `/tools` dev-mounts is marked
  non-reproducible and excluded from ranking comparisons.
- Run record (written into the run dir, v2-tracked as metadata):
  expanded settings per task, image registry digest, TOOLS_GIT_SHA,
  input content hashes, host, timestamps, rc + expected-outputs gate
  result, log location stanza.
- Long runs report via the existing progress conventions; failures
  recorded honestly in the run record, not papered over (T-003).

## Definition of Done

- [x] One real pipeline_instance triggered centrally and executed on
      an operator-chosen host end-to-end: `da3-8-giant-studio` →
      006-kubota on **tbeeprz (operator: "run it on t")** — infer
      17s + fuse 1s, alignment 2.9% (matches historical), render
      produced, store committed (803808c).
- [x] The run record alone names image digests (repo@sha256), tool
      SHAs (image label, measured from host docker), expanded
      settings, and 8 input content hashes. `repro_check check`:
      reproducible-by-record YES, deliverable NO (CC-BY-NC flag).
- [x] Gates: LFS-pointer input guard (host-side grep before
      dispatch) + expected-outputs hard gate (catalog-declared
      outputs verified post-gather; rc=0 lies) — patterns verified
      against the real gathered run. Plus chown-in-dispatch (gather
      hygiene; first attempt left root-owned residue on t, now
      cleaned + prevented).
- [x] Existing manual script path still works unchanged
      (run_transform.py untouched; matcha keeps its hardened runner,
      composing it under the trigger is follow-on).

## Implementation Notes

- `real2sim/run_pipeline.py`. Host is an explicit `--host` parameter
  (decision 3) — the permission layer independently enforced this
  when the agent tried to pick a host itself.
- Found+fixed during validation: da3_infer_gs.py's out arg is the
  data ROOT (exports/ created underneath) — first dispatch nested
  exports/exports; da3-infer catalog output patterns corrected to
  the measured layout.
- by_record digest check fixed: registry digests arrive as
  `repo@sha256:…`, not bare `sha256:…`.

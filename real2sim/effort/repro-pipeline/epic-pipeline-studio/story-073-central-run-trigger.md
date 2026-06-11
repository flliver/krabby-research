---
xid: STO-SCN-073
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-11
depends-on: [STO-SCN-070, STO-SCN-071, STO-SCN-076]
bd-id: krabby-24v
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

- [ ] One real pipeline_instance triggered centrally and executed on
      an operator-chosen host end-to-end.
- [ ] The run record alone names image digests, code SHAs, expanded
      settings, and input hashes for every task_run.
- [ ] Gates inherited: expected-outputs hard gate + LFS-pointer guard
      fire from the triggered path.
- [ ] Existing manual script path still works unchanged.

---
xid: MSG-PROJ-003
content-path: /private/var/krabby/research/messages/2026-06-09/21-16-15-handoff-2116/001/msg-handoff-2116.md
kind: msg
effort: proj
status: open
date: 2026-06-09
to: devex
from: devex
topic: handoff-2116
bd-id: krabby-0ju
---

# Handoff from previous devex session

## What Was Happening

Operator asked (given DES-SCN-TX state) to grant the project **always-allow
read access to all scene data**. Did that in `.claude/settings.local.json`.

Key correction made mid-session: the scene store is NOT at the stale
`011-scene-reconstruction/data/scenes` path the `inventory.md` cites. With
the `/Volumes/Archives-01` volume online, scenes were found migrated out to a
dedicated top-level git repo: **`/Volumes/Archives-01/krabby/scenes`**,
surfaced via the workspace symlink **`/var/krabby/scenes`**. The old `…/data`
symlink now only holds `sfm-scaling-out`.

Final grant (3 path spellings, one location, Read-tool-only):
- `Read(//var/krabby/scenes/**)`
- `Read(//private/var/krabby/scenes/**)`
- `Read(//Volumes/Archives-01/krabby/scenes/**)`

After my edits, the operator hand-added a 4th allow entry: a `Bash(ssh
t.pski.org '…')` one-liner that rm's a stale matcha repro run, then
`git checkout -- .` + `git lfs pull` + `stat`s an input JPG under
`dtu-bicycle/input/preproc-01-frame-select/…`. → Operator is actively
reproducing a MAtCha pipeline run for the `dtu-bicycle` scene on host
**theo (t.pski.org)** and working the scene git-LFS store. This is the live
thread feeding EPI-SCN-PIPELINE-RUNNER.

## What Needs to Happen Next

- If the operator wants shell-side (Bash) reads of scenes to also stop
  prompting, add the specific `Bash(...)` read patterns — the current grant
  is Read-tool-only by design.
- Watch for the scene store moving again once the `krabby scenes` sync CLI
  (STO-SCN-029) standardizes a local scenes root — the Read grant will need
  a matching path then.
- Resume EPI-SCN-PIPELINE-RUNNER (config-driven pipeline runner) — the
  dtu-bicycle matcha repro on theo is the current concrete exercise.

## Key Context

- Scene store = its own git repo w/ **git-LFS** + a `.gitignore` encoding
  store hygiene (no `.DS_Store`, no `.blend1/2`). LFS pulls happen on theo.
- `/var` → `/private/var` (macOS); permission matcher keys on the literal
  path presented, hence the 3 spellings incl. the `/Volumes/...` realpath.
- This project has **no `bin/ccc-bd`** — use the global `ccc-bd` on PATH.
- Lesson (T-004/T-012): don't trust a symlink's resolved target against an
  unmounted volume — confirm against the live filesystem.

## Active Files

- `/private/var/krabby/research/.claude/settings.local.json` — scene-read
  grant + operator's ssh-repro allow entry (do not revert).

## Beads XIDs

- `EPI-SCN-PIPELINE-RUNNER` — **in_progress** (pri 1); config-driven pipeline
  runner w/ pluggable transforms. The dtu-bicycle matcha repro on theo is the
  live exercise. Not mutated by this park.
- `DES-SCN-TX` — open; "TX — Discovered Work" design, still a template stub.
  Framed the scene-data access request.

## Status notes

- 2026-06-09: Filed by /park. Beads state left untouched (parking is
  informational).

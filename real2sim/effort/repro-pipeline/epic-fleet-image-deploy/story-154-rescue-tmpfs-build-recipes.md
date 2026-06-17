---
xid: STO-SCN-154
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-16
depends-on: []
bd-id: krabby-87d1
assignee: krabby
---

# Rescue dbeeprz /tmp build recipes; make the repo the build source

## Summary

The Dockerfiles + patches + tools that actually built `krabby-matcha`
and `krabby-da3` are recovered off dbeeprz's tmpfs into the committed
repo, and future builds run from the repo — so a reboot can never
again delete a build recipe.

## Context

Highest-urgency item from the 2026-06-16 audit. dbeeprz `/tmp` is
**tmpfs (15 G, wiped every reboot)** and the live build recipes sit
there:

- `/tmp/matcha-build/` — `Dockerfile` (mtime 06-10 16:37 = matches the
  `matcha:0.2.2` build), `patches/`, `NOTES.md`, `requirements.txt`
- `/tmp/da3-build/` — `Dockerfile` + `krabby-tools/`
- `/tmp/tools/` — `da3_infer_gs.py`, `da3_render_view.py`, `gauge_align.py`

These are the **only copies** of what built the deployed images. An
immediate copy-off was dispatched to baeprz `ops`; this story makes
the rescue durable (committed) and removes the tmpfs dependency.

## Problem

Build inputs that exist only in tmpfs are one reboot from gone, and
the deployed images can't be rebuilt without them. The repo already
carries `images/matcha/` and `images/da3/` — so the task is to
**reconcile** the tmpfs copies against the repo (diff, fold in any
delta), then build from the repo henceforth.

## Design

### Approach

1. `ops` `docker save` / `scp` the three tmpfs dirs off dbeeprz to
   persistent storage (in flight — non-destructive, read-side).
2. `diff` each rescued recipe against its repo counterpart
   (`images/matcha/`, `images/da3/`). Fold any delta into the repo.
3. Build matcha/da3 **from the repo checkout** (persistent `/home`),
   not `/tmp`; document the build cwd in each image's README.

### Changes

| File | Change |
|------|--------|
| `images/matcha/{Dockerfile,patches/,NOTES.md,requirements.txt}` | reconcile with rescued `/tmp/matcha-build` |
| `images/da3/{Dockerfile,krabby-tools/}` | reconcile with rescued `/tmp/da3-build` + `/tmp/tools` |
| `images/*/README.md` | state: build from the repo checkout, never `/tmp` |

## Definition of Done

- [ ] The three tmpfs build dirs are saved to persistent storage off dbeeprz.
- [ ] Each rescued recipe diffed against the repo; deltas committed (or confirmed identical).
- [ ] matcha + da3 documented to build from the repo checkout, not `/tmp`.
- [ ] No build input for matcha/da3 exists only in tmpfs.

## Out of scope

- The 28-file MAtCha *source* customizations (separate concern → STO-SCN-155).
- Rebuilding/pushing the images (only needed if a delta is found; otherwise the existing registry tags stand).

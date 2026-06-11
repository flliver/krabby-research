---
xid: MSG-PROJ-002
content-path: /private/var/krabby/research/messages/2026-06-09/20-47-18-handoff-2047/001/msg-handoff-2047.md
kind: msg
effort: proj
status: open
date: 2026-06-09
to: engineer
from: engineer
topic: handoff-2047
bd-id: krabby-411
---

# Handoff from Previous Session

## What Was Happening

Executed STO-SCN-030 (fleet scene-store auto-sync, gated by
`~/.config/krabby/scene-sync.toml`) end-to-end, assigned via /tell from the
krabby workspace agent. Built, tested, deployed, recorded. **All engineering
DoD items are checked**; only the T-020 operator-verification checkbox is
open, and the workspace agent has since appended T-020 evidence to the story
(real-data convergence: ~4.5 GB / 3 commits ff'd to b/s/d on the 17:17 PDT
automatic tick; t self-resolved DIRTY→OK; operator saw the DIRTY surface).
**Close decision rests with the operator** — do not self-close.

## What Needs to Happen Next

1. If the operator ratifies: close STO-SCN-030 via `/done` (the story's DoD
   + status notes already carry the evidence).
2. Open design question left for principal/AID (in story notes): strict-clean
   precondition stalls producer hosts that hold untracked run outputs —
   alternative is allowing ff when only untracked files are present. Await
   their call; if accepted, it's a small change in `krabby-scene-sync`'s
   precondition block.
3. The epic's AS-BUILT § (git+rsync transport, `krabby-scenes-sync`) predates
   the git-lfs-transfer fix and this story — STO-SCN-029 re-scope/reconcile
   to as-built is still open (noted in 029's status notes).
4. Branch `jdp/m11-real2sim` has my two local commits **not pushed**
   (HUG-KRB-002: check before pushing). Other uncommitted files in the tree
   (.ccc/*, repro-pipeline rollup flips, design.md, story-036, aid checkpoints)
   are NOT mine — left untouched.

## Key Context

- Artifacts: `scripts/scene-sync/{krabby-scene-sync,install.sh,
  krabby-scene-sync.service,krabby-scene-sync.timer}` — commits `4f94b94`
  (build) + `f5bec1e` (deploy record) on `jdp/m11-real2sim`.
- Deployed `enabled=true` on t/b/s/d: systemd **user** timer 30 min, linger
  enabled, dist copies at `~/.local/share/krabby/scene-sync-dist/` (kept in
  sync with source, incl. the install.sh gate-echo fix).
- Gate semantics verified on Mac + bbeeprz: absent config / `enabled=false`
  ⇒ rc=0, zero output. Diverged scratch clone ⇒ untouched, `DIVERGED`
  surfaced, rc=1.
- Visibility surface (engineer's call, documented in story):
  `~/.local/state/krabby/scene-sync.status` one-liner
  (`<utc> <OK|DIRTY|DIVERGED|WRONG-BRANCH|MISSING-STORE|ERROR> <detail>`)
  + rotated log + failed unit on error.
- Timer interval is templated from config by install.sh at install time —
  interval change ⇒ re-run installer (documented in the unit file).
- Mac: deliberately disabled — `~/.config/krabby/scene-sync.toml` with
  `enabled=false` + in-file rationale (author host, T-019); no launchd plist
  shipped. Manual sync one-liner documented in the config.
- Quirk learned: `kill -0` on a root-owned pid returns EPERM on macOS ⇒ the
  lock's liveness check only trusts same-user pids (fine — state dir is
  per-user under $HOME).
- Hub: `jeremy@j.pski.org:/games/krabby/scenes`, branch `trunk`,
  `receive.denyCurrentBranch=updateInstead`; canary commit `37bca02` was mine
  (file `.fleet-sync-canary` at store root — harmless, could be cleaned up or
  kept as the standing canary file).
- Fleet hosts sleep: `beeprz wake <h>` (t/b/s/d), then `jeremy@<h>beeprz`.
  Remote `systemctl --user` needs `XDG_RUNTIME_DIR=/run/user/$(id -u)`.

## Active Files

- `scripts/scene-sync/*` (committed)
- `real2sim/effort/out-of-scope/epic-scene-sync/story-030-fleet-distribution-cache.md`
  (committed; workspace agent appended a T-020 evidence note after my commit —
  that note is currently uncommitted in the tree)
- `real2sim/effort/out-of-scope/epic-scene-sync/story-029-lan-first-sync-cli.md`
  (same: my note committed; a later baeprz-ops note is uncommitted)

## Beads XIDs

- `STO-SCN-030` — in-progress; engineering complete, all DoD checked except
  operator-verification checkbox; T-020 evidence recorded in story notes by
  workspace agent; awaiting operator close decision.

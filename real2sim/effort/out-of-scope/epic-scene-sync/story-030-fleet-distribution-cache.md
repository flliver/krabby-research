---
xid: STO-SCN-030
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-04
depends-on: []
bd-id: krabby-2hq
assignee: engineer
priority: 1
title: Fleet scene-store auto-sync, gated by ~/.config/krabby/ config
---

# Fleet scene-store auto-synchronization, gated by local config

## Summary

Every fleet host (t, b, s, d + the Mac) keeps its `~/krabby/scenes` clone
automatically current with the j hub — **gated by an explicit per-host opt-in
config at `~/.config/krabby/`**. No gate file / `enabled=false` → no sync, ever.

## Context

2026-06-09: b/s/d were discovered **11 commits behind** the hub (silently —
nothing notices staleness); manual `git pull && git lfs pull` brought all hosts
to parity (~2.9 GB LFS delta each, through the new `git-lfs-transfer` server on
j — both transport directions now proven at scale, see STO-SCN-029 notes).
Operator directive: automate, but the automation must be **locally gateable via
configuration** (T-019 — automation yields to the human; a host owner can turn
it off without touching the fleet).

## Design

### Config gate (normative): `~/.config/krabby/scene-sync.toml`

```toml
[sync]
enabled  = true        # THE gate. File absent or false => script exits 0, silently.
remote   = "j"         # git remote to pull from
interval_minutes = 30  # timer cadence
lfs      = "full"      # full = git lfs pull | skip = refs/pointers only
store    = "~/krabby/scenes"   # clone location (Mac: /var/krabby/scenes)
```

- **Default OFF**: no config file means no sync. Enabling is a deliberate,
  per-host act. `~/.config/krabby/` is the home for any future krabby
  host-local configuration (single dir to audit).

### Sync script (`krabby-scene-sync`, lives in krabby-research, version-controlled)

1. Read config; exit 0 quietly if absent/disabled (the gate).
2. Take a lock (`~/.local/state/krabby/scene-sync.lock`); skip if held.
3. Preconditions: store exists, worktree **clean**, on `trunk`. Else: log + surface, no action.
4. `git fetch <remote>`; **fast-forward only** (`git merge --ff-only`). Divergence
   is never auto-resolved — log loudly + surface (this host has unpushed work).
5. `git lfs pull` per `lfs` mode.
6. Log to `~/.local/state/krabby/scene-sync.log` (timestamped, rotated).
7. Never pushes. Read-only with respect to the hub.

### Scheduling

- Linux fleet (t/b/s/d): **systemd user timer** (`krabby-scene-sync.timer`),
  interval from config; unit files installed by a setup script, not hand-copied.
- Mac (optional, same script + launchd plist) — Mac is usually the *author*, so
  default config there should stay disabled or `lfs = "skip"`.

### Visibility

- Failures and divergence must be obvious (T-026): log + a `beeprz dash`-visible
  surface (nanny-progress is for foreground jobs; a status line file or journal
  entry the dashboard can read is enough — engineer's call, documented).

## Definition of Done

- [ ] `krabby-scene-sync` script + config schema committed to krabby-research (documented in script header + this story)
- [ ] Gate honored: absent config => no-op; `enabled=false` => no-op (tested)
- [ ] Fast-forward-only: a diverged clone is left untouched and the condition surfaced (tested)
- [ ] systemd user units + installer; deployed on t, b, s, d with `enabled=true`
- [ ] Convergence test: commit on hub → all enabled hosts current within one interval
- [ ] Mac story documented (enabled or deliberately not, with rationale)
- [ ] STO-SCN-029/030 notes updated; outposts legacy tree NOT touched by this work

## Status notes

- 2026-06-09: Story fleshed out per operator directive (automate + local config
  gate at `~/.config/krabby/`); reassigned devex→engineer, P4→P1. Manual parity
  sync of b/s/d performed today is the baseline this automation replaces.

# Beads — issue tracking

This repo uses **[Beads](https://github.com/steveyegge/beads)** (`bd`)
as a portable, repo-resident issue tracker. The DB lives at the repo
root in `.beads/` and is committed to the repository so progress is
visible across the team without depending on an external service.

> **Why beads?** Lightweight, single-file tracking; auto-syncs JSONL
> exports for git portability; supports labels, priorities, and
> blocking dependencies. Adopted as part of the M11 milestone work for
> Krabby Co's Scene Reconstruction & Locomotion Benchmarking
> deliverable. Future milestones can either share this DB with their
> own `<milestone>-` prefix or set up their own.

## Install

```bash
brew install steveyegge/beads/bd      # macOS via Homebrew tap
```

The current pinned version is `1.0.3` or later; older versions used a
SQLite backend, the new versions use Dolt. Run `bd --version` to check.

For other platforms, see https://github.com/steveyegge/beads/releases.

## Daily commands

All commands are run from anywhere inside the repo (`bd` walks up to
find `.beads/`):

```bash
bd ready                 # show unblocked work, sorted by priority
bd list                  # all open beads
bd list --status closed  # historical / closed work
bd show <id>             # full bead details (e.g. bd show m11-u3l)
bd graph <id>            # text DAG visualization rooted at <id>
bd dep tree <id>         # tree view of blockers
```

Creating / closing:

```bash
bd create "Bead title" -p 0 -t task --labels m11,t2,phase-e
bd close <id> --reason "What was done; cite commits or notes"
bd dep add <blocked-id> <blocker-id>     # <blocker-id> blocks <blocked-id>
```

## Naming convention

Bead IDs are auto-generated as `<prefix>-<hash>` (e.g. `m11-u3l`).
Per-milestone prefix:

- **`m11-`** — Milestone 11 (Scene Reconstruction & Locomotion Benchmarking)
- Future milestones use their own prefix (e.g. `m12-`, `m13-`) — do not
  reuse `m11-` for unrelated work.

**Title format** for M11 beads (Task hierarchy aligned with grant):

```
T<N>.<sub-id> — <title>          # e.g. T2.E1 — Scale Calibration ★ BLOCKER
R<n> — <title>                    # cross-cutting risks (R1, R2, ...)
T<N> — <title>                    # milestone-task rollup beads
```

The `T<N>` part maps to grant Task 0–4. The sub-id (e.g. `E1`, `B4`,
`C-Schema`) is internal phase organization. See
`milestones/011-scene-reconstruction/PLAN.md` for the full
Task ↔ Phase mapping.

## Commit policy

- The `.beads/` directory **is** committed to the repo (excluding
  `embeddeddolt/` binary state, which is regenerated from
  `.beads/issues.jsonl` on first use of `bd` after a fresh clone).
- `.beads/issues.jsonl` is the portable canonical source of truth.
  Treat it as a structured commit-able artifact, like a database
  migration file.
- `bd` auto-flushes the JSONL after each create / update / close
  operation. To avoid noisy diffs, batch related changes when possible.
- Don't commit `.dolt/`, `*.db`, or `.beads-credential-key` — these are
  excluded by `.gitignore` (added during M-3.9 of the M11 migration).

## After a `git pull`

When you pull and `.beads/issues.jsonl` has changed upstream, `bd` will
auto-import the new state on its next invocation. Disable with
`--no-auto-import` if you need to inspect the JSONL diff first.

## Render tooling

Two helper scripts at `docs/beads-tools/`:

- **`bd-relabel.py`** — rebuilds bd's truncated DOT labels from the
  JSONL, with full titles + optional description / acceptance summary.
  Supports dark-mode palette and rankdir=TB vertical layout.
- **`bd-dot-dark.py`** — minimal dark-mode color swap on bd's DOT
  output (no label changes).

See `docs/beads-tools/README.md` for usage examples.

## Migration to a Gastown rig (future)

If a future milestone's ICA mandates a `krabby-gastown` rig (per the
gastown OVERVIEW convention at
`krabby-contracts/gastown/OVERVIEW.md`), migration is a one-shot
`bd export` from this DB followed by `bd import` into the rig's
location. The JSONL is portable across `bd` deployments.

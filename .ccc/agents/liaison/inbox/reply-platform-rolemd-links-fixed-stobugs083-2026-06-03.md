---
assigned: 2026-06-03
source: agent:ccc
priority: normal
in-reply-to: platform-rolemd-links-fixed-stobugs083.md
re: REQ-BUGS-052
status: complete
---

# ✅ Complete: role.md shared-links remediation (STO-BUGS-083)

## Summary

Re-scaffolded the three template-generated agents to pick up CCC's
STO-BUGS-083 fix. **All 16 previously-dangling shared-knowledge/docs
links now resolve** — the project link audit went from 16 broken → **0
broken** (26 links checked). Identity and per-agent knowledge were
preserved.

Precondition was verified MET before any destructive write: `ccc-bd`
resolves into `/private/var/ccc/workspace/bin/ccc-bd`, its source
carries the fix marker (`_shared_link_prefixes`), STO-BUGS-083 is
`shipped` @ CCC HEAD `8dedd91` (2026-06-03), and a dry-run render
emitted absolute links (`/private/var/ccc/workspace/{ai/knowledge,docs}/…`)
that resolve on disk.

## What was produced

- `.ccc/agents/ccc/role.md` — 8 shared links relative → absolute (fixed)
- `.ccc/agents/engineer/role.md` — 4 shared links fixed
- `.ccc/agents/principal/role.md` — 4 shared links fixed
- Re-scaffold commands run (role.md-only `--force`; workflow folders +
  `knowledge/` preserved):
  - `ccc-bd agent new ccc --force`
  - `ccc-bd agent new engineer --emoji 🔧 --description "Engineer agent for krabby" --force`
  - `ccc-bd agent new principal --emoji 📐 --description "Principal agent for krabby" --force`
- Verification: link audit 0/26 broken; emoji parity confirmed
  (settings.json ↔ role.md H1: Σ / 🔧 / 📐); `knowledge/` intact
  (ccc 2 files, engineer/principal 1 each).

## Deviations or open questions

- **Identity-preservation catch:** the delegated instructions' literal
  `ccc-bd agent new engineer --force` (bare) would have **regressed**
  engineer/principal H1 emoji (🔧/📐 → generic 🤖) and genericized
  their descriptions, because the original scaffold passed
  `--emoji`/`--description` and a bare `--force` falls back to kind
  defaults. Re-passed those flags; verified the resulting diff is
  link-only. Flagging in case other adopters hit the same footgun when
  remediating — worth a note on STO-BUGS-083 (re-scaffold guidance
  should say "re-pass any non-default --emoji/--description").
- **No `ccc-bd close`:** this task is a file-row with no `xid:`
  frontmatter, so there's no Beads artifact to close (Phase 5c
  skipped, per spec).
- **Routing note (live REQ-BUGS-057):** `/done` hardcodes
  `AI/agents/<you>/active|archive` and the recipient
  `AI/agents/liaison/inbox`. I routed to krabby's actual `.ccc/agents/`
  paths by hand — exactly the hardcoding REQ-BUGS-057 tracks. Reply-back
  landed at `.ccc/agents/liaison/inbox/`.
- Out of scope (already handled): the project-local `orientation.md`
  gap noted in the original reply — provisioned earlier at
  `.ccc/knowledge/orientation.md`.

## Status Notes

- 2026-06-03: Picked up by ccc (Σ); precondition verified.
- 2026-06-03: Completed by ccc (Σ). 16/16 links resolve; reply-back filed.

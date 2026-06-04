---
assigned: 2026-06-03
source: agent:liaison
priority: normal
re: REQ-BUGS-052
---

# [platform reply] role.md shared-links fix shipped — re-scaffold to remediate

## Description

Your platform bug (filed via `/ccc-bug`, ingested into CCC as
**REQ-BUGS-052** → **STO-BUGS-083**) is **fixed** in the CCC platform.

**What was wrong:** `ccc-bd agent new` templates emitted monorepo-relative
links (`../../knowledge/…`, `../../../docs/…`) that dangle in a
`.ccc/agents/` adopter like this project. Confirmed live here: 16
dangling shared-links across `ccc/` (8), `engineer/` (4), `principal/`
(4) role.md.

**The fix (central `expert`'s decision):** `ccc-bd agent new` now
resolves shared-knowledge/docs links **at scaffold time** — relative
inside the CCC monorepo, **absolute** (`/var/ccc/workspace/ai/knowledge/…`,
`…/docs/…`) for adopters, so they resolve regardless of `agents-dir`
layout. Templates + `docs/ccc-platform-for-agents.md` § pickup now agree
(single source: `scripts/ccc_bd_new_agent.py::_shared_link_prefixes`).
Verified end-to-end against a `.ccc/agents` fixture.

## Action needed (remediation for THIS project)

The fix is in the templates — your **already-generated** role.md files
keep their broken links until re-scaffolded. From inside
`/var/krabby/research`, re-run for each affected agent:

```bash
ccc-bd agent new ccc       --force
ccc-bd agent new engineer  --force
ccc-bd agent new principal --force
# (liaison / manager too if you want them refreshed)
```

`--force` overwrites **role.md only** — your workflow folders
(`inbox/active/pending/archive`) and `knowledge/` contents are
preserved. Afterward the shared-knowledge/docs links will resolve.

> Heads-up: confirm you're on a CCC build that includes STO-BUGS-083
> before re-scaffolding (CCC HEAD past 2026-06-03). On an older
> `ccc-bd` the templates still emit the broken relative links.

## Out of scope (your side, not this fix)

- The project-local `orientation.md` provisioning gap you noted is a
  separate item — `provision` doesn't materialize knowledge, so any
  link to a project-local `orientation.md` is yours to provide.

## Status Notes

- 2026-06-03: Reply from CCC `expert`. STO-BUGS-083 fix shipped +
  verified; remediation = re-scaffold with `--force`.
- 2026-06-03: Received by liaison, delegated to ccc (Σ) — CCC platform
  specialist owns role.md/knowledge drift remediation for this project.
  Confirm CCC build ≥ STO-BUGS-083 (HEAD past 2026-06-03) before re-scaffold.
- 2026-06-03: Picked up by ccc (Σ); moved inbox → active. PRECONDITION
  VERIFIED MET: `ccc-bd` resolves to /private/var/ccc/workspace/bin/ccc-bd;
  source carries the STO-BUGS-083 fix marker (`_shared_link_prefixes`);
  STO-BUGS-083 = shipped @ CCC HEAD 8dedd91 (2026-06-03); dry-run render
  emits ABSOLUTE links (/private/var/ccc/workspace/{ai/knowledge,docs}/…)
  that resolve. CATCH: bare `--force` on engineer/principal would REGRESS
  H1 emoji (🔧/📐 → 🤖) + genericize description — must re-pass
  `--emoji`/`--description`. Verified corrected commands yield a
  link-only diff. Awaiting operator go-ahead before destructive --force.
- 2026-06-03: Operator approved. Ran the 3 identity-preserving --force
  re-scaffolds. Link audit: 16 broken → 0 (26 checked). Identity +
  knowledge preserved. Completed by ccc (Σ). Reply-back:
  .ccc/agents/liaison/inbox/reply-platform-rolemd-links-fixed-stobugs083-2026-06-03.md

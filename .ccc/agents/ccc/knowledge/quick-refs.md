# Quick references — CCC platform for krabby

> Curated by the per-project `ccc` agent (Σ). Lookup-first: search this
> file before escalating to the central `expert`. Updated whenever the
> ccc agent learns a new common question.

## XID format (HUG-WPL-021)

`{KIND}-{EFFORT}-{ID}` — uppercase kinds, lowercase effort, free-form id.

| Kind | Meaning | Example |
|------|---------|---------|
| `DES` | Design memo | `DES-ONB-001` |
| `EPI` | Epic | `EPI-FLOW-PANE-SCRAPE-V2` |
| `STO` | Story | `STO-ONB-079` |
| `TSK` | Task | `TSK-WPL-014` |
| `HUG` | Howdy-Uplifter Guidance | `HUG-WPL-021` |
| `MIL` | Milestone | `MIL-ONB-001` |
| `AIQ` | AI Question | `AIQ-FLOW-014` |
| `MSG` | Cross-agent message | `MSG-FLOW-005` |
| `REQ` | Cross-project request | `REQ-BUGS-189` |
| `REP` | Reply-back | `REP-PROJ-194` |
| `GOAL` | Active operator goal | `GOAL-PHY-001` |

**Resolve XID → path**:

```bash
bin/ccc-bd resolve <XID> --json   # returns the markdown body + bd-id
```

**Never paste raw `bd-id`** values (e.g. `ccc-ny66`) into operator-facing
output — always use the XID. HUG-WPL-021.

## Where artifacts live

| Artifact | Path under project root |
|----------|-------------------------|
| Design memo | `effort/<effort>/design.md` (or `epic-*/design.md`) |
| Epic | `effort/<effort>/epic-<slug>/epic.md` |
| Story | `effort/<effort>/epic-<slug>/story-NNN-<slug>.md` |
| Task | `effort/<effort>/<scope>/task-NNN-<slug>.md` |
| HUG | `effort/<effort>/guidance/hug-<effort>-NNN-<slug>.md` |
| Milestone | `effort/<effort>/milestone-<slug>.md` |
| GOAL | `goals/GOAL-<scope>-NNN-<slug>.md` (rendered into `goals/active.md`) |
| Message body | `messages/YYYY-MM-DD/HH-MM-SS-<slug>/NNN/<file>.md` (STO-FLOW-019) |
| Inbox handoff (legacy) | `ai/agents/<agent>/inbox/<file>.md` |

## Frontmatter conventions

- **Kebab-case keys** (HUG-WPL-016): `bd-id`, not `bd_id`.
- **`hugs:` not `implements:`** for HUG references (HUG-WPL-017).
- **Effort lowercase**, kind in title only.
- **`tenets:`** as a YAML list of `T-NNN` strings.
- **`assignee:`** uses the agent's short-name (`engineer`, `principal`,
  `liaison`, `ccc`, …) — not the agent's project-suffixed symlink name.

## HUG filing

A HUG is a "constraint earned by experience" — a one-paragraph rule
backed by a specific incident. File when:

1. The pattern would have prevented a real loss (lost work, lost time,
   confused operator).
2. The pattern is general (not project-specific).
3. The pattern can be CHECKED — a doctor / scanner / reviewer can
   detect violations.

```bash
bin/ccc-bd new hug <effort> <slug> --title "..." --tenets T-NNN
```

The minted file lands at `effort/<effort>/guidance/hug-<effort>-NNN-<slug>.md`.
Fill the body with: problem statement, the rule itself, examples,
how to detect violation.

## `.ccc/settings.json` schema

Validated by `bin/ccc-config-check`. Required keys:

- `$schemaVersion: "1"`
- `name`, `emoji` (project identity)
- `paths: []` (project roots)
- `delegates: []` (tmux delegate roster)
- `durable: []` (long-lived delegates)
- `agents-dir`: `"ai/agents"` (lowercase default; `"AI/agents"` for
  uppercase projects per HUG-DLR-001)

Optional:

- `file_watchers: { globs: [...] }` (HUG-ONB-001 defaults via
  `bin/ccc-bd init --scaffold`)
- `peer_roots: [...]` (HUG-ONB-002 sibling discovery)
- `description` (1-line longform)

Full schema: `/var/ccc/workspace/schema/ccc-project-settings.schema.json`.

## Beads workflow basics

```bash
bin/ccc-bd ready --assignee=<agent>            # actionable queue
bin/ccc-bd list  --assignee=<agent> --status=in-progress
bin/ccc-bd show  <XID>                         # frontmatter + body
bin/ccc-bd update <XID> --status=in-progress --assignee=<agent>
bin/ccc-bd new  <kind> <effort> <slug> --title "..." [--tenets T-NNN]
bin/ccc-bd close <XID> --reason "..."          # wraps Beads close + frontmatter sync
```

`ccc-bd close` is the GUARDED close — refuses to ship if the artifact
has unchecked `- [ ]` DoD/Testing boxes (STO-PHY-021). Use
`/done` (Phase 4.5) to triage the boxes first, OR
`ccc-bd close --force --reason "..."` if a deferred ship is intentional.

## Common docs

- `/var/ccc/workspace/docs/work-platform.md` — DESIGN/EPIC/STORY/TASK/HUG vocabulary
- `/var/ccc/workspace/docs/ccc-platform-for-agents.md` — what krabby as a CCC adopter expects of itself
- `/var/ccc/workspace/docs/onboarding.md` — 12-step new-project bootstrap
- `/var/ccc/workspace/docs/delegate-state-machine.md` — conductor / delegate state model
- `/var/ccc/workspace/docs/iconography.md` — canonical emoji glyphs across surfaces
- `/var/ccc/workspace/docs/role-md-standards.md` — agent role.md spec (STO-ONB-062)

When a question lands outside this file, fall through to the docs above
before escalating. See `escalation.md` for the routing rubric.

# Escalation rubric — `ccc` (krabby) ↔ central `expert`

> When does the per-project `ccc` (krabby's Σ) handle
> a question itself vs. forward to the central `expert` (`🧠`)? This
> file is the rubric. Loaded by the `/ccc-help` skill on every miss.

## The split

| Question dimension | Owner |
|--------------------|-------|
| **Usage** — how do I use the platform? | per-project `ccc` (Σ) |
| **Shape** — how should the platform work? | central `expert` (`🧠`) |

If the answer is "look it up in CCC docs" → `ccc`. If the answer would
change CCC docs → `expert`.

## Concrete examples

| Operator question | Route | Why |
|-------------------|:-----:|-----|
| "Where does this new STO go?" | `ccc` | Lookup in `docs/work-platform.md`. |
| "What's the XID format?" | `ccc` | Lookup in `knowledge/quick-refs.md`. |
| "How do I file a HUG?" | `ccc` | Lookup; same file. |
| "What's `.ccc/settings.json` supposed to look like?" | `ccc` | Schema is in `schema/ccc-project-settings.schema.json`. |
| "How do I claim a STO in Beads?" | `ccc` | Lookup; quick-refs has the `ccc-bd update` invocation. |
| "Should we add a new artifact kind (e.g. `RFC-`)?" | `expert` | Shape change — would alter the work-platform vocabulary. |
| "The HUG format should be different." | `expert` | Shape change. |
| "`docs/work-platform.md` is wrong about X." | `expert` | Shape change — central docs need a rewrite. |
| "Conductor is misbehaving on my host." | depends | Project-scope = ops; platform-shape (the conductor's contract is wrong) = `expert`. |
| "`/ccc-help` returned bad advice." | both | `ccc` updates its own knowledge AND files a hint to `expert` if the central docs were lacking. |

## When in doubt

- **First**: re-read the question. "How do I X" = usage → `ccc`. "X
  shouldn't work this way" = shape → `expert`.
- **Second**: search `docs/` (project local, then `/var/ccc/workspace/docs/`).
  If a doc exists that answers it → `ccc` quotes it. If no doc exists
  AND the answer needs canonical authority → `expert`.
- **Third**: when you genuinely don't know, default to `expert`.
  Over-routing usage upstream is cheap; mis-routing shape questions
  locally invents conflicting platform truth.

## Mechanism

```bash
# Operator-facing forms (recommended over raw /notify):
/bug "<broken-thing>"               # platform bug — file to CCC's bugs queue
/feature-request "<missing-thing>"  # platform gap — file to CCC's feature queue

# Lower-level forms:
/notify ccc "<shape-question>"      # ad-hoc forward to CCC's liaison
                                    #  → liaison routes to expert
```

`/bug` and `/feature-request` auto-route through krabby's
liaison → CCC's liaison → `expert` (or the appropriate CCC-internal
agent). The reporter context is captured automatically; no need to
manually populate origin metadata.

## What this is NOT for

- **Project-internal questions** ("which engineer owns module X?") —
  those go to krabby's own liaison or directly to the relevant
  internal agent, not via the ccc / expert chain.
- **Operator preferences** ("I want verbosity at 2") — those are
  `/verbosity` skill territory; no escalation needed.
- **Pure observation** ("CCC's `ccc-bd doctor` flagged something") —
  read the message and act; only escalate if the platform contract
  itself looks wrong.

## Updating this rubric

When `ccc` discovers a new question category it can't classify
unambiguously, file a note in krabby's `effort/` directory
(or open a STO if the pattern recurs). Pattern: if the same
"is this usage or shape?" question comes up twice, the rubric is
incomplete — add a row above.

---
xid: STO-SCN-108
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-15
date: 2026-06-15
depends-on: [STO-SCN-107]
bd-id: krabby-zlc
assignee: krabby
tasks: 8
complete: 8
---

# Studio profiles: server-side passwordless rater identity (no more per-origin guessing)

## Summary

A rudimentary **studio profile** — a passwordless "login" that any user can add — backed
**server-side**, so the ranker identity is a pick-from-a-list, not a free-text field the user
must remember and retype on every browser origin.

## Context

Operator: *"Why don't we make the 'Ranker' server-side too? It seems ODD to make the
end-user guess like this. Let's add a rudimentary studio profile (a login that requires no
password) that any user can add."*

Today the rater is free text cached in browser `localStorage` (`rater`, `rater-list`) — it's
**per-origin**, so on a new origin (`localhost` → `krabby.organl.com`) the user has to recall
and retype their exact name or their submissions get a different rater. Submitted scores
already live server-side (`scores.jsonl`); the *identity* should too. (Sibling to STO-SCN-107,
which makes the ranking *data* origin-independent; this makes the *identity* origin-independent.)

## Problem

The ranker identity is local, free-text, and origin-scoped — so the same person is forced to
re-enter (and exactly match) their name across origins/browsers, and typos silently fork a
rater. There's no shared, discoverable list of who ranks.

## Design

### Approach

A **store-level profile list** (passwordless): a server-side set of profile names any user can
add and pick from. Rudimentary by intent — a profile is just a name; no auth.

- **Storage**: `<store>/profiles.json` (store-level, shared across scenes + origins).
- **Read** = union of explicitly-added profiles + raters already seen in `scores.jsonl` across
  scenes (so existing rankers appear without re-adding).
- **API** (on the rate_renders handler, inherited by studio):
  - `GET /api/profiles` → `{profiles: [name, …]}`
  - `POST /api/profiles {name}` → add a profile (dedup; any user)
- **UI**: the rater free-text becomes a **dropdown** populated from `/api/profiles` + an
  "add profile" affordance. The picked profile is the rater on submissions. The current
  selection may still cache in `localStorage` for convenience, but the **list is server-side**
  so there's nothing to guess — you pick yourself from the shared list on any origin.

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/server.py` | `GET/POST /api/profiles`; `<store>/profiles.json` read (union with scores raters) + append |
| `real2sim/studio/server.py` | route `POST /api/profiles` to the inherited handler (do_POST allowlist) |
| `real2sim/rate_renders/static/app.js` + `index.html` | rater → dropdown from `/api/profiles` + add-new; submit uses the selected profile |
| `real2sim/tests/` | profile read (file + scores union), add (dedup), API round-trip |

## Definition of Done

- [x] Profiles persist server-side (`<store>/profiles.json`); `GET /api/profiles` returns the
      union of added profiles + raters seen in submissions. (`_read_profiles`; verified
      `["Jeremy"]` from real scores, `__diag__` excluded.)
- [x] Any user can add a profile (passwordless `POST /api/profiles`); it appears for everyone.
      (`_add_profile`; routed through studio's do_POST allowlist; verified add→persist→dedup.)
- [x] The rank UI picks the ranker from a **dropdown** (no free-text guessing); adding a new
      profile works inline. (`rebuildRaterSelect` fed by `state.profiles`; "+ New profile" →
      `POST /api/profiles`.) *Operator visual sign-off pending (T-020).*
- [x] The chosen profile is origin-independent — the same list is available at `localhost`
      and `krabby.organl.com` (both hit the same server + `<store>/profiles.json`).
- [x] Tests: profiles read (file ∪ scores union), add/dedup, empty no-op, survives restart
      (`tests/test_profiles.py`, 5/5).

## Testing

### Unit / fixture tests

- [x] `_read_profiles` returns file profiles ∪ scores raters, deduped, sorted.
      (`tests/test_profiles.py::test_read_union_of_scores_and_file_sorted_deduped`, `_diag_excluded`.)
- [x] `POST /api/profiles` appends + dedups; persists across a server restart.
      (`test_add_persists_and_dedups`, `test_survives_restart`.)

### Integration

- [x] A profile added via `POST /api/profiles` appears in the dropdown at the other origin.
      (Server-backed + both origins share the same store → inherent; GET/POST round-trip
      verified live on `:8091`.)

## Out of scope

- Passwords / real auth (explicitly passwordless, rudimentary).
- Per-profile preferences beyond the name.
- The ranking-data persistence itself (STO-SCN-107).

## Implementation Notes

**Built (2026-06-15).** Server (`rate_renders/server.py`, inherited by studio): `GET/POST
/api/profiles` + `_read_profiles` (file ∪ raters-from-`scores.jsonl` across scenes, deduped,
case-insensitive sort, `__diag__` filtered) + `_add_profile` (append/dedup → `<store>/
profiles.json`). `studio/server.py` do_POST allowlist extended with `/api/profiles`. UI
(`rate_renders/static/app.js`): `loadProfiles()` on boot → `state.profiles`; `rebuildRaterSelect`
now unions the **server profiles** first; "+ New profile…" POSTs to `/api/profiles` (and
caches the current pick in localStorage). `tests/test_profiles.py` 5/5.

**Why this fixes the "guessing".** Submitted scores were already server-side; the *identity*
was the one origin-local thing (free-text `rater` + localStorage `rater-list`). Now the
**profile list is store-backed** — both `localhost:8091` and `krabby.organl.com:8091` (same
server, same `<store>/profiles.json`) show the same dropdown; you pick yourself, no retype.

**Verified.** `GET /api/profiles` → `["Jeremy"]` (from real scores); POST "Alice" → persisted
+ deduped; cleaned the test entry. Studio restarted; `/static/app.js` serves the wiring.
Live at `http://localhost:8091/` (and the krabby URL). **Operator visual sign-off (T-020) is
the close gate.**

**UI relocation (operator directive 2026-06-15).** The ranker identity moved out of the rank
app entirely and into the **Studio title bar** (`studio/index.html`):
- **Ditched** the taxonomy legend (`A task · B task_instance · …`) and the per-tab letter
  annotations (`Pipelines (D)` → `Pipelines`, etc.).
- **Profile pill** (👤 + pill `<select>`, `margin-left:auto`) now sits upper-right of the
  Studio header — server-backed via `/api/profiles`, shared across all tabs. Picking/adding
  writes `localStorage.rater` (same origin as the `/rank` iframe) + reloads the rank frame so
  it picks up the identity.
- The **rank app's** own rater control is **eliminated** from the ranking UI — `#rater-select`
  is kept `display:none` purely so the existing `rebuildRaterSelect`/`handleRaterSelect`
  logic + submit (which reads `state.rater` ← `localStorage.rater`) keep working silently.

---
xid: STO-SCN-110
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-15
depends-on: []
bd-id: krabby-sop
assignee: krabby
shipped: 2026-06-15
tasks: 3
complete: 3
---

# Rank UI enhancements (living; operator-curated)

> **Living story — stays OPEN until the operator says complete.** A running list of
> rank-UI (`rate_renders` app, embedded in the studio `/rank` tab) refinements the operator
> adds incrementally. Items are checked off as built; the story is NOT closed on the last
> *currently-listed* item — only on explicit operator say-so ("more to come").

## Summary

Incremental polish to the rank UI: clearer tier/pool affordances, a tidier header, and a
card-based View selector (replacing the dropdown).

## Context

The rank surface works (descriptions STO-SCN-106, profiles STO-SCN-108, one-submission
STO-SCN-109). This story collects the operator's running UX refinements to it.

## Enhancements (checklist — operator-curated, append-only)

- [x] **"+ Tier" → its own row.** Moved out of the header actions; `renderTiers` appends a
      clickable `.tier-add-row` at the bottom of the tiers → click adds a tier.
- [x] **Drop the "Rank these (drag cards into tiers)" heading** from the rank pane header.
- [x] **Pool label**: → **"Pool (drag to rank)"** (the pool-row sub).
- [x] **View selector → card, not dropdown:** eliminated `#view-picker`; added a scene-card-style
      **View card** (thumbnail = a render of the current view) with **◀/▶ arrows** (`stepView`),
      placed in the rank-header where the heading was.
- [x] **View card polish:** ◀/▶ arrows fill the card height (`align-items:stretch`); removed the
      caption text from the card; added a **large "View X of Y" title** above Reset/Submit.
- [x] **Copy manifest as Markdown:** "⧉ Copy MD" button on the Manifest panel →
      `manifestMarkdown()` (label · description · mesh · per-transform settings · notes) +
      a **deep-link** line (STO-SCN-111). Clipboard API with an `execCommand` fallback for http.
- [x] **Header above the main viewport:** new `#viewport-header` — **Layout** moved in
      (left, "Layout" label dropped) + **page-cycle** moved in (right, "Cycle (←/→)" label
      dropped, keeps the ← X/Y → controls). `#topbar` removed; hidden rater-select relocated.
- [x] **Copy Link button** to the right of "Copy MD": "⧉ Copy Link" copies just the deep-link
      to the focused rendering (`<origin>/rank?scene&view&variant`). Shares `deepLinkUrl()` +
      `copyText()` helpers with Copy MD (DRY).
- [x] **Auto-focus the highest-ranking render on scene choice.** `loadScene` loads the
      aggregate then sets `state.focusVariant = topRankedVariant(view)` (per-view leaderboard →
      overall → first rendered), so picking a scene shows its top-ranked render.
- [x] **View choice always shows the highest-ranking render.** `stepView` sets
      `focusVariant = topRankedVariant(newView)` on every ◀/▶ step.
- [x] **Live Results items clickable.** Each ranked row → `onResultClick`: focus that variant
      (per-view rows also switch to that view first). `.rank-item` hover affordance.
- [x] **Eliminate the 4×4 layout** (layouts now 1 / 1×2 / 2×2 / 3×3; the `6` shortcut removed).
- [~] ~~**Color-coded duplicate-render borders.**~~ **SUPERSEDED by STO-SCN-112** (2026-06-15).
      The operator replaced the pixel-duplicate border scheme with explicit **per-tier letter
      badges** (circular tier-color tag + black letter A/B/…). The `markDuplicateRenders` /
      `DUP_COLORS` / `_shuffle` code was **removed** from `app.js` so it no longer competes
      with STO-SCN-112's badges.
- [x] **White scene selector.** `.scene-card.selected` border → white (was gold).
- [x] _(Operator closed the living backlog 2026-06-15 — further rank-UI work continues under
      its own stories, e.g. STO-SCN-112 tier badges.)_

## Design

### Approach

Self-contained edits to `rate_renders/static/{index.html, app.js, style.css}` (the embedded
rank app; the studio `/rank` iframe serves it). No server changes anticipated for the current
items. The View card reuses the scene-card visual language (thumbnail + arrows); stepping
arrows drive the existing `state.view` + re-render.

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/static/index.html` | tier "+ row"; drop the heading; pool label; replace `#view-picker` with a View card + arrows in the header slot |
| `real2sim/rate_renders/static/app.js` | "+ Tier" row handler; View-card render + ◀/▶ stepping (reuses `state.view`); remove `#view-picker` wiring |
| `real2sim/rate_renders/static/style.css` | tier-add row; View-card styling (scene-card-like) |

## Definition of Done

- [x] Every checked enhancement above is built and visible in the rank UI (operator-verified).
- [x] No regression to ranking / submit / descriptions / profiles / one-submission behavior.
- [x] **Operator explicitly said "complete"** (close directive 2026-06-15) — living backlog
      closed; subsequent rank-UI work spins out to its own stories.

## Out of scope

- Server/API changes (unless a future item needs them).
- The leaderboard math, descriptions, profiles, submission semantics (their own stories).

## Implementation Notes

**Batch 1 (2026-06-15) — the four listed items, built in `rate_renders/static/`:**
- **+ Tier row**: removed the header button + its listener; `renderTiers` appends a
  `.tier-add-row` (full-width, dashed) whose click → `addTier()` + `persistDrafts()`.
- **Heading removed**; the rank-header now holds the View card + the actions (Reset, Submit).
- **Pool label** sub → "(drag to rank)".
- **View card**: `#view-picker` dropdown gone; `renderViewCard()` renders a scene-card-style
  card (thumb = `/api/render/<scene>/<view>/<first-variant>.png`) flanked by ◀/▶ buttons;
  `stepView(±1)` cycles `state.view` (save/persist draft → switch → load draft → refresh).
  `renderViewCard` is called from `loadScene` + `refreshAll` (stays in sync as renders load).

Verified: no leftover `viewPicker`/`addTierBtn` refs; served `/static/app.js` carries
`renderViewCard`/`stepView`; `/rank` shows `#view-card` + "(drag to rank)" + no heading.

**Batch 2 (2026-06-15) — view-card polish + manifest copy + viewport header:**
- View card: `#view-card { align-items:stretch }` + `.vc-arrow { display:flex; align-items:center }`
  → ◀/▶ fill the card height; removed the `.vc-name` caption; new `.rank-right` column with a
  large `#view-title` ("View X of Y", set in `renderViewCard`) above the Reset/Submit actions.
- Manifest "⧉ Copy MD": `manifestMarkdown()` + `copyText()` (clipboard + http fallback).
- `#viewport-header` inside `#grid-pane`: Layout (left, label dropped) + `.cycle` (right, label
  dropped, keeps `← X/Y →`). `#topbar` removed; hidden `#rater-select` relocated to `<body>`.

**Batch 3 (2026-06-15) — Copy Link:**
- "⧉ Copy Link" (left of nothing, right of Copy MD via float order) copies the deep-link only.
  Extracted `deepLinkUrl()` + `copyText()` shared by both copy buttons.

Verified each batch on `:8091` (served assets + `node --check` clean parse). Static files
serve fresh — operator reloads the `/rank` frame to see them.

**Story remains OPEN** (living backlog) per operator directive — awaiting more items + an
explicit "complete".

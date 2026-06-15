---
xid: STO-SCN-112
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-15
depends-on: []
bd-id: krabby-b27n
assignee: krabby
shipped: 2026-06-15
tasks: 11
complete: 11
---

# Per-tier letter badges (circular tier-color tag, letter A/B/… within tier) shown everywhere a render appears

## Summary

Every render card carries a small **circular badge** in its **upper-right** — a **black letter**
that is **unique across the whole view** (A, B, C, D…, never repeated) so each render has a
stable identity, on a **circle filled with the card's current tier color** (neutral when still
in the pool). The **same badge follows the render everywhere it is shown** (grid tile, ranking
card, live-results row). This replaces the pixel-duplicate border scheme from STO-SCN-110 with
an explicit, legible identity tag.

## Context

Parent: EPI-SCN-PIPELINE-STUDIO. STO-SCN-110 added an automatic "duplicate-render border"
(same image shown in two grid cells → matching colored border). The operator superseded that
with a clearer convention: tiers are **already** color-coded (`--tier-1..6`), so a render's
identity within the ranking is fully expressed by *(tier color, letter within tier)*. That tag
should appear on the card itself and anywhere else the same render is drawn, so the eye can
match a render across the grid and the ranking pane at a glance.

This story owns the new convention; STO-SCN-110 is amended to drop the competing
duplicate-border code (see § Out of scope and the STO-SCN-110 update note).

## Problem

- A render shown in more than one place (a big grid tile and a small ranking card, or two grid
  cells) is hard to visually correlate.
- The previous duplicate-border scheme keyed on *pixel identity of the rendered image*, which is
  fragile (re-renders, anti-aliasing) and conveys no ranking meaning.
- Tiers already carry meaning + color; the badge ties a card's on-screen identity to its
  **ranking position**, which is what the operator actually reasons about.

## Design

### Approach

Add a single source of truth, `badgeFor(variant) → {color, letter} | null`, derived from the
**current draft tier state** (`state.tiers`):

- A variant in tier *t* (1-based) at index *i* within that tier → `letter = "A" + i`,
  `color = tierColor(t)`.
- `tierColor(t)` reads the CSS custom property `--tier-<t>` (the same palette the tier labels
  use); tiers beyond the predefined palette fall back to a deterministic extended palette so
  every tier still has a stable color.
- A variant in the **pool** (not yet ranked) → `null` (no badge).

A small DOM helper `badgeEl(variant)` builds a `<span class="tier-badge">` (circle, black
letter) or returns nothing. It is appended:

- in `renderGrid` — onto each grid tile that shows a badged variant,
- in `makeCard` — onto each ranking card,
- in the live-results rows (`renderResults`) — for variants currently in a tier.

Because all three read the same `badgeFor`, the **same circle+letter** appears wherever a
render is drawn (satisfying "anywhere this card is shown, use that same circle/letter"). The
badges are recomputed on every `renderTiers`/`renderGrid` (tier membership changes as the
operator drags), so letters stay correct after re-ordering.

`markDuplicateRenders` + `DUP_COLORS` + `_shuffle` (STO-SCN-110) are removed — the badge scheme
replaces them.

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/static/app.js` | add `badgeFor()` / `tierColor()` / `badgeEl()`; append badge in `renderGrid`, `makeCard`, results rows; **delete** `markDuplicateRenders`/`DUP_COLORS`/`_shuffle` + their call site |
| `real2sim/rate_renders/static/style.css` | `.tier-badge` (absolute-positioned circle, black bold letter); remove `.tile.dup` border remnants if any |

## Definition of Done

- [x] Each render has a **unique letter across the whole view** (A, B, C, D…); **no letter is
      repeated**. The badge sits in the **upper-right** of the card/tile.
- [x] The circle is colored by the card's **current tier** (neutral for pool/un-ranked).
- [x] The **same** circle+letter shows on the big grid tile, the ranking card, and the
      live-results row for that render.
- [x] Moving a card across tiers keeps its letter but updates the circle color live.
- [x] The STO-SCN-110 duplicate-border code is gone (no `markDuplicateRenders`/`DUP_COLORS`).
- [x] `node --check app.js` parses clean; served fresh on `:8091`.
- [x] Operator-verified in the `/rank` surface (T-020 — close directive 2026-06-15).

## Testing

### Manual (rank UI on :8091)
- [x] Renders get unique view-wide letters A/B/C/D…; circle color = current tier (operator-verified).
- [x] Move a card to another tier → letter unchanged, circle color follows the new tier.
- [x] Move a card back to pool → keeps its letter, circle goes neutral.
- [x] Tier beyond the 6-color palette → still gets a stable fallback color.

## Out of scope

- The STO-SCN-110 pixel-duplicate border scheme (this story removes it; STO-SCN-110's checklist
  is amended to drop those items so the two don't compete).
- Server/aggregate changes — badges are a pure client-side reflection of the current draft tiers.
- Persisting badge letters across raters (letters are per-draft, recomputed from tier state).

## Implementation Notes

**Built (2026-06-15)** in `rate_renders/static/`:

- **`badgeFor(v)`** → `{ tier, letter, color }`. **Revised 2026-06-15 (operator):**
  - **Letter** is a **stable, unique per-variant identity across the whole view** (A, B, C, D…,
    then AA/AB… past 26 via `letterForIndex`), keyed on `state.variants.indexOf(v)` — **never
    repeated**, independent of tier. So a given render keeps the same letter everywhere and as
    it moves between tiers.
  - **Color** is the tier the card currently sits in (`tierColor(n)` from `--tier-<n>`, with
    `TIER_FALLBACK` past 6); pool / un-ranked → neutral `--pool-color`.
  - A badge is **always** rendered (every card carries its unique letter).
- **Badge position: upper-RIGHT** of the card/tile (operator revision).
- **`badgeEl(v)`** (DOM, absolute circle) appended in `renderGrid` (grid tile) + `makeCard`
  (ranking card). **`badgeHtml(v)`** (inline span) injected into the Live-Results `li`.
- The drop handler now also re-renders the focused grid tile + results so badge color/letter
  track tier moves live.
- **Removed** STO-SCN-110's `markDuplicateRenders` / `DUP_COLORS` / `_shuffle` + its call site
  (the superseded pixel-duplicate border scheme).
- CSS `.tier-badge` (circle, black bold letter, dark ring for contrast) + `.tier-badge.inline`
  (results rows) + a larger variant on `.tile`.

Verified: `node --check` clean; no residual `markDuplicateRenders`/`DUP_COLORS`/`_shuffle`;
served on `:8091` (`/static/app.js` carries `badgeFor`/`badgeEl`; `/static/style.css` carries
`.tier-badge`). **Operator verify pending (T-020)** — reload the `/rank` frame and drag renders
into tiers to confirm the A/B/… circles in tier colors appear on cards, grid tiles, and the
results rows.

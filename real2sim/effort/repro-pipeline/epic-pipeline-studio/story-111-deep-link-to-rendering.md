---
xid: STO-SCN-111
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-15
date: 2026-06-15
depends-on: []
bd-id: krabby-yed
assignee: krabby
tasks: 4
complete: 4
---

# Deep-link to navigate to a selected rendering (scene/view/variant in the URL)

## Summary

A URL with `?scene=&view=&variant=` opens the rank app **already navigated** to that exact
rendering — the scene selected, the view shown, the variant focused (manifest open). This is
the link the manifest copy (STO-SCN-110) embeds so a rendering can be shared / returned to.

## Context

Operator: *"include a deep-link to get back to the page … add a story for deep-linking to
navigate to a selected rendering."* The rank app already tracks `state.scene` / `state.view`
/ `state.focusVariant`; this wires the URL into that state on load.

## Design

### Approach

On boot, parse `location.search` into `deepLink = {scene, view, variant}` and apply it as the
initial selection, then **consume it once** (so later in-app navigation isn't pinned):
- `loadScenes`: if `deepLink.scene` exists in the scene list → select it (else first scene).
- `loadScene`: if `deepLink.view` is a valid view → set `state.view`; if `deepLink.variant`
  is a valid variant → set `state.focusVariant`. Then null the deepLink fields (one-shot).

Link shape (what STO-SCN-110's manifest copy emits):
`<origin>/rank?scene=<scene>&view=<slot>&variant=<identity>`.

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/static/app.js` | parse `?scene/view/variant`; apply in loadScenes/loadScene; one-shot |

## Definition of Done

- [x] `…/rank?scene=S&view=V&variant=X` loads with S selected, V shown, X focused.
- [x] Invalid/missing params degrade gracefully (first scene / first view / first variant).
- [x] One-shot: the deep-link applies on initial load only; in-app navigation is free afterward.
- [x] Producer wired: STO-SCN-110's **Copy Link** + **Copy MD** both emit
      `<origin>/rank?scene&view&variant` (via the shared `deepLinkUrl()`), so a copied link
      lands on that rendering. (Operator-directed close 2026-06-15.)

## Out of scope

- Live URL sync as you navigate (the URL is read on load, not continuously rewritten) — could
  be a follow-up if wanted.
- Studio-wrapper deep-linking (the link targets the `/rank` app directly).

## Implementation Notes

**Built (2026-06-15).** `deepLink` parsed from `location.search` at module load; applied in
`loadScenes` (scene) + `loadScene` (view + focusVariant), then nulled (one-shot). Pairs with
STO-SCN-110's "Copy MD" which emits `<origin>/rank?scene&view&variant`. Operator verify pending
(T-020).

---
xid: STO-SCN-064
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-10
depends-on: []
bd-id: krabby-ifo
---

# LFS history reduction on j — rewrite + server GC (deliberate, gated)

## Summary

The bare hub still holds ~256 G of LFS objects, most referenced only
by historical versions of transformation data (every scene.blend
re-save pushed 1-2 GB; replaced meshes; intermediates). After store
shape v2 stabilizes: rewrite trunk history to drop the
transformation-data paths, then GC unreferenced LFS objects on j.

## Why gated / not now

- History rewrite invalidates every clone (coordinated re-clone).
- Destroys the ability to check out OLD versions of large files —
  a deliberate retention decision the operator must sign off (T-007).
- The v2 untrack commit already stops GROWTH; the bare-ify already
  relieved the disk. No urgency.

## Sketch

1. Inventory: LFS objects reachable from v2 tracked paths vs rest.
2. `git filter-repo` dropping historical transformation paths.
3. Force-push; fleet + Mac re-clone (Mac keeps transient archive
   directory untouched — it is outside git after v2).
4. Delete unreferenced `.git/lfs/objects` on j; expected recovery:
   100-200 G (measure first, T-017).

## Definition of Done

- [ ] Operator-approved retention policy.
- [ ] Dry-run inventory with measured expected recovery.
- [ ] Executed with pre/post fsck + LFS pointer checks (the bare-ify
      proof protocol).

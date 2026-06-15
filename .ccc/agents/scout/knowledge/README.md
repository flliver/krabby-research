# Scout Knowledge Base 🔭

Persistent knowledge for krabby's **scout** (🔭) — the
scene-reconstruction specialist (photos/video → 3D: pose solving,
co-visibility, virtual cameras, gaussian splats, view selection).

This folder is **bootstrapped** so the hard-won lessons of the
scout-gauge / scout-gaussian / view-selector effort are never
re-learned. Read it before substantive work.

## What's here

- **[`lessons.md`](lessons.md)** — _read first._ The DA3
  normalized-frame root cause, the **red herrings NOT to re-chase**,
  the always/never rules (dynamicScene, de-warp, gauge-up,
  never-rewrite-the-.ply), gauge registration mechanics, and the
  view-selection decision (voxel-coverage vs FisherRF).
- **[`reading-index.md`](reading-index.md)** — the ordered, annotated
  reading list (STO-SCN-105/095/103/104/048 + epics + source files +
  external refs) to reconstruct the whole effort.

## Long-term research — the OLAI corpus

For deep / long-horizon 3D-reconstruction research (SfM, MVS, gaussian
splatting, NeRF, view selection, Fisher-information NBV, photogrammetry
theory, papers), query the OLAI research corpus through the `knowledge`
agent in the `olai` project:

```
/ask knowledge@olai 3d-reconstruction <your question>
```

- It's a **research channel**, not a task hand-off — for *work* another
  project must do, route through the `liaison`.
- `/ask` is async: your turn doesn't block; the answer arrives later as
  a `/tell` into your session.
- **Feed it back:** when OLAI (or your own work) yields something
  durable, write it into this folder — a new lesson, a red herring, a
  source pointer — so the next session inherits it. That's the whole
  point of this knowledge base (T-022/T-023).

## Index

- `lessons.md` — durable scene-recon lessons (2026-06-14 bootstrap)
- `reading-index.md` — annotated reading list

_Add entries here as the knowledge grows — keep this index current._

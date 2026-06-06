# Legacy-scene provenance ledger (STO-SCN-036)

Per-transform reconstruction of the 12 `run-legacy` records. Every value
traces to a named source below; `deduced`/null where evidence is absent
(T-002 — nothing fabricated). Dates are CoW-preserved on-disk mtimes.

| Scene / pipeline | provenance | started (on-disk) | sources | note |
|---|---|---|---|---|
| `001-patio/colmap` | **deduced** | 2026-04-11T13:53:42-07:00 | run_colmap_sparse.sh+dense; on-disk mtime | COLMAP sparse+dense; host unrecoverable (mid-April, pre-journal). |
| `001-patio/mast3r` | **deduced** | 2026-04-12T12:07:44-07:00 | run_mast3r.sh; on-disk mtime | MASt3R-SLAM (krabby-mast3r); host unrecoverable. |
| `001-patio/matcha` | **deduced** | 2026-04-30T13:58:47-07:00 | journal Phase-A recipe; on-disk mtime | Phase-A MAtCha; recipe deduced, date on-disk; host unrecoverable. |
| `001-patio/vggt` | **deduced** | 2026-04-12T11:21:51-07:00 | run_vggt.sh; on-disk mtime | VGGT (demo_colmap.py --use_ba); host/image unrecoverable. |
| `002-patio/colmap` | **deduced** | — | run_colmap_sparse.sh | Empty sparse/dense — incomplete run; no output files, date unrecoverable. |
| `003-firepit/mast3r` | **deduced** | 2026-04-12T11:33:33-07:00 | run_mast3r.sh; on-disk mtime | MASt3R-SLAM; host unrecoverable. |
| `003-firepit/matcha` | **deduced** | 2026-04-30T14:36:03-07:00 | journal Phase-A recipe; on-disk mtime | Phase-A MAtCha; journal names firepit among Phase-A scenes; host unrecoverable. |
| `003-firepit/slam3r` | **deduced** | 2026-04-12T06:18:18-07:00 | on-disk mtime | SLAM3R — NO run-script, journal-silent: params unrecoverable (only date on-disk). |
| `004-sky-house/mast3r` | **deduced** | 2026-04-29T13:09:27-07:00 | run_mast3r.sh; on-disk mtime | MASt3R-SLAM on the sky-house pool; host probably tbeeprz but not separately attested → deduced. |
| `004-sky-house/matcha` | **measured** | 2026-04-29T22:45:34-07:00 | journal matcha-quality thread (tbeeprz); backfill_manifests.py host pattern; on-disk mtime | MEASURED: host tbeeprz from journal+backfill; date on-disk; params (recipe) deduced. |
| `dtu-bicycle/colmap` | **deduced** | — | run_colmap_sparse.sh | DTU benchmark COLMAP; on-disk mtime is 2022 UPSTREAM dataset date, NOT our run → date null. |
| `dtu-bicycle/matcha` | **deduced** | 2026-05-02T19:11:44-07:00 | journal 3d-scene-examples note (r=0.1 for DTU); on-disk mtime (output) | MAtCha on DTU; r=0.1 per scene-examples note; date from output mtime (inputs predate). |

**Summary:** 1 measured, 11 deduced (of 12 legacy transforms). Records enriched with real dates + script-derived params even where provenance stays `deduced`.

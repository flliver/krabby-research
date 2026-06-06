# Legacy-scene provenance ledger (STO-SCN-036)

Per-transform reconstruction of the 12 `run-legacy` records. Every value
traces to a named source below; `deduced`/null where evidence is absent
(T-002 — nothing fabricated). Dates are CoW-preserved on-disk mtimes;
`nvidia_driver` is deduced from each host's dpkg.log nvidia-driver timeline
(the package version installed as of the run date) — host-pinned records only.

| Scene / pipeline | provenance | started (on-disk) | sources | note |
|---|---|---|---|---|
| `001-patio/colmap` | **deduced** | 2026-04-11T13:53:42-07:00 | run_colmap_sparse.sh+dense; on-disk mtime | COLMAP sparse+dense; host unrecoverable (mid-April, pre-journal). |
| `001-patio/mast3r` | **measured** | 2026-04-12T12:07:44-07:00 | mast3r_output artifact present ONLY on sbeeprz (outposts partial per-host tree); run_mast3r.sh; on-disk mtime; mast3r-build.log base image | MEASURED: ran on sbeeprz (RTX 4080) — mast3r_output lives only there; date 04-12; base nvcr pytorch:25.10. |
| `001-patio/matcha` | **measured** | 2026-04-30T13:58:47-07:00 | journal: MAtCha pipeline was a tbeeprz workflow (matcha-quality thread + all matcha train-logs on t); on-disk mtime | MEASURED: host tbeeprz (journal-inferred, not artifact-located); date on-disk; recipe deduced (Phase-A). |
| `001-patio/vggt` | **measured** | 2026-04-12T11:21:51-07:00 | vggt_output artifact present ONLY on dbeeprz (outposts partial per-host tree); run_vggt.sh; on-disk mtime | MEASURED: ran on dbeeprz (RTX 4080) — vggt_output lives only there; date 04-12. Image name not in script. |
| `002-patio/colmap` | **deduced** | — | run_colmap_sparse.sh | Empty sparse/dense — incomplete run; no output files, date unrecoverable. |
| `003-firepit/mast3r` | **measured** | 2026-04-12T11:33:33-07:00 | mast3r_output artifact present ONLY on sbeeprz (outposts partial per-host tree); run_mast3r.sh; on-disk mtime; mast3r-build.log base image | MEASURED: ran on sbeeprz (RTX 4080) — mast3r_output lives only there; date 04-12; base nvcr pytorch:25.10. |
| `003-firepit/matcha` | **measured** | 2026-04-30T14:36:03-07:00 | journal: MAtCha pipeline was a tbeeprz workflow; firepit named among Phase-A scenes; on-disk mtime | MEASURED: host tbeeprz (journal-inferred); date on-disk; recipe deduced (Phase-A). |
| `003-firepit/slam3r` | **measured** | 2026-04-12T06:18:18-07:00 | slam3r_output artifact present ONLY on dbeeprz (outposts partial per-host tree); on-disk mtime; OLAI corpus 3d-reconstruction/slam3r: CUDA 12.8 / Py3.11 / PyTorch 2.5, tested on 003-firepit | MEASURED host/date: ran on dbeeprz (RTX 4080); date 04-12; CUDA 12.8/Py3.11/PyTorch2.5 from corpus. Invocation params still unrecoverable (no run-script). |
| `004-sky-house/mast3r` | **measured** | 2026-04-29T13:09:27-07:00 | OLAI corpus 3d-reconstruction/mast3r-slam: sky-house-dining = ~40 min on RTX 5080 (tbeeprz); run_mast3r.sh; on-disk mtime; mast3r-build.log base image | MEASURED: corpus note pins sky-house-dining MASt3R to RTX 5080 (tbeeprz), ~40 min (2400 s); CUDA 13 (multi-arch build). |
| `004-sky-house/matcha` | **measured** | 2026-04-29T22:45:34-07:00 | journal matcha-quality thread (tbeeprz); backfill_manifests.py host pattern; on-disk mtime | MEASURED: host tbeeprz from journal+backfill; date on-disk; params (recipe) deduced. |
| `dtu-bicycle/colmap` | **deduced** | — | run_colmap_sparse.sh | DTU benchmark COLMAP; on-disk mtime is 2022 UPSTREAM dataset date, NOT our run → date null. |
| `dtu-bicycle/matcha` | **measured** | 2026-05-02T19:11:44-07:00 | journal: MAtCha pipeline was a tbeeprz workflow; journal 3d-scene-examples note (r=0.1 for DTU); on-disk mtime (output) | MEASURED: host tbeeprz (journal-inferred); r=0.1 per scene-examples note; date from output mtime (inputs predate). |

**Summary:** 9 measured, 3 deduced (of 12 legacy transforms). Records enriched with real dates + script-derived params even where provenance stays `deduced`.

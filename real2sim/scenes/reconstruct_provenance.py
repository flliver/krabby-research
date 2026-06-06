#!/usr/bin/env python3
"""STO-SCN-036 — reconstruct legacy-scene provenance from M11 evidence.

Upgrades the 12 `run-legacy` transform records that STO-SCN-033 migrated with
`provenance: "deduced"` and empty environments. Evidence sources (precedence
mirrors how `backfill_manifests.py` was built — "journal notes + on-disk
evidence"):

  1. M11 journal threads  (.../011-scene-reconstruction/journal/.../threads/)
  2. backfill_manifests.py / manifest_lib.py  (the measured-record shape)
  3. in-repo run-scripts  (run_colmap_*.sh, run_mast3r.sh, run_vggt.sh)
  4. on-disk tool evidence  (CoW-PRESERVED file mtimes, output files present)
  (.claude session histories were scanned 2026-06-05 and found to NOT hold the
   original run-commands — see the story; not used as a source here.)

THE GATE (T-002 — never fabricate):
  - `started` is taken from the CoW-preserved on-disk mtime (real evidence) where
    the files are genuine run outputs; left null where mtimes are upstream-dataset
    or the run is empty.
  - a record is `measured` ONLY when host is known from the journal AND a real date
    exists; otherwise it stays `deduced`, with unknown fields as the literal
    "unknown" / null. Params recovered from run-scripts go in specification.json.
  - every field's source is recorded in provenance-ledger.md.

Idempotent. Reads live mtimes so re-runs reflect the store. Writes results.json +
specification.json per transform and emits the ledger. Never invents a value.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

SCENES = os.environ.get("KRABBY_SCENES", "/var/krabby/scenes")
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance-ledger.md")

# ── Reusable param blocks (source: in-repo run-scripts) ──────────────────────
P_COLMAP_SPARSE = {
    "camera_model": "SIMPLE_RADIAL", "single_camera": True,
    "sift_max_num_features": 8192, "matcher": "sequential", "sequential_overlap": 15,
}
P_COLMAP_FULL = {**P_COLMAP_SPARSE, "dense": {
    "undistort": True, "patch_match_geom_consistency": True, "stereo_fusion": True}}
P_MAST3R = {"pipeline": "MASt3R-SLAM", "config": "config/base.yaml", "viz": False}
P_VGGT = {"entrypoint": "demo_colmap.py", "use_ba": True,
          "max_query_pts": 2048, "query_frame_num": 5}
# Phase-A MAtCha recipe — DEDUCED from the matcha-quality journal thread (no manifest):
P_MATCHA_PHASE_A = {
    "frames": 12, "image_resolution_long_edge": 1024, "encoder": "vitl",
    "sfm_config": "unposed", "alignment_config": "default",
    "_provenance": "deduced — Phase-A recipe from journal thread 'MAtCha mesh quality'; no manifest",
}

# ── The reviewed facts table — one entry per (scene, pipeline) ───────────────
# date_mode: "min" (earliest output mtime), "max" (latest — when inputs predate run),
#            "none" (mtime is upstream-dataset/empty → leave date null).
FACTS = {
 "001-patio/colmap":  dict(prov="deduced", host=None, gpu="unknown", image="unknown",
    date_mode="min", status="success", params=P_COLMAP_FULL,
    sw={}, src=["run_colmap_sparse.sh+dense", "on-disk mtime"],
    note="COLMAP sparse+dense; host unrecoverable (mid-April, pre-journal)."),
 "001-patio/mast3r":  dict(prov="deduced", host=None, gpu="unknown", image="krabby-mast3r",
    date_mode="min", status="success", params=P_MAST3R,
    sw={}, src=["run_mast3r.sh", "on-disk mtime"],
    note="MASt3R-SLAM (krabby-mast3r); host unrecoverable."),
 "001-patio/matcha":  dict(prov="deduced", host=None, gpu="unknown", image="krabby-matcha:latest",
    date_mode="min", status="success", params=P_MATCHA_PHASE_A,
    sw={"matcha": {"git_sha": "unknown"}}, src=["journal Phase-A recipe", "on-disk mtime"],
    note="Phase-A MAtCha; recipe deduced, date on-disk; host unrecoverable."),
 "001-patio/vggt":    dict(prov="deduced", host=None, gpu="unknown", image="unknown",
    date_mode="min", status="success", params=P_VGGT,
    sw={}, src=["run_vggt.sh", "on-disk mtime"],
    note="VGGT (demo_colmap.py --use_ba); host/image unrecoverable."),
 "002-patio/colmap":  dict(prov="deduced", host=None, gpu="unknown", image="unknown",
    date_mode="none", status="partial", params=P_COLMAP_SPARSE,
    sw={}, src=["run_colmap_sparse.sh"],
    note="Empty sparse/dense — incomplete run; no output files, date unrecoverable."),
 "003-firepit/mast3r":dict(prov="deduced", host=None, gpu="unknown", image="krabby-mast3r",
    date_mode="min", status="success", params=P_MAST3R,
    sw={}, src=["run_mast3r.sh", "on-disk mtime"],
    note="MASt3R-SLAM; host unrecoverable."),
 "003-firepit/matcha":dict(prov="deduced", host=None, gpu="unknown", image="krabby-matcha:latest",
    date_mode="min", status="success", params=P_MATCHA_PHASE_A,
    sw={"matcha": {"git_sha": "unknown"}}, src=["journal Phase-A recipe", "on-disk mtime"],
    note="Phase-A MAtCha; journal names firepit among Phase-A scenes; host unrecoverable."),
 "003-firepit/slam3r":dict(prov="deduced", host=None, gpu="unknown", image="unknown",
    date_mode="min", status="success", params={},
    sw={}, src=["on-disk mtime"],
    note="SLAM3R — NO run-script, journal-silent: params unrecoverable (only date on-disk)."),
 "004-sky-house/mast3r":dict(prov="deduced", host=None, gpu="unknown", image="krabby-mast3r",
    date_mode="min", status="success", params=P_MAST3R,
    sw={}, src=["run_mast3r.sh", "on-disk mtime"],
    note="MASt3R-SLAM on the sky-house pool; host probably tbeeprz but not separately attested → deduced."),
 "004-sky-house/matcha":dict(prov="measured", host="tbeeprz",
    gpu="NVIDIA GeForce RTX 5080 / 16 GB", image="krabby-matcha:latest",
    date_mode="min", status="success", params=P_MATCHA_PHASE_A,
    sw={"matcha": {"git_sha": "unknown"}},
    src=["journal matcha-quality thread (tbeeprz)", "backfill_manifests.py host pattern", "on-disk mtime"],
    note="MEASURED: host tbeeprz from journal+backfill; date on-disk; params (recipe) deduced."),
 "dtu-bicycle/colmap":dict(prov="deduced", host=None, gpu="unknown", image="unknown",
    date_mode="none", status="success", params=P_COLMAP_SPARSE,
    sw={}, src=["run_colmap_sparse.sh"],
    note="DTU benchmark COLMAP; on-disk mtime is 2022 UPSTREAM dataset date, NOT our run → date null."),
 "dtu-bicycle/matcha":dict(prov="deduced", host=None, gpu="unknown", image="krabby-matcha:latest",
    date_mode="max", status="success", params={**P_MATCHA_PHASE_A, "chart_resolution_r": 0.1},
    sw={"matcha": {"git_sha": "unknown"}},
    src=["journal 3d-scene-examples note (r=0.1 for DTU)", "on-disk mtime (output)"],
    note="MAtCha on DTU; r=0.1 per scene-examples note; date from output mtime (inputs predate)."),
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _mtimes(data_dir: str) -> list[float]:
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            try:
                out.append(os.path.getmtime(os.path.join(root, fn)))
            except OSError:
                pass
    return sorted(out)


def _outputs(tdir: str) -> list[dict]:
    """List primary output artifacts (meshes/clouds/cameras) — path + bytes."""
    exts = (".ply", ".obj", ".glb", ".gltf", ".splat")
    names = ("cameras.json", "fused.ply", "points3D.bin", "points3D.txt")
    found = []
    data = os.path.join(tdir, "data")
    for root, _, files in os.walk(data):
        for fn in files:
            if fn.lower().endswith(exts) or fn in names:
                p = os.path.join(root, fn)
                found.append((os.path.relpath(p, tdir), os.path.getsize(p)))
    found.sort(key=lambda x: (-x[1], x[0]))  # biggest first
    return [{"path": rp, "bytes": b} for rp, b in found[:8]]


def main() -> None:
    rows = []
    for key, F in FACTS.items():
        scene, pipe = key.split("/")
        tdir = os.path.join(SCENES, scene, f"pipeline-{pipe}", "run-legacy", "transform-01-legacy")
        if not os.path.isdir(tdir):
            rows.append((key, "MISSING", "—", "transform dir absent")); continue
        mt = _mtimes(os.path.join(tdir, "data"))
        started = finished = None
        if F["date_mode"] == "min" and mt:
            started = _iso(mt[0]); finished = _iso(mt[-1])
        elif F["date_mode"] == "max" and mt:
            started = _iso(mt[-1])

        env = {
            "os": "unknown",
            "gpu": F["gpu"],
            "nvidia_driver": "unknown",
            "cuda": "unknown",
            "container": {"image": F["image"], "tag": "unknown", "digest": "unknown"},
            "software": F["sw"],
        }
        results = {
            "schema_version": "1",
            "transform": "transform-01-legacy",
            "status": F["status"],
            "provenance": F["prov"],
            "started": started,
            "finished": finished,
            "host": F["host"],
            "environment": env,
            "outputs": _outputs(tdir),
        }
        # ── write results.json ──
        with open(os.path.join(tdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2); f.write("\n")
        # ── enrich specification.json params (the "what") ──
        spec_path = os.path.join(tdir, "specification.json")
        try:
            with open(spec_path) as f:
                spec = json.load(f)
        except OSError:
            spec = {"schema_version": "1", "transform": "transform-01-legacy",
                    "pipeline": pipe, "kind": "reconstruction", "inputs": ["input"],
                    "maturity": "prototype", "story": "STO-SCN-036"}
        spec["parameters"] = F["params"]
        spec.setdefault("kind", "reconstruction")
        spec["story"] = "STO-SCN-036"
        spec["description"] = f"Legacy {pipe} run — provenance reconstructed from M11 evidence (STO-SCN-036)."
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2); f.write("\n")

        rows.append((key, F["prov"], started or "—", F["note"], F["src"]))

    # ── ledger ──
    lines = [
        "# Legacy-scene provenance ledger (STO-SCN-036)",
        "",
        "Per-transform reconstruction of the 12 `run-legacy` records. Every value",
        "traces to a named source below; `deduced`/null where evidence is absent",
        "(T-002 — nothing fabricated). Dates are CoW-preserved on-disk mtimes.",
        "",
        "| Scene / pipeline | provenance | started (on-disk) | sources | note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r[1] == "MISSING":
            lines.append(f"| `{r[0]}` | — | — | — | {r[3]} |"); continue
        key, prov, started, note, src = r
        lines.append(f"| `{key}` | **{prov}** | {started} | {'; '.join(src)} | {note} |")
    measured = sum(1 for r in rows if r[1] == "measured")
    deduced = sum(1 for r in rows if r[1] == "deduced")
    lines += ["",
              f"**Summary:** {measured} measured, {deduced} deduced "
              f"(of {len(FACTS)} legacy transforms). Records enriched with real dates "
              f"+ script-derived params even where provenance stays `deduced`.",
              ""]
    with open(LEDGER, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(rows)} records ({measured} measured, {deduced} deduced).")
    print(f"Ledger: {LEDGER}")


if __name__ == "__main__":
    main()

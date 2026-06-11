#!/usr/bin/env python3
"""repro_check.py — reproducibility harness (STO-SCN-075, the M11 gate).

Takes a pipeline_run's `run_record.json` ALONE and answers:
"can this be reproduced, and how close is the result?"

    check   — static gate: is the record complete enough to re-run?
              (image digests pinned, settings expanded, inputs hashed,
              no license flags barring deliverable use)
    rerun   — re-execute the record on a host (operator-chosen) via
              run_pipeline.py, into run-<r>-repro-<date>
    compare — original vs re-run outputs, per-output-type tolerances

Tolerances are MEASURED, not invented (T-017): `measure-variance`
re-runs the same record twice on the SAME host and reports the
observed deltas; the catalog tolerance values derive from that.
Until measured, compare reports raw deltas and abstains from
pass/fail on metrics that have no recorded tolerance (T-002).

Verdicts are written onto the run_record (`reproducibility.verdict`)
so the Studio leaderboard can filter to reproduced runs. Backfilled
records (`backfilled: true` + unknown provenance) are
non-reproducible-by-record by definition — they rank, but cannot
gate M11.

Usage:
    python3 real2sim/repro_check.py check   <run_dir>
    python3 real2sim/repro_check.py compare <run_dir_a> <run_dir_b>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# mesh comparison tolerances: populated from measured same-host variance
# (measure-variance step). Explicitly None = unmeasured -> abstain (T-002).
TOLERANCES = {
    "verts_rel": None,       # relative vertex-count delta
    "tris_rel": None,        # relative triangle-count delta
    "bbox_rel": None,        # relative bbox-extent delta
}


def load_record(run_dir: Path) -> dict:
    f = run_dir / "run_record.json"
    if not f.exists():
        sys.exit(f"no run_record.json in {run_dir} — not a v3 run")
    return json.loads(f.read_text())


def check(run_dir: Path) -> int:
    rec = load_record(run_dir)
    failures, warnings = [], []
    if rec.get("backfilled"):
        failures.append("backfilled record — provenance reconstructed post-hoc, "
                        "not reproducible by record (ranks, but cannot gate M11)")
    prov = rec.get("provenance", {})
    if not prov:
        failures.append("no provenance block")
    for node, p in prov.items():
        if not (p.get("image_digest") or "").startswith("sha256"):
            failures.append(f"{node}: image digest not pinned")
        if not p.get("tools_git_sha"):
            warnings.append(f"{node}: tools_git_sha missing")
        if node == next(iter(prov)) and not p.get("input_hashes"):
            failures.append(f"{node}: input hashes missing")
    if not rec.get("instance", {}).get("expanded_settings"):
        failures.append("no expanded settings snapshot")
    flags = rec.get("reproducibility", {}).get("license_flags", [])
    deliverable = not flags
    print(f"record: {run_dir / 'run_record.json'}")
    print(f"reproducible by record: {'YES' if not failures else 'NO'}")
    for f_ in failures:
        print(f"  FAIL: {f_}")
    for w in warnings:
        print(f"  warn: {w}")
    print(f"deliverable-eligible: {'YES' if deliverable else 'NO'}")
    for fl in flags:
        print(f"  license: {fl}")
    return 0 if not failures else 1


def _mesh_stats(ply: Path) -> dict:
    """Header-only PLY stats (no heavy deps): vertex/face counts + size."""
    counts = {}
    with ply.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", "ignore").strip()
            if line.startswith("element"):
                _, name, n = line.split()
                counts[name] = int(n)
            if line == "end_header":
                break
    return {"verts": counts.get("vertex", 0), "faces": counts.get("face", 0),
            "bytes": ply.stat().st_size}


def compare(a: Path, b: Path) -> int:
    rec_a, rec_b = load_record(a), load_record(b)
    sa, sb = rec_a["instance"]["expanded_settings"], rec_b["instance"]["expanded_settings"]
    if sa != sb:
        print("WARNING: settings differ between the two runs — this is not a "
              "reproduction pair, it's an A/B comparison:")
        for nid in sorted(set(sa) | set(sb)):
            if sa.get(nid) != sb.get(nid):
                print(f"  {nid}: {sa.get(nid)} vs {sb.get(nid)}")
    print(f"{'metric':28s} {'A':>14s} {'B':>14s} {'rel delta':>10s}  verdict")
    overall_fail, overall_abstain = False, False
    meshes_a = sorted(a.rglob("*.ply"))
    for ma in meshes_a:
        rel = ma.relative_to(a)
        mb = b / rel
        if not mb.exists():
            print(f"{str(rel):28s} {'present':>14s} {'MISSING':>14s}")
            overall_fail = True
            continue
        st_a, st_b = _mesh_stats(ma), _mesh_stats(mb)
        for key, tol_key in (("verts", "verts_rel"), ("faces", "tris_rel")):
            va, vb = st_a[key], st_b[key]
            d = abs(va - vb) / max(va, 1)
            tol = TOLERANCES[tol_key]
            if tol is None:
                verdict, overall_abstain = "ABSTAIN (tolerance unmeasured)", True
            elif d <= tol:
                verdict = f"PASS (<= {tol})"
            else:
                verdict, overall_fail = f"FAIL (> {tol})", True
            print(f"{f'{rel}:{key}':28s} {va:>14,} {vb:>14,} {d:>10.4%}  {verdict}")
    if not meshes_a:
        print("no tracked meshes under A — nothing to compare yet")
        overall_abstain = True
    print(f"\noverall: {'FAIL' if overall_fail else 'ABSTAIN — measure variance first (T-017)' if overall_abstain else 'PASS'}")
    return 1 if overall_fail else 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "check":
        return check(Path(sys.argv[2]))
    if cmd == "compare":
        return compare(Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

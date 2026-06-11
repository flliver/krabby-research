#!/usr/bin/env python3
"""backfill_run_records.py — materialize v3 run_record.json for historical runs (STO-SCN-077).

For every `pipeline-*/run-*` in the store, derives a run_record
(schema: real2sim/schemas/run_record.json) from what the run actually
recorded:

    instance.expanded_settings  <- specification.json parameters
    execution                   <- results.json (host, times, status)
    provenance                  <- results.json environment.container
                                   (+ spec parameter pins)
    reproducibility.by_record   <- "unknown" unless digests are pinned
    license_flags               <- model_license pins (e.g. DA3 CC-BY-NC)

T-002 hard rule: unrecoverable fields are explicit null/"unknown" with
`backfilled: true` — never inferred. Idempotent: REFUSES to overwrite
an existing run_record.json (re-run = zero diff, zero writes).

    python3 real2sim/backfill_run_records.py            # DRY RUN (default)
    python3 real2sim/backfill_run_records.py --write    # gated on operator
                                                        # review (T-007,
                                                        # STO-SCN-076)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from studio_model import STORE, scenes, pipeline_runs  # noqa: E402

PIPELINE_OF = {"matcha": "matcha-trunk", "da3": "da3-eval"}


def derive_record(r) -> dict:
    """PipelineRun (adapter view) -> run_record dict. Honest-unknown."""
    settings, prov, flags = {}, {}, []
    started = finished = host = None
    status = "unknown"
    for t in r.task_runs:
        node = t.transform
        params = t.settings if isinstance(t.settings, dict) else {}
        settings[node] = params
        digest = t.image_digest
        pinned = isinstance(digest, str) and digest.startswith("sha256:")
        prov[node] = {
            "image": None if t.image == "unknown" else t.image,
            "image_digest": digest if pinned else None,
            "tools_git_sha": None,          # not recorded historically
            "code_ref": None,
            "input_hashes": None,           # cannot recover post-hoc (T-002)
        }
        lic = params.get("model_license")
        if lic and ("NC" in lic or "non-commercial" in lic.lower()):
            flags.append(f"{params.get('model', node)}: {lic}")
        if t.host != "unknown":
            host = t.host
        if t.status != "unknown":
            status = t.status
        if t.started not in ("unknown", None):
            started = min(x for x in (started, t.started) if x)
        if t.finished not in ("unknown", None):
            finished = max(x for x in (finished, t.finished) if x)
    # render-variant runs: settings live in the variant tool's record
    if r.kind == "render-variant":
        settings["(render-variant)"] = {"source_run": r.source_run}
    return {
        "schema": 3,
        "scene": r.scene,
        "pipeline": PIPELINE_OF.get(r.pipeline, r.pipeline),
        "run": r.run,
        "variant": r.variant,
        "source_run": r.source_run,
        "instance": {"name": None, "expanded_settings": settings},
        "execution": {"host": host, "trigger": "backfill",
                      "started": started, "finished": finished,
                      "status": {"success": "success", "failed": "failure",
                                 "partial": "partial"}.get(status, "unknown")},
        "provenance": prov,
        "reproducibility": {
            "by_record": False if not any(p["image_digest"] for p in prov.values())
                         else "unknown",
            "license_flags": flags,
            "notes": "backfilled from spec/results JSONs; digests/tool SHAs/input "
                     "hashes not pinned at run time -> not reproducible by record "
                     "alone (075 harness excludes from M11 gating)",
        },
        "backfilled": True,
        "backfill_notes": "STO-SCN-077",
    }


def main() -> int:
    write = "--write" in sys.argv
    plan, skipped, unknown_counts = [], [], {"fields_total": 0, "fields_unknown": 0}
    for sdir in scenes():
        for r in pipeline_runs(sdir):
            rdir = STORE / r.scene / f"pipeline-{r.pipeline}" / f"run-{r.run}"
            out = rdir / "run_record.json"
            if out.exists():
                skipped.append(str(out))
                continue
            rec = derive_record(r)
            for p in rec["provenance"].values():
                for k in ("image", "image_digest", "tools_git_sha", "input_hashes"):
                    unknown_counts["fields_total"] += 1
                    unknown_counts["fields_unknown"] += p[k] is None
            plan.append((out, rec))

    print(f"{'WRITE' if write else 'DRY RUN'}: {len(plan)} run_records to create, "
          f"{len(skipped)} already exist (skipped — idempotent)")
    by_scene: dict[str, int] = {}
    flagged = 0
    for out, rec in plan:
        by_scene[rec["scene"]] = by_scene.get(rec["scene"], 0) + 1
        flagged += bool(rec["reproducibility"]["license_flags"])
    for s, n in sorted(by_scene.items()):
        print(f"  {s:16s} {n}")
    ft, fu = unknown_counts["fields_total"], unknown_counts["fields_unknown"]
    print(f"provenance fields unknown: {fu}/{ft} ({fu*100//max(ft,1)}%) — explicit nulls, T-002")
    print(f"license-flagged runs (CC-BY-NC, not deliverable): {flagged}")

    if not write:
        print("\n(no writes — re-run with --write after operator review of "
              "real2sim/STORE-SCHEMA-V3.md; T-007 gate STO-SCN-076)")
        return 0

    import jsonschema
    schema = json.loads((Path(__file__).parent / "schemas/run_record.json").read_text())
    v = jsonschema.Draft202012Validator(schema)
    for out, rec in plan:
        errs = [e.message for e in v.iter_errors(rec)]
        if errs:
            sys.exit(f"REFUSING {out}: schema-invalid record: {errs}")
        out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {len(plan)} run_records (all schema-valid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Write the rough-terrain kind rows deterministically from a yaml spec (PLAN step 4 (a); CPU).

    make_manifest.py [--spec manifests/rt_kinds.spec.yaml] [--check]

Writes three files (``rt_spec.outputs``): ``manifests/scenarios_rt.yaml`` (the 21 kind rows ``rt_<kind>_b<n>__omni``,
field_hard__omni's header and schedule verbatim), ``manifests/envs/rt_kind_eval.env`` (the kind rows' base env:
TDR's field_hard env minus field_hard__omni's own block) and ``manifests/rows_rt.json`` (every kind and retention
row: manifest, scenario, env file, level file, args, env seeds, crossing radius, envelope flag). ``--check``
regenerates in memory and exits 1 when any file on disk differs (byte equality, as SITE's make_site_manifest.py and
``pipeline.py gen --check`` do). The output headers record the spec's sha256.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rt_spec as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, default=None, help=f"default {S.SPEC_PATH.name}")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    spec = S.load_spec(a.spec)
    outs = S.outputs(spec, (a.spec or S.SPEC_PATH).name)
    if a.check:
        bad = [p for p, t in outs.items() if not (p.is_file() and p.read_text() == t)]
        for p in outs:
            print(f"RT_MANIFEST {'DIFFERS' if p in bad else 'unchanged'}: {p}")
        return 1 if bad else 0
    for p, t in outs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
        print(f"RT_MANIFEST wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

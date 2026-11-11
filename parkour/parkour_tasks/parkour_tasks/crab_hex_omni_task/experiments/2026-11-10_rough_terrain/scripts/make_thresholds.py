#!/usr/bin/env python3
"""Freeze ``thresholds_rough_v1.json`` from TDR's step-6 rows (PLAN step 7, STOP 0; CPU; never hand-edited).

    make_thresholds.py [--root LR_DIR ...] [--tdr SHA] [--out JSON] [--check]

The rule (PLAN step 7 + Gates + Settled 4), applied to TDR's es3 kind rows (the band-choice seed; G is read on
es1-2), each row's statistic = success (schedule complete AND crossed, readout.py):

* a band is ABOVE the blind envelope when it is a step / drop / pothole band whose bottom exceeds 0.14 m -- the b3
  bands of ledge_up, ledge_down and pits (rt_spec.above_envelope; asserted). Those are report-only;
* per kind, the GATE band = the lowest in-envelope band where TDR's es3 success <= 0.80;
* a kind with no such band is SOLVED: it enters R on its highest in-envelope band;
* k = 2. Decision: 0 gated kinds -> ``close`` ("TDR already handles this terrain"; steps 8-9 do not run);
  exactly 1 -> ``stop_and_ask`` (k = 2 cannot be met; ask before step 8); >= 2 -> ``proceed``;
* a pick whose es3 success is borderline (0.74-0.86) is recorded (PLAN item 14); no extra band is run.

The file records the rule, every TDR es3 row (completion, success, stuck, crossed, the r_max distribution,
termination reasons, the slopes up / down share), TDR's retention rows when present (report), the stamps read and
the sha256 of this script, readout.py and rt_spec.py. It holds no timestamp, so ``--check`` (regenerate in memory,
byte-compare, exit 1 on a difference) is exact. An existing file that differs is never overwritten (exit 3): the
thresholds are frozen at STOP 0.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rt_spec as S  # noqa: E402


def _load_readout():
    """readout.py under a campaign-unique module name (``rt_readout``; a bare ``readout`` could collide)."""
    import importlib.util  # noqa: PLC0415

    if "rt_readout" in sys.modules:
        return sys.modules["rt_readout"]
    spec = importlib.util.spec_from_file_location("rt_readout", str(HERE / "readout.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rt_readout"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("rt_readout", None)
        raise
    return mod


RO = _load_readout()

SCHEMA = "rt_thresholds_v1"
GATE_LINE = 0.80
BORDERLINE = (0.74, 0.86)
BAND_ORDER = tuple(S.BANDS)                    # b1, b2, b3 (ascending difficulty)


def pick_bands(kind: str, success: dict) -> dict:
    """PLAN step 7 for one kind. ``success``: {band: TDR es3 success share} over every band of the kind."""
    S.above_envelope_set()
    inb = [b for b in BAND_ORDER if not S.above_envelope(kind, b)]
    missing = [b for b in inb if success.get(b) is None]
    if missing:
        raise ValueError(f"{kind}: no TDR es3 success for in-envelope band(s) {missing}")
    gate = next((b for b in inb if success[b] <= GATE_LINE + RO.EPS), None)
    out = {"in_envelope_bands": inb, "above_envelope_bands": [b for b in BAND_ORDER if b not in inb]}
    if gate is not None:
        out.update(status="gated", gate_band=gate, r_band=None, pick_success=success[gate])
    else:
        out.update(status="solved", gate_band=None, r_band=inb[-1], pick_success=success[inb[-1]])
    v = out["pick_success"]
    out["borderline"] = bool(BORDERLINE[0] - RO.EPS <= v <= BORDERLINE[1] + RO.EPS)
    out["borderline_bands"] = [b for b in inb if BORDERLINE[0] - RO.EPS <= success[b] <= BORDERLINE[1] + RO.EPS]
    return out


def decision_of(n_gated: int) -> dict:
    if n_gated == 0:
        return {"decision": "close", "why": "no gated kind (every kind solved in-envelope): report 'TDR already "
                                            "handles this terrain'; steps 8-9 do not run; close"}
    if n_gated == 1:
        return {"decision": "stop_and_ask", "why": "exactly 1 gated kind: G as settled (k = 2) cannot pass; stop "
                                                   "and ask before step 8"}
    return {"decision": "proceed", "why": f"{n_gated} gated kinds >= k = 2: steps 8-9 run"}


def freeze(base: dict, *, sources: dict | None = None) -> dict:
    """The thresholds document from a ``readout.baseline`` result (pure: no I/O beyond what ``base`` holds)."""
    if base.get("kind_env_seeds") != [3]:
        raise ValueError(f"baseline env seeds {base.get('kind_env_seeds')} (the band choice reads es3 only)")
    kinds, table, missing = {}, {}, []
    for kind in S.KINDS:
        succ = {}
        for b in BAND_ORDER:
            rep = (base["kinds"].get(kind) or {}).get(b)
            if rep is None or rep.get("status") != "OK" or "stats" not in rep:
                missing.append(f"{S.row_id(kind, b)} es3: {None if rep is None else rep.get('status')} "
                               f"{[f.get('problems') for f in (rep or {}).get('stamps') or []][:1]}")
                continue
            st = rep["stats"]
            if st["n"] != 100:
                missing.append(f"{S.row_id(kind, b)} es3: {st['n']} episodes (want 100)")
                continue
            succ[b] = st["success"]
            table[S.row_id(kind, b)] = {
                "kind": kind, "band": b, "d": list(S.BANDS[b]), "ladder": dict(zip(("bottom", "top"),
                                                                                   S.band_range(kind, b))),
                "above_envelope": S.above_envelope(kind, b), "cross_r": S.CROSS_R[kind], "n": st["n"],
                "completion": st["completion"], "success": st["success"], "stuck": st["stuck"],
                "crossed": st["crossed"], "r_max_all_p50": st["r_max_all"].get("p50"),
                "r_max_completers": st["r_max_completers"], "termination_reasons": st["termination_reasons"],
                **({"slopes_variant": st["slopes_variant"]} if "slopes_variant" in st else {}),
                "stamp": rep["stamps"][0]["stamp"]}
        if len(succ) == len(BAND_ORDER):
            kinds[kind] = {**pick_bands(kind, succ), "tdr_es3_success": succ}
    if missing:
        raise ValueError("TDR's step-6 kind rows are incomplete: " + "; ".join(missing[:12]))
    gated = [k for k in S.KINDS if kinds[k]["status"] == "gated"]
    solved = [k for k in S.KINDS if kinds[k]["status"] == "solved"]
    ret = {rid: {"status": r.get("status"), "completion": (r.get("stats") or {}).get("completion"),
                 "n": (r.get("stats") or {}).get("n"), "stamps": [f.get("stamp") for f in r.get("stamps") or []]}
           for rid, r in (base.get("retention") or {}).items()}
    return {
        "schema": SCHEMA, "tdr_sha256": base["tdr_sha256"],
        "rule": {"gate_line": GATE_LINE, "band_choice_env_seed": 3, "gate_env_seeds": [1, 2], "k": RO.K_GATED,
                 "borderline": list(BORDERLINE), "envelope_m": S.ENVELOPE_M,
                 "above_envelope": [list(x) for x in S.above_envelope_set()], "cross_r": S.CROSS_R,
                 "bands": {b: list(v) for b, v in S.BANDS.items()}, "statistic": "success = complete AND crossed",
                 "G": {**RO.G_RULE, "text": "gain: new - TDR >= max(0.10, 3 SE) and CI95 lower >= 0.05; no gated kind "
                                            "below TDR - max(0.05, 2 SE); >= k kinds gain"},
                 "R_completion": {**RO.R_COMPLETION_CFG, "text": "new >= TDR - max(0.05, 2 SE)"},
                 "R_tracking": {"call": "gate_read.zero_kg_check", "translate": RO.ZERO_KG_TRANSLATE,
                                "its_past_p2a": RO.ITS_PAST_R, "thresholds": "thresholds_v1_2.json",
                                "text": "each decisive suite line worse by <= max(R, 3 SE), its v1.2 R, drift 0; "
                                        "side22 FAST/FAST"},
                 "noise_effect": RO.NOISE_EFFECT, "se": "thresholds.paired_completion_se on (env seed, env_index)"},
        "kinds": kinds, "gated": gated, "solved": solved, "n_gated": len(gated), **decision_of(len(gated)),
        "tdr_baseline_es3": table, "tdr_retention_report": ret, "sources": sources or {},
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, default=RO._jsonable) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--tdr", default=RO.TDR_SHA)
    ap.add_argument("--out", type=Path, default=RO.THRESHOLDS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    roots = [Path(r) for r in a.root] if a.root else [RO.RT_LR]
    R = RO.Rows(roots, shas={a.tdr})
    base = RO.baseline(R, a.tdr, (3,))
    srcs = {"roots": [str(r) for r in roots],
            "sha256": {n: RO.sha256_file(HERE / n) for n in RO.FROZEN_SCRIPTS},
            "rows_rt_json_sha256": RO.sha256_file(S.ROWS_PATH)}
    try:
        text = render(freeze(base, sources=srcs))
    except ValueError as e:
        print(f"RT THRESHOLDS refused: {e}")
        return 2
    if a.check:
        same = a.out.is_file() and a.out.read_text() == text
        print(f"RT THRESHOLDS {'unchanged' if same else 'DIFFERS'}: {a.out}")
        return 0 if same else 1
    if a.out.is_file():
        if a.out.read_text() == text:
            print(f"RT THRESHOLDS unchanged: {a.out}")
            return 0
        print(f"RT THRESHOLDS refused: {a.out} exists and differs (frozen at STOP 0; never overwritten)")
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    doc = json.loads(text)
    print(f"RT THRESHOLDS wrote {a.out}: gated {doc['gated']}, solved {doc['solved']}, decision {doc['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

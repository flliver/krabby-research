#!/usr/bin/env python3
"""DR stress sweep reader (CPU): per-seed classification, the baseline gate, verdicts and deltas (CHARTER §3.3, §3.5).

Every cell-seed is read with Task 4's own line reader (``tools/pipeline/payload_read.level_reads``) on ONE seed's two
stamps (``suite@K`` + ``field_light@K``): the 17 ``p3a.stable90.*.esK`` lines plus the field_light completion, which
``level_reads`` computes from the field_light stamps it is given -- here exactly seed K's (n = 100). The other seed's
17 lines come back unread and are dropped; ``level``'s pooled two-seed call is never used.

A cell-seed is a LAUNCH ERROR (``pass`` None, never a reading) when either stamp's ``run_meta.env_vars`` misses the
cell's knob / payload keys or carries a stray ``KRABBY_STRESS_*`` / ``KRABBY_PAYLOAD_*``, when ``run_meta.student.zero_depth`` is not what the
cell asks for, or when a stamp's ``code_state.json`` is dirty or names another code / eval hash than the launch pin
(``results/launch.json``, written by ``run_sweep.py`` before its first row; a missing pin is itself a launch error).

Run with the Isaac venv python (numpy): ``/home/nickmagus/krabby/isaac_venv/bin/python -I stress_read.py ...``

    stress_read.py cellseed NAME ES [--term T --level V --base B --policy student|teacher]
    stress_read.py baseline --out JSON          the step-7 gate (exit 3 = STOP)
    stress_read.py es2plan --out JSON           the es2 cell list after the es1 grid
    stress_read.py table --out JSON [--md MD]   the per-term verdict table
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as C  # noqa: E402

TH_PATH = C.T4 / "thresholds_v1_2.json"
RECORD = C.MAIN / "parkour/outputs/omni_plan/task4/F19/lr"
RECORD_NAMES = {"L000": "T4F19_sel_c28000", "L090": "T4F19_sel_c28000_L090"}
RESULTS = Path(os.environ.get("DRS_RESULTS") or C.CAMPAIGN / "results")
LAUNCH = RESULTS / "launch.json"
CLOSING = RESULTS / "closing.json"
DROP_FLAG = 0.05
EPS = 1e-9
COMPLETION_TOL = 0.04
LENIENT_SURVIVAL = 0.5
LENIENT_CATS = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
#: Verdict precedence (worst first) for combining sides, levels and bases.
ORDER = ("FRAGILE", "MARGINAL", "INCOMPLETE", "ROBUST")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _payload_read():
    return _load("drs_payload_read", C.EXPERIMENTS / "tools/pipeline/payload_read.py")


def _thresholds_mod():
    return _load("drs_tp_thresholds", C.EXPERIMENTS / "tools/pipeline/thresholds.py")


def _th():
    return json.loads(TH_PATH.read_text())


def _key(line: dict) -> str:
    """A seed-free line key: ``survival.fwd`` / ``h_meas.suite`` / ``field_light_completion``."""
    if line["stat"] == "field_light_completion":
        return "field_light_completion"
    return f"{line['stat']}.{line['scope']}"


def worst(verdicts) -> str | None:
    vs = [v for v in verdicts if v is not None]
    return next((v for v in ORDER if v in vs), None)


def read_stamps(suite_dir: Path, fl_dir: Path, es: int) -> dict:
    """One seed's record from its two stamp dirs (no launch checks)."""
    pr = _payload_read()
    rd = pr.level_reads({f"suite@{es}": str(suite_dir), f"field_light@{es}": str(fl_dir)}, _th())
    lines = {}
    for ln in rd["lines"]:
        if ln["stat"] == "field_light_completion" or ln.get("es") == es:
            lines[_key(ln)] = {"value": ln["value"], "line": ln["line"], "direction": ln["direction"],
                               "pass": bool(ln["pass"])}
    cats = _thresholds_mod().suite_categories(Path(suite_dir))
    surv = {c: (cats.get(c) or {}).get("survival") for c in LENIENT_CATS}
    lenient = all(v is not None and v >= LENIENT_SURVIVAL for v in surv.values())
    return {"es": es, "pass": all(v["pass"] for v in lines.values()) and len(lines) == 18, "n_lines": len(lines),
            "failed": sorted(k for k, v in lines.items() if not v["pass"]), "lines": lines,
            "survival_all": surv, "lenient_feasible": lenient,
            "stamps": {"suite": str(suite_dir), "field_light": str(fl_dir)}}


def launch_pin() -> dict | None:
    return json.loads(LAUNCH.read_text()) if LAUNCH.is_file() else None


def stamp_problems(stamp: Path, want_env: dict, want_zero_depth: bool | None, pin: dict | None) -> list[str]:
    """What makes a stamp a launch error (empty = fine)."""
    bad = []
    meta = json.loads((stamp / "run_meta.json").read_text())
    ev = meta.get("env_vars") or {}
    bad += [f"{k}={v!r} (got {ev.get(k)!r})" for k, v in want_env.items() if str(ev.get(k)) != str(v)]
    bad += [f"unexpected {k}={ev[k]!r}" for k in sorted(ev)
            if k.startswith(("KRABBY_STRESS_", "KRABBY_PAYLOAD_")) and k not in want_env]
    if want_zero_depth is not None:
        zd = (meta.get("student") or {}).get("zero_depth")
        if zd is not bool(want_zero_depth):
            bad.append(f"student.zero_depth={zd!r} (want {bool(want_zero_depth)})")
    cs_path = stamp / "code_state.json"
    cs = json.loads(cs_path.read_text()) if cs_path.is_file() else {}
    if str(cs.get("dirty")) != "False":
        bad.append(f"code_state dirty={cs.get('dirty')!r}")
    if pin is None:
        bad.append("no launch pin (results/launch.json): the driver writes it before any row")
    else:
        for k in ("code_sha", "eval_sha"):
            if cs.get(k) != pin.get(k):
                bad.append(f"code_state {k}={cs.get(k)!r} (launch pin {pin.get(k)!r})")
    return bad


def cellseed(name: str, es: int, term=None, level=None, base="L000", policy="student") -> dict | None:
    """The record of one cell-seed of the sweep (None while either stamp is missing)."""
    sd = C.stamp_dir(name, C.ROWS["suite"]["scenario"], es)
    fd = C.stamp_dir(name, C.ROWS["field_light"]["scenario"], es)
    if sd is None or fd is None:
        return None
    rec = read_stamps(sd, fd, es)
    want = C.expected_env(term, level, base)
    want_zd = (term is not None and term["kind"] == "zero_depth") if policy == "student" else None
    pin = launch_pin()
    bad = stamp_problems(sd, want, want_zd, pin) + stamp_problems(fd, want, want_zd, pin)
    rec["launch_error"] = bad or None
    if bad:
        rec["pass"] = None  # a launch error is never a reading
    return rec


def record_seed(base: str, es: int) -> dict:
    """The Task 4 student record's seed ``es`` at a base (the step-7 comparison)."""
    root = RECORD / RECORD_NAMES[base]

    def newest(sc):
        d = root / sc / f"seed{es:03d}"
        return sorted(p for p in d.iterdir() if (p / "tp_row.json").is_file())[-1]
    return read_stamps(newest("suite__omni"), newest("field_light__omni"), es)


def _completion(rec: dict) -> float | None:
    return rec["lines"].get("field_light_completion", {}).get("value")


# ------------------------------------------------------------------------------------------ the baseline gate
def baseline() -> dict:
    """CHARTER §3.5: stop on any failed line or launch error of a nominal launch; stop on completion only when EVERY
    launch of a base deviates from the record's same seed by more than 0.04 (a split between the two 0 kg launches is
    the documented two-class flip, never a stop)."""
    out = {"launches": {}, "record": {}, "stop": False, "reasons": [], "split": False}
    for base, launches in (("L000", ("A", "B")), ("L090", ("A",))):
        for es in (1, 2):
            rec = record_seed(base, es)
            rc = _completion(rec)
            out["record"][f"{base}@{es}"] = {"completion": rc, "pass": rec["pass"]}
            devs = []
            for la in launches:
                r = cellseed(C.gate_name(C.nominal_id(base, la)), es, None, None, base)
                key = f"{base}_{la}@{es}"
                if r is None:
                    out["stop"] = True
                    out["reasons"].append(f"{key}: stamps missing")
                    continue
                c = _completion(r)
                out["launches"][key] = {"pass": r["pass"], "failed": r["failed"], "completion": c,
                                        "launch_error": r["launch_error"]}
                if r["launch_error"]:
                    out["stop"] = True
                    out["reasons"].append(f"{key}: launch error {r['launch_error']}")
                    continue
                if not r["pass"]:
                    out["stop"] = True
                    out["reasons"].append(f"{key}: stable90 lines failed {r['failed']}")
                devs.append((key, c, None if (c is None or rc is None) else abs(c - rc)))
            if devs and all(d is None or d > COMPLETION_TOL + EPS for _, _, d in devs):
                out["stop"] = True
                out["reasons"].append(f"{base}@{es}: every launch's completion deviates from the record {rc} by more "
                                      f"than {COMPLETION_TOL}: {[(k, c) for k, c, _ in devs]}")
    for es in (1, 2):
        a, b = out["launches"].get(f"L000_A@{es}"), out["launches"].get(f"L000_B@{es}")
        if a and b and a["completion"] != b["completion"]:
            out["split"] = True
    return out


# ------------------------------------------------------------------------------------------- the grid reads
def nominal_refs(base: str, es: int) -> list[dict]:
    """The same-day nominal records of a base and seed: launches A and B at 0 kg, A at L090 (never the closing Z,
    which is the drift check, not a reference)."""
    refs = []
    for la in (("A", "B") if base == "L000" else ("A",)):
        r = cellseed(C.gate_name(C.nominal_id(base, la)), es, None, None, base)
        if r is not None and r["pass"] is not None:
            refs.append(r)
    return refs


def deltas(rec: dict, refs: list[dict], *, flag: bool = True) -> dict:
    """Per line: value - nominal for each nominal launch. ``flag`` marks a survival / completion DROP of more than 0.05
    against EVERY nominal launch (two classes: it fires only if it exceeds against both)."""
    out = {}
    for k, v in rec["lines"].items():
        ds = [None if (v["value"] is None or r["lines"].get(k, {}).get("value") is None)
              else v["value"] - r["lines"][k]["value"] for r in refs]
        drop_stat = k.startswith("survival.") or k == "field_light_completion"
        f = bool(flag and drop_stat and ds and all(d is not None and d < -DROP_FLAG - EPS for d in ds))
        out[k] = {"value": v["value"], "delta": ds, "flag": f}
    return out


def _ref_base(cell: dict) -> str:
    """The nominal a cell's deltas are taken against: L090 for the 90 kg cells and the payload terms, else L000."""
    kind = C.TERM[cell["term"]]["kind"]
    return "L090" if cell["base"] == "L090" or kind in ("payload_com", "payload_kg") else "L000"


def grid_reads() -> dict:
    """Every cell's per-seed records: ``{cell_id: {"cell": ..., "seeds": {es: rec}}}``."""
    out = {}
    for cell in C.cells():
        t = C.TERM[cell["term"]]
        seeds = {}
        for es in (1, 2):
            r = cellseed(C.gate_name(cell["id"]), es, t, cell["level"], cell["base"])
            if r is not None:
                seeds[es] = r
        out[cell["id"]] = {"cell": cell, "seeds": seeds}
    return out


def level_status(entry: dict) -> str | None:
    """``pass`` / ``fail`` / ``split`` / ``error`` / None (unread) of a cell from the seeds it has."""
    s = entry["seeds"]
    if not s:
        return None
    if any(r["pass"] is None for r in s.values()):
        return "error"
    vals = [r["pass"] for r in s.values()]
    if all(vals):
        return "pass"
    if not any(vals):
        return "fail"
    return "split"


def level_verdict(term: dict, level, entry: dict) -> str | None:
    """The verdict an IN-BAND level contributes (CHARTER §3.3), or None (contributes nothing).

    An edge decides on both seeds: pass -> ROBUST, fail -> FRAGILE, split -> MARGINAL, anything unresolved ->
    INCOMPLETE. A non-edge in-band level contributes nothing while its es1 passes; an es1 failure needs es2 (FRAGILE on
    two fails, MARGINAL on a split, INCOMPLETE while es2 is missing); an unread or errored one is INCOMPLETE."""
    s = entry["seeds"]
    st = level_status(entry)
    edge = level in C.band_edges(term)
    if st in (None, "error"):
        return "INCOMPLETE"
    if edge or st != "pass":
        if 1 not in s or 2 not in s:
            return "INCOMPLETE"
        return {"pass": "ROBUST", "fail": "FRAGILE", "split": "MARGINAL"}[st]
    return None


def es2_plan(reads: dict) -> list[str]:
    """The cells that need es2 (CHARTER §3.3): band edges, in-band es1 failures, and the terms 13-15 transitions (last
    passing, first failing level from the nominal outward; a launch error stops the search -- it is not a failure)."""
    need = []
    for cid, e in reads.items():
        cell, t = e["cell"], C.TERM[e["cell"]["term"]]
        r1 = e["seeds"].get(1)
        if r1 is None or r1["pass"] is None:
            continue
        if cell["level"] in C.band_edges(t):
            need.append(cid)
        elif C.inside_band(t, cell["level"]) and r1["pass"] is False:
            need.append(cid)
    for tname in ("T13", "T14", "T15"):
        t = C.TERM[tname]
        for base in t["bases"]:
            ordered = sorted(t["levels"], key=lambda v: abs(v - t["nominal"]))
            last_pass = first_fail = None
            for lv in ordered:
                r1 = reads[C.cell_id(t, lv, base)]["seeds"].get(1)
                if r1 is None or r1["pass"] is None:
                    break
                if r1["pass"]:
                    last_pass = lv
                else:
                    first_fail = lv
                    break
            for lv in (last_pass, first_fail):
                if lv is not None:
                    need.append(C.cell_id(t, lv, base))
    return sorted(set(need))


def closing_record() -> dict | None:
    return json.loads(CLOSING.read_text()) if CLOSING.is_file() else None


def newest_read_time(reads: dict) -> float:
    """The newest ``tp_row.json`` mtime among the grid stamps read (0 when none): the closing's staleness test."""
    ts = []
    for e in reads.values():
        for r in e["seeds"].values():
            for st in (r.get("stamps") or {}).values():
                f = Path(st) / "tp_row.json"
                if f.is_file():
                    ts.append(f.stat().st_mtime)
    return max(ts) if ts else 0.0


def feasibility(term: dict, base: str, reads: dict) -> dict:
    """The report-only bound of terms 13-15 (CHARTER §3.3), from the nominal outward: the last level that passes and
    the first that fails, each on every seed read; a split level is MARGINAL and the bound is then an interval;
    levels read on es1 only are listed (a one-seed bound)."""
    ordered = sorted(term["levels"], key=lambda v: abs(v - term["nominal"]))
    last_pass = first_fail = None
    split, single, stop = [], [], None
    for lv in ordered:
        e = reads[C.cell_id(term, lv, base)]
        st = level_status(e)
        if st in (None, "error"):
            stop = f"{C.fmt(lv)}: {st or 'unread'}"
            break
        if len(e["seeds"]) == 1:
            single.append(C.fmt(lv))
        if st == "pass":
            last_pass = lv
        elif st == "split":
            split.append(C.fmt(lv))
        else:
            first_fail = lv
            break
    return {"last_pass": last_pass, "first_fail": first_fail, "split_levels": split, "single_seed_levels": single,
            "search_stopped": stop, "interval": [last_pass, first_fail]}


def term_table(reads: dict) -> dict:
    rows = []
    for t in C.TERMS:
        for base in t["bases"]:
            ents = {lv: reads[C.cell_id(t, lv, base)] for lv in t["levels"]}
            st = {lv: level_status(e) for lv, e in ents.items()}
            tol = {}
            for sd in ("low", "high"):
                lvls = sorted((lv for lv in t["levels"] if C.side(t, lv) == sd), key=lambda v: abs(v - t["nominal"]))
                edge = None
                for lv in lvls:
                    if st[lv] != "pass":
                        break
                    edge = lv
                lenient = None
                for lv in lvls:
                    e = ents[lv]
                    if not e["seeds"] or not all(r["pass"] is not None and r["lenient_feasible"]
                                                 for r in e["seeds"].values()):
                        break
                    lenient = lv
                tol[sd] = {"tolerated_edge": edge, "lenient_edge": lenient, "levels": lvls}
            verdict, per_level = None, {}
            if t["band"] is not None:
                for lv in t["levels"]:
                    if C.inside_band(t, lv) and abs(lv - t["nominal"]) > EPS:
                        per_level[C.fmt(lv)] = level_verdict(t, lv, ents[lv])
                side_verdicts = {sd: worst(per_level.get(C.fmt(lv)) for lv in tol[sd]["levels"])
                                 for sd in ("low", "high")}
                verdict = worst(per_level.values())
            else:
                side_verdicts = {}
            ref = "L090" if (base == "L090" or t["kind"] in ("payload_com", "payload_kg")) else "L000"
            flag_ok = t["kind"] != "payload_kg"  # 135+ kg has no same-day reference: deltas reported, never flagged
            dl, flags, nrefs = {}, [], {}
            for lv, e in ents.items():
                for es, r in e["seeds"].items():
                    if r["pass"] is None:
                        continue
                    refs = nominal_refs(ref, es)
                    nrefs[f"{C.fmt(lv)}@es{es}"] = len(refs)
                    d = deltas(r, refs, flag=flag_ok)
                    dl.setdefault(C.fmt(lv), {})[es] = d
                    flags += [f"{C.fmt(lv)}@es{es}:{k}" for k, x in d.items() if x["flag"]]
            rows.append({"term": t["term"], "slug": t["slug"], "base": base, "band": t["band"], "verdict": verdict,
                         "side_verdicts": side_verdicts, "level_verdicts": per_level,
                         "status": {C.fmt(k): v for k, v in st.items()}, "tolerance": tol, "delta_reference": ref,
                         "drop_flags": flags, "deltas": dl, "nominal_refs_used": nrefs,
                         "no_reference": sorted(k for k, n in nrefs.items() if n == 0),
                         "feasibility": (feasibility(t, base, reads) if t["term"] in ("T13", "T14", "T15") else None)})
    combined = {}
    for r in rows:
        combined.setdefault(r["term"], []).append(r["verdict"])
    combined = {k: worst(v) for k, v in combined.items()}
    closing = closing_record()
    newest = newest_read_time(reads)
    stale_now = closing is None or closing.get("stamp_time") is None or float(closing["stamp_time"]) < newest
    closing_flag = closing is None or closing.get("status") != "match" or bool(closing.get("stale", True)) or stale_now
    for r in rows:  # CHARTER §3.5: a closing flag marks every cell
        r["closing_flag"] = closing_flag
    return {"rows": rows, "combined_verdicts": combined, "closing": closing, "closing_flag": closing_flag}


def _summary_deltas(d: dict) -> tuple:
    """``(worst survival delta, completion delta)`` of one cell-seed: per line the delta that holds against every
    nominal (the max over launches), then the min over survival lines."""
    def held(x):
        ds = [v for v in x["delta"] if v is not None]
        return max(ds) if ds else None
    surv = [held(x) for k, x in d.items() if k.startswith("survival.")]
    surv = [v for v in surv if v is not None]
    comp = held(d["field_light_completion"]) if "field_light_completion" in d else None
    return (min(surv) if surv else None), comp


def write_md(table: dict, path: Path) -> None:
    f3 = lambda v: "-" if v is None else f"{v:+.3f}"  # noqa: E731
    out = ["| term | base | band | verdict | status by level | tolerated low / high | lenient low / high | drop flags |",
           "|---|---|---|---|---|---|---|---|"]
    for r in table["rows"]:
        tl, th = r["tolerance"]["low"], r["tolerance"]["high"]
        out.append(f"| {r['term']} {r['slug']} | {r['base']} | {r['band']} | {r['verdict']} | "
                   f"{', '.join(f'{k}: {v}' for k, v in r['status'].items())} | "
                   f"{tl['tolerated_edge']} / {th['tolerated_edge']} | {tl['lenient_edge']} / {th['lenient_edge']} | "
                   f"{len(r['drop_flags'])} |")
    out += ["", "**Combined verdicts (worst over bases):** " +
            ", ".join(f"{k} {v}" for k, v in table["combined_verdicts"].items() if v is not None), ""]
    if table["closing_flag"]:
        cl = table["closing"] or {}
        out += [f"**CLOSING NOMINAL FLAG** (status {cl.get('status', 'absent')}, stale {cl.get('stale', True)}): every "
                "cell since the last nominal is flagged (CHARTER §3.5).", ""]
    fz = [r for r in table["rows"] if r.get("feasibility")]
    if fz:
        out += ["| report-only term | base | last pass | first fail | split | one-seed levels | search stopped |",
                "|---|---|---|---|---|---|---|"]
        for r in fz:
            f = r["feasibility"]
            out.append(f"| {r['term']} {r['slug']} | {r['base']} | {f['last_pass']} | {f['first_fail']} | "
                       f"{', '.join(f['split_levels']) or '-'} | {', '.join(f['single_seed_levels']) or '-'} | "
                       f"{f['search_stopped'] or '-'} |")
        out.append("")
    out += ["| cell-seed | ref (n) | worst survival delta | completion delta | flags |", "|---|---|---|---|---|"]
    for r in table["rows"]:
        for lv, by_es in r["deltas"].items():
            for es, d in sorted(by_es.items()):
                s, c = _summary_deltas(d)
                nf = sum(1 for x in d.values() if x["flag"])
                n = r["nominal_refs_used"].get(f"{lv}@es{es}", 0)
                ref = f"{r['delta_reference']} ({n})" if n else f"{r['delta_reference']} (NO REF)"
                out.append(f"| {r['term']} {r['slug']} {lv} {r['base']} es{es} | {ref} | {f3(s)} | {f3(c)} | {nf} |")
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("cellseed")
    a1.add_argument("name")
    a1.add_argument("es", type=int)
    a1.add_argument("--term", default=None)
    a1.add_argument("--level", type=float, default=None)
    a1.add_argument("--base", default="L000")
    a1.add_argument("--policy", default="student", choices=("student", "teacher"))
    for cmd in ("baseline", "es2plan", "table"):
        p = sub.add_parser(cmd)
        p.add_argument("--out", required=True)
        if cmd == "table":
            p.add_argument("--md", default=None)
    a = ap.parse_args()
    if a.cmd == "cellseed":
        term = C.TERM[a.term] if a.term else None
        print(json.dumps(cellseed(a.name, a.es, term, a.level, a.base, a.policy), indent=1, default=str))
        return 0
    if a.cmd == "baseline":
        res = baseline()
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        print(f"DRS BASELINE stop={res['stop']} split={res['split']} reasons={res['reasons']}", flush=True)
        return 3 if res["stop"] else 0
    reads = grid_reads()
    if a.cmd == "es2plan":
        plan = es2_plan(reads)
        Path(a.out).write_text(json.dumps(plan, indent=1))
        print(f"DRS ES2PLAN n={len(plan)}", flush=True)
        return 0
    table = term_table(reads)
    Path(a.out).write_text(json.dumps({**table, "reads": reads}, indent=1, default=str))
    if a.md:
        write_md(table, Path(a.md))
    print(f"DRS TABLE rows={len(table['rows'])} closing_flag={table['closing_flag']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Calibration driver: run the tool's own run_gate() at many parameter settings in ONE process, with the
eval stamps and economy primitives cached (they do not depend on the stroke/centring parameters).

usage: calib.py <config.json> <out_root> <tag>=<k=v,k=v> [<tag>=...] [--grid] [--no-p]
Each setting's full gate report goes to <out_root>/<tag>/ (replay_gate.json + REPLAY_GATE.md); a one-line
summary per setting is printed. Without --grid the R9 neighbour grid is skipped (a no-op grid entry).
"""
import copy
import json
import sys
import time
from pathlib import Path
import importlib.util

TOOL = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/"
            "parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py")
spec = importlib.util.spec_from_file_location("rsp_calib", TOOL)
RSP = importlib.util.module_from_spec(spec)
sys.modules["rsp_calib"] = RSP
spec.loader.exec_module(RSP)

_stamps = {}
_prims = {}
_orig_load = RSP.load_stamp
_orig_prims = RSP.econ_primitives


def load_cached(path, *, limit=None, heading_guard=True):
    key = (str(path), limit)
    if key not in _stamps:
        _stamps[key] = _orig_load(path, limit=limit, heading_guard=heading_guard)
    st = _stamps[key]
    st.notes = list(dict.fromkeys(st.notes))
    return st


def prims_cached(st, econ, info):
    key = st.run_dir
    if key not in _prims:
        _prims[key] = _orig_prims(st, econ, info)
    return _prims[key]


RSP.load_stamp = load_cached
RSP.econ_primitives = prims_cached

_captured = {}
_orig_replay = RSP.replay


def replay_capture(st, params=None, *, dup=None):
    rp = _orig_replay(st, params, dup=dup)
    if dup is None and st.run_dir not in _captured:
        _captured[st.run_dir] = rp
    return rp


RSP.replay = replay_capture
sys.path.insert(0, str(Path(__file__).parent))
import restated as RS  # noqa: E402


def restated_for(cfg, rep):
    roles = {}
    for key in ("healthy", "sideways"):
        for n, pth in (cfg.get(key) or {}).items():
            st = load_cached(pth)
            roles[n] = (key, st, _captured[st.run_dir])
    return {"R1r": RS.r1r(RSP, roles), "R2r": RS.r2r(rep), "R3r": RS.r3r(RSP, roles), "R6r": RS.r6r(rep),
            "R8r": RS.r8r(RSP, roles), "P1r": RS.p1r(rep)}


def r9r(cfg, base_params, grid):
    """R9 under the restated R1-R3: every neighbour of the base re-replayed on healthy + sideways."""
    base = RSP.parse_params([f"{k}={v}" for k, v in base_params.items()])
    out = {}
    runs = {}
    for key in ("healthy", "sideways"):
        for n, pth in (cfg.get(key) or {}).items():
            st = load_cached(pth)
            runs[n] = RSP.GateRun(name=n, role=key, st=st)
    for pname, values in grid.items():
        for v in values:
            val = getattr(base, pname) + v if pname in RSP._OFFSET_PARAMS else v
            if val == getattr(base, pname):
                continue
            prm = base.with_(**{pname: (int(val) if pname in ("n_off", "min_loaded") else float(val))})
            nrp = {n: _orig_replay(g.st, prm) for n, g in runs.items()}
            for n, g in runs.items():
                g.rp = nrp[n]
                g.vals = RSP.episode_values(g.st, g.rp)
            tool = RSP._r123(runs, nrp, prm)
            roles = {n: (g.role, g.st, nrp[n]) for n, g in runs.items()}
            r1 = RS.r1r(RSP, roles)
            r3 = RS.r3r(RSP, roles, B=400)
            r2ok = bool(tool["R2"]["clauses"]["slide_kept<=0.25"] and tool["R2"]["clauses"]["side<=0.2xsideways"] is not False)
            r3ok = bool(tool["R3"]["clauses"]["short<=2%"] and r3["pass"])
            worst_r1 = min(r1["rows"].items(), key=lambda kv: kv[1]["kept_planted"] or 0)
            worst_r3 = max(((k, v["ci95"][0]) for k, v in r3["rows"].items() if v["ci95"][0] is not None), key=lambda kv: kv[1])
            tag = f"{pname}={val:g}"
            out[tag] = {"R1r": r1["pass"], "R2r": r2ok, "R3r": r3ok, "pass": bool(r1["pass"] and r2ok and r3ok),
                        "worst_R1r": [worst_r1[0], worst_r1[1]["kept_planted"]],
                        "R2_slide_kept": tool["R2"]["slide_kept"], "R2_side_vs_sideways_max": max(
                            v for v in tool["R2"]["side_vs_sideways"].values() if v is not None),
                        "R3a_max_short": max(v for v in tool["R3"]["short_share"].values() if v is not None),
                        "R3b_max_ci_lo": list(worst_r3),
                        "literal": {k: tool[k]["pass"] for k in ("R1", "R2", "R3")}}
            print(f"  R9r {tag}: {'PASS' if out[tag]['pass'] else 'FAIL'} R1r {r1['pass']} ({worst_r1[0]} {worst_r1[1]['kept_planted']:.3f}) "
                  f"R2r {r2ok} R3r {r3ok} (max CI lo {worst_r3[0]} {worst_r3[1]:+.3f}) literal {out[tag]['literal']}", flush=True)
    return out


def summary(rep):
    c = rep["criteria"]
    out = {"w_S": rep["w_S"], "w_P": rep["w_P"]}
    for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "P1", "P2", "P3", "P4"):
        v = c.get(k) or {}
        out[k] = v.get("pass")
        out[k + "_hl"] = v.get("headline")
    return out


def run(cfg_path, out_root, settings, grid=False):
    cfg0 = json.loads(Path(cfg_path).read_text())
    res = {}
    for tag, kv in settings:
        cfg = copy.deepcopy(cfg0)
        params = dict(cfg.get("params") or {})
        for pair in filter(None, kv.split(",")):
            k, v = pair.split("=")
            params[k] = v
        cfg["params"] = params
        if not grid:
            cfg["grid"] = {"rho0": [0.0]}
        t0 = time.time()
        _captured.clear()
        rep = RSP.run_gate(cfg, Path(out_root) / tag, log=lambda *a, **k: None)
        s = summary(rep)
        rs = restated_for(cfg, rep)
        (Path(out_root) / tag / "restated.json").write_text(json.dumps(RSP._jsonable(rs), indent=1))
        res[tag] = {"params": params, **s, **{k: v["pass"] for k, v in rs.items()}}
        fails = [k for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "P1", "P2", "P3", "P4")
                 if s[k] is False]
        print(f"[{tag}] {kv or 'defaults'}  ({time.time() - t0:.0f}s) w_S {s['w_S']:.4f} w_P {s['w_P']}  FAIL {fails}",
              flush=True)
        for k in ("R1", "R2", "R3", "R5", "R6", "R8", "P1", "P2", "P3"):
            print(f"    {k}: {s[k + '_hl']}", flush=True)
        print("    restated: " + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in rs.items()), flush=True)
        w1 = min(rs["R1r"]["rows"].items(), key=lambda kv: kv[1]["kept_planted"] or 0)
        print(f"    R1r worst {w1[0]} {w1[1]['kept_planted']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    grid = "--grid" in args
    do_r9r = "--r9r" in args
    args = [a for a in args if a not in ("--grid", "--r9r")]
    cfg_path, out_root = args[0], args[1]
    settings = []
    for a in args[2:]:
        tag, kv = a.split("=", 1)
        settings.append((tag, kv))
    res = run(cfg_path, out_root, settings, grid=grid)
    if do_r9r:
        tag, kv = settings[0]
        bp = dict(json.loads(Path(cfg_path).read_text()).get("params") or {})
        for pair in filter(None, kv.split(",")):
            k, v = pair.split("=")
            bp[k] = v
        r9 = r9r(json.loads(Path(cfg_path).read_text()), bp, RSP.DEFAULT_GRID)
        (Path(out_root) / tag / "r9r.json").write_text(json.dumps(RSP._jsonable(r9), indent=1))
        print(f"[R9r] {tag}: {'PASS' if all(v['pass'] for v in r9.values()) else 'FAIL'}; failing: "
              f"{[k for k, v in r9.items() if not v['pass']]}", flush=True)
    p = Path(out_root) / "calib_summary.jsonl"
    with p.open("a") as f:
        for tag, v in res.items():
            f.write(json.dumps({"tag": tag, **RSP._jsonable(v)}) + "\n")

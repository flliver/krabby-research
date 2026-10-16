#!/usr/bin/env python3
"""PLAN step 2 read (CPU): P-config, P-early and A3, under the rules pinned in CHARTER §3 (2026-10-16, step 2).

- P-config: each Phase-A smoke's trainer-written params/{env,agent}.yaml against the lineage run dir of the same
  stage (X43 s1, X57 c4, X60 c5/c6/c6x, X08 p2a). Differences are classified as allowed or not.
- P-early: the PF_s1 tfevents against X43's at steps 0-19, tag by tag, bitwise.
- A3: the realised added mass, the zero-action settle and the suite survival per mass. Also the bonus bitwise
  0-kg suite row against REFS_S1_s18 es1.

    t4_stepA_read.py [--json OUT] [--md OUT]
Exit 0 always; the verdicts are in the output (the orchestrator reads them).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
C = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
LOGS = R / "parkour/logs/rsl_rl"
OUT = R / "parkour/outputs/omni_plan/task4/lr"
LINEAGE = {
    "s1": LOGS / "crab_hex_omni_flat_walk/2026-10-04_10-01-39_seed18_X43_s1_A15pB",
    "c4": LOGS / "crab_hex_omni_flat_walk/2026-10-05_17-41-16_seed18_X57_c4_A15pB",
    "c5": LOGS / "crab_hex_omni_flat_walk/2026-10-06_00-56-48_seed18_X60_c5_A15pB",
    "c6": LOGS / "crab_hex_omni_flat_walk/2026-10-06_01-30-49_seed18_X60_c6_A15pB",
    "c6x": LOGS / "crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB",
    "p2a": LOGS / "crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB",
}
REFS_ES1 = R / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24"
BODY0 = 158.75732421875
MASSES = (0, 45, 90, 135, 180)
ALLOWED_KEY = re.compile(r"(^|\.)(max_iterations|resume|load_run|load_checkpoint|run_name|seed|log_dir|"
                         r"experiment_name)$")
PATHLIKE = re.compile(r"(^/|\.pt$|\.usda?$|\.npz$|\.json$|\.ya?ml$|\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)")
EARLY_TOL = {"Loss/priv_reg": 2e-6}


class _AnyLoader(yaml.SafeLoader):
    pass


def _any(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_AnyLoader.add_multi_constructor("tag:yaml.org,2002:python/", _any)
_AnyLoader.add_multi_constructor("!", _any)


def flat(d, pre="") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d):
            out[pre] = tuple(d)
        else:
            for i, v in enumerate(d):
                out.update(flat(v, f"{pre}[{i}]"))
    else:
        out[pre] = d
    return out


def _default_off(v) -> bool:
    return v in (None, False, 0, 0.0, "", "0", "off", "none", (), [])


def pconfig(new: Path, old: Path) -> dict:
    res = {"new": str(new), "old": str(old), "files": {}}
    ok = True
    for f in ("env.yaml", "agent.yaml"):
        a = flat(yaml.load((old / "params" / f).read_text(), Loader=_AnyLoader))
        b = flat(yaml.load((new / "params" / f).read_text(), Loader=_AnyLoader))
        bad, allowed = [], []
        for k in sorted(set(a) | set(b)):
            va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
            if va == vb:
                continue
            why = None
            if ALLOWED_KEY.search(k):
                why = "allowed key"
            elif va == "<absent>" and _default_off(vb):
                why = "new key, default-off"
            elif all(isinstance(x, str) and PATHLIKE.search(x) for x in (va, vb) if x != "<absent>"):
                why = "path / run name / timestamp"
            (allowed if why else bad).append({"key": k, "old": repr(va)[:160], "new": repr(vb)[:160],
                                              **({"why": why} if why else {})})
        res["files"][f] = {"n_keys_old": len(a), "n_keys_new": len(b), "allowed": allowed, "not_allowed": bad}
        ok = ok and not bad
    res["pass"] = ok
    return res


def _scalars(run: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)} for t in ea.Tags()["scalars"]}


def pearly(new: Path, old: Path, steps=range(20)) -> dict:
    a, b = _scalars(old), _scalars(new)
    shared = sorted(set(a) & set(b))
    res = {"new": str(new), "old": str(old), "tags_old": len(a), "tags_new": len(b), "shared": len(shared),
           "only_old": sorted(set(a) - set(b)), "only_new": sorted(set(b) - set(a)), "bitwise": 0, "differ": []}
    ok = True
    for t in shared:
        st = [s for s in steps if s in a[t] and s in b[t]]
        if not st:
            continue
        va = np.array([a[t][s] for s in st], dtype=np.float32)
        vb = np.array([b[t][s] for s in st], dtype=np.float32)
        if np.array_equal(va, vb, equal_nan=True):
            res["bitwise"] += 1
            continue
        d = float(np.nanmax(np.abs(va.astype(np.float64) - vb.astype(np.float64))))
        ulp = int(np.nanmax(np.abs(va.view(np.int32).astype(np.int64) - vb.view(np.int32).astype(np.int64))))
        # "*/time" tags (Train/mean_reward/time, Train/mean_episode_length/time) are logged with wall-clock seconds
        # as their step, so they are timing, like Perf/* (CHARTER step-2 note, recorded after the first read)
        timing = t.startswith("Perf/") or t.endswith("/time")
        tolerated = (timing or (t in EARLY_TOL and d <= EARLY_TOL[t])
                     or ("hist_latent" in t and ulp <= 1))
        res["differ"].append({"tag": t, "max_abs": d, "max_ulp": ulp, "tolerated": tolerated, "timing": timing,
                              "first_step": int(st[int(np.argmax(va != vb))])})
        ok = ok and tolerated
    res["pass"] = ok
    return res


def _newest(p: Path) -> Path | None:
    c = sorted(p.glob("*/seed001/*/scenario_metrics.json"), key=lambda x: x.stat().st_mtime)
    return c[-1].parent if c else None


def a3() -> dict:
    out = {}
    for m in MASSES:
        row = {}
        for kind in ("settle", "suite"):
            st = _newest(OUT / f"A3_{kind}_{m}kg")
            if st is None:
                row[kind] = {"missing": True}
                continue
            meta = json.loads((st / "run_meta.json").read_text())
            sm = json.loads((st / "scenario_metrics.json").read_text())
            body = float(meta["body_masses_kg"][0])
            r = {"stamp": str(st), "body_mass": body, "added": body - BODY0,
                 "mass_ok": abs(body - BODY0 - m) <= 0.01,
                 "dr_mass_env": meta["env_vars"].get("KRABBY_DR_MASS")}
            eps = sm.get("episodes") or []
            if kind == "settle":
                # the harness records a clean finish as "schedule_complete"; anything else ended the episode early
                term = [e for e in eps if e.get("termination_reason") != "schedule_complete"]
                r["n_episodes"] = len(eps)
                r["n_terminated"] = len(term)
                r["termination_reasons"] = sm["aggregate"].get("termination_reasons")
                r["tips"] = len(term) > 0
                tilt, sag = [], []
                for z in sorted((st / "raw").glob("episode_*.npz")):
                    a = np.load(z)
                    q = a["root_quat_w"].astype(np.float64)  # (T, 4) w, x, y, z
                    up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                    tilt.append(float(np.degrees(np.arccos(np.clip(up_z, -1, 1))).max()))
                    h = a["root_pos_w"][:, 2].astype(np.float64)
                    sag.append(float(h[: max(1, len(h) // 10)].mean() - h[-max(1, len(h) // 10):].mean()))
                r["max_tilt_deg"] = max(tilt) if tilt else None
                r["mean_height_sag_m"] = float(np.mean(sag)) if sag else None
            else:
                sv = sm["aggregate"]["suite"]["survival_by_hold"]
                vals = [float(v) for v in (sv.values() if isinstance(sv, dict) else sv) if v is not None]
                r["survival_mean"] = float(np.mean(vals)) if vals else None
                r["survival_by_hold"] = sv
                r["completion"] = sm["aggregate"].get("schedule_completion_rate")
            row[kind] = r
        out[f"{m}kg"] = row
    # the knob is inert unless every non-zero mass shows in run_meta
    inert = any(not out[f"{m}kg"].get(k, {}).get("mass_ok", False)
                for m in MASSES for k in ("settle", "suite") if not out[f"{m}kg"].get(k, {}).get("missing"))
    tips_le90 = any(out[f"{m}kg"].get("settle", {}).get("tips", False) for m in (0, 45, 90))
    s90 = out["90kg"].get("suite", {}).get("survival_mean")
    out["verdict"] = {
        "knob_inert": inert,
        "settle_tips_le_90": tips_le90,
        "suite_survival_90": s90,
        "stop_a": (not inert) and tips_le90 and (s90 is not None and s90 < 0.5),
        "status": "INCONCLUSIVE (knob inert)" if inert else ("STOP-A" if (tips_le90 and s90 is not None
                                                                            and s90 < 0.5) else "PASS"),
    }
    st0 = _newest(OUT / "A3_suite_0kg")
    if st0 is not None:
        spec = importlib.util.spec_from_file_location(
            "_bcb", R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/"
                        "scripts/bc_bitwise.py")
        bcb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bcb)
        cmp = bcb.compare(REFS_ES1, st0, set())
        out["bonus_refs_bitwise"] = {"metrics_equal": cmp.get("metrics_equal"),
                                     "n_value_mismatch": len(cmp.get("value_mismatch") or {}),
                                     "n_key_mismatch": len(cmp.get("key_mismatch") or {}),
                                     "missing_episodes": len(cmp.get("missing_episodes") or [])}
    return out


def smoke_runs() -> dict:
    """stage -> the PF smoke's run dir, from the driver's own record of each reference head (orchestrator.log
    "reference PF_<stage>_s18 s18 recorded (Xnn, head <path>)"). Never by run name: other campaigns reuse X ids."""
    runs = {}
    log = C / "preflight/orchestrator.log"
    if log.is_file():
        for m in re.finditer(r"reference PF_(\w+?)_s18 s18 recorded \((X\d+), head (\S+?)\)", log.read_text()):
            runs[m.group(1)] = Path(m.group(3)).parent
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(C / "preflight/stepA_read.json"))
    ap.add_argument("--md", default=str(C / "preflight/stepA_read.md"))
    a = ap.parse_args()
    runs = smoke_runs()
    res = {"smoke_runs": {k: str(v) for k, v in runs.items()}, "pconfig": {}, "pearly": None}
    for stage, old in LINEAGE.items():
        if stage in runs:
            res["pconfig"][stage] = pconfig(runs[stage], old)
    if "s1" in runs:
        res["pearly"] = pearly(runs["s1"], LINEAGE["s1"])
    res["a3"] = a3()
    pc_ok = bool(res["pconfig"]) and all(v["pass"] for v in res["pconfig"].values()) and len(res["pconfig"]) == 6
    pe_ok = bool(res["pearly"] and res["pearly"]["pass"])
    res["verdict"] = {"pconfig": pc_ok, "pearly": pe_ok, "a3": res["a3"]["verdict"]["status"],
                      "stop_a": (not pc_ok) or (not pe_ok) or res["a3"]["verdict"]["stop_a"]}
    Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    L = ["# Step 2 (Phase A) read", "", f"Verdict: P-config {'PASS' if pc_ok else 'FAIL'}; P-early "
         f"{'PASS' if pe_ok else 'FAIL'}; A3 {res['a3']['verdict']['status']}; "
         f"STOP-A {'YES' if res['verdict']['stop_a'] else 'no'}", "", "## P-config", ""]
    for st, v in res["pconfig"].items():
        L.append(f"- **{st}** {'PASS' if v['pass'] else 'FAIL'}: " + "; ".join(
            f"{f} {len(x['allowed'])} allowed / {len(x['not_allowed'])} not" for f, x in v["files"].items()))
        for f, x in v["files"].items():
            for d in x["not_allowed"][:12]:
                L.append(f"  - NOT ALLOWED {f} `{d['key']}`: {d['old']} → {d['new']}")
    pe = res["pearly"]
    if pe:
        L += ["", "## P-early (s1, steps 0–19)", "",
              f"- shared tags {pe['shared']} (X43 {pe['tags_old']}, new {pe['tags_new']}); bitwise {pe['bitwise']}"
              f"; differ {len(pe['differ'])} → {'PASS' if pe['pass'] else 'FAIL'}"]
        for d in pe["differ"]:
            L.append(f"  - `{d['tag']}` max|Δ| {d['max_abs']:.3g} ({d['max_ulp']} ulp) from step {d['first_step']}"
                     f" — {'tolerated' if d['tolerated'] else 'NOT tolerated'}")
        if pe["only_old"] or pe["only_new"]:
            L.append(f"  - only X43: {pe['only_old'][:8]}; only new: {pe['only_new'][:8]}")
    L += ["", "## A3 (S1 + fixed added mass)", "", "| mass | added (run_meta) | settle: terminated / n | settle max "
          "tilt ° | settle sag m | suite survival | completion |", "|---|---|---|---|---|---|---|"]
    nan = float("nan")
    for m in MASSES:
        r = res["a3"][f"{m}kg"]
        se, su = r.get("settle", {}), r.get("suite", {})
        L.append(f"| {m} kg | {se.get('added', nan):.2f} / {su.get('added', nan):.2f} | "
                 f"{se.get('n_terminated', '–')} / {se.get('n_episodes', '–')} | "
                 f"{(se.get('max_tilt_deg') if se.get('max_tilt_deg') is not None else nan):.2f} | "
                 f"{(se.get('mean_height_sag_m') if se.get('mean_height_sag_m') is not None else nan):.4f} | "
                 f"{(su.get('survival_mean') if su.get('survival_mean') is not None else nan):.3f} | "
                 f"{su.get('completion', '–')} |")
    L += ["", f"A3 verdict: {res['a3']['verdict']}"]
    if "bonus_refs_bitwise" in res["a3"]:
        L.append(f"Bonus (0-kg suite vs REFS_S1_s18 es1): {res['a3']['bonus_refs_bitwise']}")
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

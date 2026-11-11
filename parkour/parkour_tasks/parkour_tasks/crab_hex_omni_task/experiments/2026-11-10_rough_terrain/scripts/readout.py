#!/usr/bin/env python3
"""The rough-terrain campaign's read-out (PLAN step 4 (c) / step 7 / step 9; CPU; saved eval stamps only).

    readout.py rows      --sha SHA [--es N ...] [--row ID ...]          per-row statistics of one policy
    readout.py baseline  --tdr SHA                                     TDR's step-6 table (es3 kind rows, retention)
    readout.py gates     --new SHA --tdr SHA [--thresholds JSON]       G per kind, R, the decision (PLAN Gates)
    readout.py validate-g1    [--g1 JSON]                             (e1) TDR's recorded G1 numbers to 1e-12
    readout.py validate-site                                          (e2) SITE's completed_and_crossed on T* rows
    readout.py calibrate                                              (e3) radius calibration on band-free TDR rows
    readout.py validate                                               e1 + e2 + e3
    common: [--root LR_DIR ...] [--out JSON] [--dup earliest|latest|error]

ROWS ARE KEYED BY CHECKPOINT SHA256 (``run_meta.json`` ``checkpoint_sha256``), never by a row-name prefix (the
rod-speed campaign's ``RSDT_72d63d51_*`` rows are the no-DR control TCTL, not TDR). A logical row (``rows_rt.json``,
``rt_spec``) is found as the complete stamp (``tp_row.json`` written last) whose run_meta says: that sha256, that
``scenario_id``, that env seed, the row's composed env (``run_meta.env_vars`` == env file + level file + manifest
block, ``rt_spec.expected_env``) and whose ``tp_row.args`` are the row's args. Two such stamps (a closing / relaunch
row of the same composition) are resolved by ``--dup`` (default: the earliest, the first launch) and recorded.

PER-ROW STATISTICS (kind rows; field rows report the same, crossing off):
* completion = ``termination_reason == schedule_complete`` (thresholds.field_episodes);
* r_max = the episode's maximum HORIZONTAL (xy) root distance from its spawn, the spawn being the first raw
  ``root_pos_w`` sample (raw/episode_NN.npz, every policy step, length == n_steps asserted);
* crossed = r_max >= the kind's radius (rt_spec.CROSS_R: ledges / pits / rubble 3.2 m, slopes / hills / gravel
  3.0 m); success = complete AND crossed; stuck = complete AND NOT crossed;
* the r_max distribution (all episodes and completers), termination reasons, the harness fall classes, slip
  (aggregate slip_ratio median), and on slopes rows the up (sunk pad) / down (raised pad) share from the spawn pad
  level (mean ``terrain_z`` under the six feet at step 0).

GATES (PLAN "Gates", frozen in thresholds_rough_v1.json by make_thresholds.py; statistics from the repo's own
thresholds.py / select_student.py / gate_read.py, never re-implemented):
* SE = thresholds.paired_completion_se on {(env seed, env_index): 0/1} (the env_index-paired bootstrap), CI95 =
  diff +- 1.96 SE; a pairing also asserts equal key sets and equal spawns (xy within 1e-4 m) between the two sides;
* G per gated kind, gate band, es1-2 (200), success: gain = new - TDR >= max(0.10, 3 SE) AND CI lower >= 0.05;
  regress = new - TDR < -max(0.05, 2 SE). G passes iff >= k (= 2) kinds gain and none regresses;
* R completion rows (field_hard es1-4; field_light, ring_obst, field_turn, field_light 90 / 135 kg, x0.7 L090
  field_light es1-2) and the solved kinds' R bands (success): new >= TDR - max(0.05, 2 SE)
  (select_student.completion_margin with floor 0.05, se_mult 2, split 0 -- the sel stage's completion_margin);
* R tracking-and-turning = gate_read.zero_kg_check(new, TDR, ['fwd','side','back','diag'], thresholds v1.2,
  its_past_p2a = 0 (drift term 0, passed explicitly), Task 4's resolution files): its decisive suite rows
  (zero_kg_decisive_stats(..., 'track_err'): track_err per translate category, the move_turn ratio drop, survival
  per translate category, turn facing_closed and yaw_progress), each line max(R, 3 SE) with its own v1.2 R, and
  side22 FAST/FAST. zero_kg_check's field completion lines (v1.2's looser margin) are reported, not decisive (PLAN
  item 19: the R completion rows above are);
* differences of 0.04 or less in a success / completion share are never read as effects (flagged ``le_noise``;
  every margin above is >= 0.05 so no pass / fail turns on them); the tracking lines use only their v1.2 margins.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rt_spec as S  # noqa: E402

PIPE = S.EXPS / "tools" / "pipeline"
T4 = S.EXPS / "2026-10-16_task4_pipeline"
RSD_CAMP = S.EXPS / "2026-11-07_rod_speed_dr"
TH_V12 = T4 / "thresholds_v1_2.json"
TH_INPUTS = T4 / "threshold_inputs.yaml"
MAIN_OUT = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs")
#: the campaign's eval output root (steps 5-9); override with RT_TP_OUT
RT_LR = Path(os.environ.get("RT_TP_OUT") or MAIN_OUT / "omni_plan" / "rough_terrain") / "lr"
RSD_LR = MAIN_OUT / "omni_plan" / "rod_speed_dr" / "lr"
SITE_LR = MAIN_OUT / "site_teacher" / "rows" / "lr"
SITE_CAMP = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/sen0581-cameras/parkour/parkour_tasks/"
                 "parkour_tasks/crab_hex_omni_task/experiments/2026-10-28_site_teacher")
THRESHOLDS = S.CAMP / "thresholds_rough_v1.json"
TDR_SHA = "03c3e4728ac2175b4fbff66bc3944134c28fcbec0f6d8cc4aee4e5f3640c8567"
TSTAR_SHA = "03b6009a6e952133f6c21c36c8d315b76f945d431615ef6d6798b7f4a5f59fc7"
TCTL_SHA = "72d63d51daaa9e517c964804d1ddf993f1fab79c0c6a818d9f98e49680dc6732"   # the no-DR control (NOT TDR)
EPS = 1e-12
Z95 = 1.96
NOISE_EFFECT = 0.04
SPAWN_PAIR_TOL_M = 1e-4
SITE_SPAWN_TOL_M = 0.5
K_GATED = 2
KIND_ROW_EPISODES = 100              # PLAN Eval: every kind row, per env seed, on both sides of a G / R-solved pair
FROZEN_SCRIPTS = ("make_thresholds.py", "readout.py", "rt_spec.py")   # the sources whose sha256 the thresholds file records
G_RULE = {"gain_floor": 0.10, "gain_se_mult": 3.0, "ci_lb": 0.05, "regress_floor": 0.05, "regress_se_mult": 2.0}
R_COMPLETION_CFG = {"floor": 0.05, "se_mult": 2.0, "manifest_shape_split": 0.0}
ZERO_KG_TRANSLATE = ["fwd", "side", "back", "diag"]
TRACK_STATISTIC = "track_err"        # R tracking: thresholds v1.2's translate statistic (fixed by TH_V12; asserted)
TRACK_THRESHOLDS_VERSION = "v1.2"
SIDE22_FAST = ("FAST", "FAST")
ITS_PAST_R = 0                       # PLAN item 7 / 19: teacher vs teacher, drift term 0 (passed explicitly)
ITS_PAST_DRIFT_READ = 6000           # item 7: an R line that fails is re-read against v1.2's drift at 6000 its
PAD_LEVEL_FLAT_M = 0.01
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class KeySetError(Exception):
    """An asserted pairing (equal key sets, equal spawns, episode count) failed -> INVALID."""


# ============================================================================== pipeline modules (lazy, by path)
_M: dict = {}


def _load(path: Path, name: str):
    import importlib.util  # noqa: PLC0415

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def gr():
    """tools/pipeline/gate_read.py (it registers tp_lib); tp_lib configured from Task 4's pipeline.yaml (side22)."""
    if "gr" not in _M:
        mod = _load(PIPE / "gate_read.py", "tp_gate_read")
        mod.L.configure_from(T4 / "pipeline.yaml")
        _M["gr"] = mod
    return _M["gr"]


def th():
    return gr().L.thresholds_mod()


def ss():
    if "ss" not in _M:
        _M["ss"] = gr().L.load_by_path(PIPE / "select_student.py", "tp_select_student")
    return _M["ss"]


def th_view():
    return gr().Thresholds.load(TH_V12)


def resolution_files() -> list:
    return list(gr().L.load_yaml(TH_INPUTS).get("resolution_files") or [])


# ============================================================================== small io
def read_json(p: Path, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else default


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    os.replace(tmp, p)


# ============================================================================== the stamp index (keyed by sha256)
def _complete(st: Path) -> bool:
    return all((st / f).is_file() for f in ("run_meta.json", "scenario_metrics.json", "summary.md", "tp_row.json")) \
        and (st / "raw").is_dir()


def stamp_record(st: Path) -> dict:
    meta = json.loads((st / "run_meta.json").read_text())
    return {"stamp": str(st), "sha256": meta.get("checkpoint_sha256"), "scenario": meta.get("scenario_id"),
            "es": (meta.get("env_overrides") or {}).get("seed"), "role": meta.get("role"), "task": meta.get("task"),
            "checkpoint": meta.get("checkpoint"), "env_vars": dict(meta.get("env_vars") or {}),
            "num_envs": meta.get("num_envs"), "suite": bool(meta.get("suite")),
            "tp": read_json(st / "tp_row.json", {}) or {}, "code_state": read_json(st / "code_state.json")}


def scan(roots, scenarios=None, shas=None) -> dict:
    """{(sha256, scenario, es): [stamp records]} over every COMPLETE stamp ``<root>/<name>/<scenario>/seedNNN/<t>/``
    of the given lr roots (``scenarios`` / ``shas``: optional filters). The row-name directory is never read."""
    idx: dict = collections.defaultdict(list)
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for nd in sorted(p for p in root.iterdir() if p.is_dir()):
            for sd in sorted(p for p in nd.iterdir() if p.is_dir()):
                if scenarios is not None and sd.name not in scenarios:
                    continue
                for ed in sorted(p for p in sd.iterdir() if p.is_dir() and p.name.startswith("seed")):
                    for st in sorted(p for p in ed.iterdir() if p.is_dir()):
                        if not _complete(st):
                            continue
                        rec = stamp_record(st)
                        if shas is not None and rec["sha256"] not in shas:
                            continue
                        idx[(rec["sha256"], rec["scenario"], rec["es"])].append(rec)
    return dict(idx)


def _arg(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def env_diff(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _rc_ok(tp: dict) -> bool:
    try:
        return int(tp.get("rc", -1)) == 0
    except (TypeError, ValueError):
        return False


def find(idx: dict, row: dict, es: int, sha: str, dup: str = "earliest") -> dict:
    """The stamp of logical ``row`` at env seed ``es`` for checkpoint ``sha`` (status OK / MISSING / BAD)."""
    out = {"row": row["id"], "es": int(es), "sha256": sha, "status": "MISSING", "stamp": None, "problems": [],
           "notes": [], "duplicates": []}
    want_env = S.expected_env(row)
    cands = idx.get((sha, row["scenario"], int(es)), [])
    hits = [c for c in cands if c["env_vars"] == want_env and list(c["tp"].get("args") or []) == list(row["args"])]
    if not hits:
        out["problems"].append("no complete stamp with this sha256 / scenario / env seed / composed env / args")
        for c in cands[:4]:
            out["notes"].append(f"near miss {c['stamp']}: env keys differ {env_diff(c['env_vars'], want_env)[:8]}, "
                                f"args {c['tp'].get('args')}")
        return out
    hits = sorted(hits, key=lambda c: Path(c["stamp"]).name)
    ok_rc = [c for c in hits if _rc_ok(c["tp"])]
    if ok_rc and len(ok_rc) < len(hits):
        # a failed launch (rc != 0) that still wrote tp_row.json never shadows an OK relaunch: dropped BEFORE the
        # duplicate rule (recorded); if EVERY match failed, they stay and the row reads BAD on the rc below
        out["notes"].append(f"dropped {len(hits) - len(ok_rc)} stamp(s) with tp_row rc != 0: "
                            f"{[c['stamp'] for c in hits if c not in ok_rc]}")
        hits = ok_rc
    if len(hits) > 1:
        if dup == "error":
            out["status"] = "BAD"
            out["problems"].append(f"{len(hits)} matching stamps: {[h['stamp'] for h in hits]}")
            return out
        pick = hits[0] if dup == "earliest" else hits[-1]
        out["duplicates"] = [h["stamp"] for h in hits if h is not pick]
        out["notes"].append(f"{len(hits)} matching stamps; read the {dup} ({Path(pick['stamp']).name})")
    else:
        pick = hits[0]
    out["stamp"] = pick["stamp"]
    pr = []
    role = _arg(row["args"], "--policy-role") or "teacher"
    if pick["role"] != role:
        pr.append(f"run_meta.role {pick['role']!r} != {role!r}")
    task = _arg(row["args"], "--task")
    if task and pick["task"] != task:
        pr.append(f"run_meta.task {pick['task']!r} != {task!r}")
    dr = sorted(k for k in pick["env_vars"] if k.startswith(S.DR_PREFIX))
    if dr:
        pr.append(f"eval env carries {dr} (no KRABBY_DR_* on an eval row)")
    if row.get("rt_friction"):
        if pick["env_vars"].get(S.FRICTION_VAR) != row["rt_friction"]:
            pr.append(f"{S.FRICTION_VAR} {pick['env_vars'].get(S.FRICTION_VAR)!r} != {row['rt_friction']!r} "
                      "(the gravel-eval friction contract must not be stripped)")
    elif S.FRICTION_VAR in pick["env_vars"]:
        pr.append(f"{S.FRICTION_VAR} on a row that does not carry it")
    tp = pick["tp"]
    if not _rc_ok(tp):
        pr.append(f"tp_row rc {tp.get('rc')}")
    if int(tp.get("env_seed", es)) != int(es):
        pr.append(f"tp_row env_seed {tp.get('env_seed')} != {es}")
    if tp.get("stripped_keys"):
        pr.append(f"tp_row stripped {tp.get('stripped_keys')} (teacher rows strip nothing)")
    cs = pick["code_state"]
    if cs is None:
        pr.append("no code_state.json")
    else:
        out["code_sha"], out["eval_sha"] = cs.get("code_sha"), cs.get("eval_sha")
        if str(cs.get("dirty")) != "False":
            pr.append(f"code_state dirty {cs.get('dirty')}")
    out["problems"] = pr
    out["status"] = "OK" if not pr else "BAD"
    return out


def build_problems(*found) -> list:
    """One comparison reads ONE build (code_sha, eval_sha) on both sides (SITE's build_problems rule)."""
    builds: dict = {}
    for f in found:
        builds.setdefault((f.get("code_sha"), f.get("eval_sha")), []).append(f"{f['row']} es{f['es']} "
                                                                             f"{str(f['sha256'])[:8]}")
    if len(builds) <= 1:
        return []
    return [f"paired rows at {len(builds)} builds: " + "; ".join(f"{str(c)[:12]}/{str(e)[:12]}: {', '.join(v)}"
                                                               for (c, e), v in builds.items())]


# ============================================================================== per-episode reads
_CACHE: dict = {}


def episode_list(st: Path) -> list:
    """scenario_metrics episodes, asserted: count == unique env_index == run_meta num_envs (suite: env_index_map)."""
    st = Path(st)
    key = ("eps", str(st))
    if key not in _CACHE:
        rm = json.loads((st / "run_meta.json").read_text())
        want = int(rm.get("num_envs") or 0)
        if rm.get("suite"):
            want = len(rm.get("env_index_map") or []) or want
        eps = json.loads((st / "scenario_metrics.json").read_text())["episodes"]
        ids = [int(e["env_index"]) for e in eps]
        if not (len(ids) == len(set(ids)) == want):
            raise KeySetError(f"{st}: {len(ids)} episodes, {len(set(ids))} unique env_index (want {want})")
        _CACHE[key] = eps
    return _CACHE[key]


def completion_of(st: Path) -> dict:
    episode_list(st)
    return th().field_episodes(Path(st))


def raw_episode(st: Path, i: int) -> dict:
    """root xy (float64, every policy step), spawn xyz, spawn pad level (mean terrain_z at step 0), n samples."""
    key = ("raw", str(st), int(i))
    if key not in _CACHE:
        with np.load(Path(st) / "raw" / f"episode_{int(i):02d}.npz", allow_pickle=False) as z:
            r = np.asarray(z["root_pos_w"], np.float64)
            tz = np.asarray(z["terrain_z"], np.float64) if "terrain_z" in z.files else None
        _CACHE[key] = {"xy": r[:, :2], "spawn": r[0].copy(), "n": int(r.shape[0]),
                       "pad_level": (float(tz[0].mean()) if tz is not None and tz.size else None)}
    return _CACHE[key]


def r_max_from(xy: np.ndarray, origin) -> float:
    """Maximum horizontal distance of an xy track from ``origin`` (PLAN item 4: horizontal, from the spawn)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    return float(np.linalg.norm(xy - np.asarray(origin, np.float64).reshape(1, 2), axis=1).max())


def r_max_of(st: Path) -> dict:
    """{env_index: r_max} from the spawn (= the first raw root_pos_w sample); raw length == n_steps asserted."""
    out = {}
    for e in episode_list(st):
        i = int(e["env_index"])
        ep = raw_episode(st, i)
        if ep["n"] != int(e["n_steps"]):
            raise KeySetError(f"{st}: env {i} raw has {ep['n']} samples, n_steps {e['n_steps']}")
        out[i] = r_max_from(ep["xy"], ep["xy"][0])
    return out


def crossed_from(rmax: dict, radius: float) -> dict:
    return {i: bool(v >= float(radius) - EPS) for i, v in rmax.items()}


def success_from(comp: dict, cro: dict) -> dict:
    assert_keys("completion vs crossed", comp, cro)
    return {i: bool(comp[i] and cro[i]) for i in comp}


def stuck_from(comp: dict, cro: dict) -> dict:
    assert_keys("completion vs crossed", comp, cro)
    return {i: bool(comp[i] and not cro[i]) for i in comp}


def assert_keys(name: str, *ds) -> None:
    ks = [set(d) for d in ds]
    if any(k != ks[0] for k in ks[1:]):
        raise KeySetError(f"{name}: key sets differ ({[len(k) for k in ks]} keys; symmetric difference "
                          f"{sorted(set.union(*ks) - set.intersection(*ks), key=str)[:10]})")


def spawns_of(st: Path) -> dict:
    return {int(e["env_index"]): raw_episode(st, int(e["env_index"]))["spawn"][:2] for e in episode_list(st)}


def distribution(vals) -> dict:
    v = np.asarray(list(vals), np.float64)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            **{f"p{int(round(q * 100)):02d}": float(np.quantile(v, q)) for q in QUANTILES}}


def share(d: dict) -> float | None:
    return float(sum(bool(x) for x in d.values()) / len(d)) if d else None


def episode_table(st: Path, radius: float | None) -> dict:
    """Per-episode maps of one stamp: completion, termination reason, n_steps (+ r_max / crossed / success / stuck
    and the spawn pad level when ``radius`` is given)."""
    eps = episode_list(st)
    comp = completion_of(st)
    t = {"completion": comp, "reason": {int(e["env_index"]): str(e["termination_reason"]) for e in eps},
         "n_steps": {int(e["env_index"]): int(e["n_steps"]) for e in eps}}
    if radius is not None:
        rm = r_max_of(st)
        cro = crossed_from(rm, radius)
        t.update(r_max=rm, crossed=cro, success=success_from(comp, cro), stuck=stuck_from(comp, cro),
                 pad_level={i: raw_episode(st, i)["pad_level"] for i in comp})
    return t


def aggregate_of(st: Path) -> dict:
    a = json.loads((Path(st) / "scenario_metrics.json").read_text())["aggregate"]
    return {"fall_classes": a.get("fall_classes"), "termination_reasons": a.get("termination_reasons"),
            "slip_ratio_median": ((a.get("slip_ratio") or {}).get("median"))}


def summarize(tables: dict, kind: str | None = None) -> dict:
    """Pooled per-row statistics over ``{es: episode_table}``."""
    def pooled(key):
        out = {}
        for es, t in sorted(tables.items()):
            out.update({(es, i): v for i, v in t[key].items()})
        return out

    comp = pooled("completion")
    s = {"n": len(comp), "completion": share(comp),
         "termination_reasons": dict(sorted(collections.Counter(pooled("reason").values()).items())),
         "per_es": {f"es{es}": {"n": len(t["completion"]), "completion": share(t["completion"])}
                    for es, t in sorted(tables.items())}}
    if all("r_max" in t for t in tables.values()) and tables:
        rm, cro, suc, stk = pooled("r_max"), pooled("crossed"), pooled("success"), pooled("stuck")
        s.update(crossed=share(cro), success=share(suc), stuck=share(stk),
                 r_max_all=distribution(rm.values()),
                 r_max_completers=distribution(v for k, v in rm.items() if comp[k]))
        for es, t in sorted(tables.items()):
            s["per_es"][f"es{es}"].update(success=share(t["success"]), stuck=share(t["stuck"]),
                                          crossed=share(t["crossed"]))
        if kind == "slopes":
            pads = [v for v in pooled("pad_level").values() if v is not None]
            s["slopes_variant"] = {"up_sunk_pad": int(sum(p < -PAD_LEVEL_FLAT_M for p in pads)),
                                   "down_raised_pad": int(sum(p > PAD_LEVEL_FLAT_M for p in pads)),
                                   "flat": int(sum(abs(p) <= PAD_LEVEL_FLAT_M for p in pads)), "n": len(pads),
                                   "rule": f"sign of the spawn pad level (mean terrain_z at step 0, |z| > "
                                           f"{PAD_LEVEL_FLAT_M} m)"}
    return s


# ============================================================================== paired statistics
def keyed(tables: dict, key: str) -> dict:
    """{(env seed, env_index): value} pooled over ``{es: episode_table}``."""
    out = {}
    for es, t in sorted(tables.items()):
        out.update({(int(es), int(i)): v for i, v in t[key].items()})
    return out


def paired(a: dict, b: dict, what: str = "") -> dict:
    """``a - b`` on equal key sets: shares, diff, SE (thresholds.paired_completion_se), CI95, the noise flag."""
    assert_keys(what or "paired", a, b)
    va, vb = share(a), share(b)
    se = th().paired_completion_se({k: float(bool(v)) for k, v in a.items()},
                                   {k: float(bool(v)) for k, v in b.items()})
    d = va - vb
    return {"a": va, "b": vb, "diff": d, "se": se, "ci95": [d - Z95 * se, d + Z95 * se], "n": len(a),
            "le_noise": abs(d) <= NOISE_EFFECT + EPS}


def g_line(new: dict, ref: dict, what: str = "", rule: dict = G_RULE) -> dict:
    """PLAN Gates G for one kind: gain and regression on the paired success share."""
    p = paired(new, ref, what)
    need = max(rule["gain_floor"], rule["gain_se_mult"] * p["se"])
    reg_m = max(rule["regress_floor"], rule["regress_se_mult"] * p["se"])
    gain = bool(p["diff"] >= need - EPS and p["ci95"][0] >= rule["ci_lb"] - EPS and not p["le_noise"])
    regress = bool(p["diff"] < -reg_m - EPS and not p["le_noise"])
    return {"new": p["a"], "tdr": p["b"], "diff": p["diff"], "se": p["se"], "ci95": p["ci95"], "n": p["n"],
            "gain_need": need, "ci_lb": p["ci95"][0], "gain": gain, "regress_margin": reg_m, "regress": regress,
            "le_noise": p["le_noise"]}


def retention_line(head: dict, ref: dict, row: str = "", what: str = "", cfg: dict = R_COMPLETION_CFG) -> dict:
    """PLAN Gates R (completion / success): head >= ref - max(0.05, 2 SE_diff), select_student.completion_pairs'
    margin (``completion_margin`` with the sel stage's cfg) and record shape (``head`` / ``ref`` for its
    student / tstar), as the rod-speed G1 read recorded it (rsd_lib.paired_completion)."""
    assert_keys(what or row, head, ref)
    a = {k: float(bool(v)) for k, v in head.items()}
    b = {k: float(bool(v)) for k, v in ref.items()}
    pa, pb = sum(a.values()) / len(a), sum(b.values()) / len(b)
    se = th().paired_completion_se(a, b)
    m = ss().completion_margin(se, cfg)
    out = {"row": row, "se_diff": se, **m, "line": pb - m["margin"], "diff": pa - pb, "head": pa, "ref": pb,
           "n": len(a)}
    out["noninferior"] = bool(pa >= pb - m["margin"] - 1e-12)
    out["beats_by_m"] = bool(out["diff"] >= m["margin"] - 1e-12)
    out["pass"] = out["noninferior"]
    out["le_noise"] = abs(out["diff"]) <= NOISE_EFFECT + EPS
    return out


def g_gate(lines: dict, k: int = K_GATED) -> dict:
    gains = sorted(kd for kd, ln in lines.items() if ln.get("gain"))
    regs = sorted(kd for kd, ln in lines.items() if ln.get("regress"))
    return {"k": k, "n_gated": len(lines), "n_gain": len(gains), "gain_kinds": gains, "regressed": regs,
            "status": "PASS" if (len(gains) >= k and not regs) else "FAIL"}


def r_tracking(zk: dict) -> dict:
    """The decisive suite lines + side22 FAST/FAST of a zero_kg_check result (its field completion lines are
    reported only: PLAN item 19).

    Fail closed on the frozen semantics: the zero_kg_check must have read ``track_err`` under thresholds ``v1.2``
    (``TRACK_STATISTIC`` / ``TRACK_THRESHOLDS_VERSION``), else the status is INVALID. side22 is read TWICE and both
    must hold: (relative) zero_kg_judge's v1.2 line, 'FAST/FAST not lost relative to TDR'; and (absolute) the NEW
    teacher's own side22 class is FAST/FAST -- the two coincide only while TDR is FAST/FAST (as in G1)."""
    if zk.get("translate_statistic") != TRACK_STATISTIC or zk.get("thresholds_version") != TRACK_THRESHOLDS_VERSION:
        why = (f"zero_kg_check read translate_statistic {zk.get('translate_statistic')!r} under thresholds "
               f"{zk.get('thresholds_version')!r}; this campaign's R tracking is {TRACK_STATISTIC!r} / "
               f"{TRACK_THRESHOLDS_VERSION!r} (fail closed)")
        return {"status": "INVALID", "fails": [why], "n_suite_lines": 0, "suite_lines": [],
                "side22": zk.get("side22") or {}, "field_lines_report_only": [],
                "its_past_p2a": zk.get("its_past_p2a"), "thresholds_version": zk.get("thresholds_version"),
                "translate_statistic": zk.get("translate_statistic")}
    suite = [r for r in zk.get("rows") or [] if not r.get("watch") and str(r.get("row", "")).startswith("suite")]
    field = [r for r in zk.get("rows") or [] if not r.get("watch") and not str(r.get("row", "")).startswith("suite")]
    fails = [f"{r['statistic']} {r['row']}" for r in suite if r.get("fails")]
    s22 = zk.get("side22") or {}
    pre = [f for f in zk.get("fails") or [] if "suite env seeds differ" in f or "no zero_kg_paired" in f
           or "translate_statistic" in f or "side22 stamp missing" in f]
    if s22.get("fast_fast_lost"):
        fails.append(f"side22 FAST/FAST lost: new {s22.get('head')} vs TDR {s22.get('p2a')}")
    if list(s22.get("head") or []) != list(SIDE22_FAST):
        fails.append(f"side22 absolute: the new teacher's class is {s22.get('head')}, not {list(SIDE22_FAST)}")
    fails += pre
    n_dec = len(suite)
    return {"status": "PASS" if not fails and n_dec else "FAIL", "fails": fails, "n_suite_lines": n_dec,
            "suite_lines": suite, "side22": s22,
            "side22_reading": "relative (zero_kg_judge v1.2: FAST/FAST not lost vs TDR) AND absolute (the new "
                              "teacher's class is FAST/FAST)",
            "field_lines_report_only": field,
            "its_past_p2a": zk.get("its_past_p2a"), "thresholds_version": zk.get("thresholds_version"),
            "translate_statistic": zk.get("translate_statistic")}


# ============================================================================== rows -> tables
class Rows:
    """Rows of rows_rt.json + the stamp index of the lr roots."""

    def __init__(self, roots, rows_doc: dict | None = None, dup: str = "earliest", shas=None):
        self.doc = rows_doc or S.load_rows()
        self.kind = {r["id"]: r for r in self.doc["kind_rows"]}
        self.ret = {r["id"]: r for r in self.doc["retention_rows"]}
        self.all = {**self.kind, **self.ret}
        self.dup = dup
        self.roots = [Path(r) for r in roots]
        scen = {r["scenario"] for r in self.all.values()}
        self.idx = scan(self.roots, scenarios=scen, shas=shas)

    def found(self, rid: str, es: int, sha: str) -> dict:
        return find(self.idx, self.all[rid], es, sha, self.dup)

    def tables(self, rid: str, ess, sha: str) -> tuple:
        """({es: episode_table}, [found records]) of OK stamps; a missing / bad one is in the records."""
        row = self.all[rid]
        radius = row.get("cross_r")
        out, recs = {}, []
        for es in ess:
            f = self.found(rid, es, sha)
            recs.append(f)
            if f["status"] == "OK":
                out[int(es)] = episode_table(Path(f["stamp"]), radius)
        return out, recs


def row_report(R: Rows, rid: str, ess, sha: str) -> dict:
    row = R.all[rid]
    t, recs = R.tables(rid, ess, sha)
    rep = {"row": rid, "env_seeds": list(ess), "stamps": recs,
           "status": "OK" if all(f["status"] == "OK" for f in recs) else "INCOMPLETE"}
    if t:
        rep["stats"] = summarize(t, row.get("kind"))
        rep["aggregate"] = {f"es{f['es']}": aggregate_of(Path(f["stamp"])) for f in recs if f["status"] == "OK"}
    if row.get("kind"):
        rep.update(kind=row["kind"], band=row["band"], cross_r=row["cross_r"], above_envelope=row["above_envelope"])
    return rep


def pair_tables(R: Rows, rid: str, ess, new_sha: str, ref_sha: str, key: str) -> dict:
    """The keyed per-episode maps of ``rid`` for both policies + the pairing / build checks."""
    ta, ra = R.tables(rid, ess, new_sha)
    tb, rb = R.tables(rid, ess, ref_sha)
    out = {"row": rid, "env_seeds": list(ess), "stamps": {"new": ra, "ref": rb}}
    bad = [f"{f['row']} es{f['es']} {str(f['sha256'])[:8]}: {f['status']} {f['problems'][:2]}"
           for f in ra + rb if f["status"] != "OK"]
    if bad:
        out.update(status="INCOMPLETE", why=bad)
        return out
    short = []
    if rid in R.kind:                    # a kind row is 100 episodes per env seed on BOTH sides (fail closed)
        for side, tabs in (("new", ta), ("ref", tb)):
            short += [f"{rid} es{es} {side}: {len(t['completion'])} episodes (want {KIND_ROW_EPISODES})"
                      for es, t in sorted(tabs.items()) if len(t["completion"]) != KIND_ROW_EPISODES]
    if short:
        out.update(status="INVALID", why=short)
        return out
    bp = build_problems(*ra, *rb)
    sp = []
    for fa, fb in zip(ra, rb):
        A, B = spawns_of(Path(fa["stamp"])), spawns_of(Path(fb["stamp"]))
        if set(A) != set(B):
            sp.append(f"es{fa['es']}: env_index sets differ")
            continue
        far = [i for i in A if np.linalg.norm(np.asarray(A[i]) - np.asarray(B[i])) > SPAWN_PAIR_TOL_M]
        if far:
            sp.append(f"es{fa['es']}: {len(far)} envs spawn apart (> {SPAWN_PAIR_TOL_M} m), e.g. {far[:5]}")
    if bp or sp:
        out.update(status="INVALID", why=bp + sp)
        return out
    out.update(status="OK", a=keyed(ta, key), b=keyed(tb, key), tables={"new": ta, "ref": tb})
    return out


def zero_kg_inputs(R: Rows, sha: str) -> tuple:
    """zero_kg_check's {suite, side22, field rows} for one policy from the retention rows, + problems."""
    out = {"suite": {}, "side22": None, "field_hard": [], "field_light": [], "ring_obst": [], "field_turn": []}
    probs, recs = [], []
    for es in (1, 2):
        f = R.found("R_suite", es, sha)
        recs.append(f)
        if f["status"] == "OK":
            out["suite"][es] = f["stamp"]
    f = R.found("R_side22", 1, sha)
    recs.append(f)
    if f["status"] == "OK":
        out["side22"] = f["stamp"]
    for fam, (rid, ess) in S.ZERO_KG_FIELD.items():
        for es in ess:
            f = R.found(rid, es, sha)
            recs.append(f)
            if f["status"] == "OK":
                out[fam].append(f["stamp"])
    probs = [f"{f['row']} es{f['es']}: {f['status']} {f['problems'][:2]}" for f in recs if f["status"] != "OK"]
    return out, probs, recs


def zero_kg(R: Rows, new_sha: str, ref_sha: str, its_past: float) -> dict:
    h, ph, rh = zero_kg_inputs(R, new_sha)
    r, prf, rr = zero_kg_inputs(R, ref_sha)
    if ph or prf:
        return {"status": "INCOMPLETE", "why": (ph + prf)[:12]}
    bp = build_problems(*rh, *rr)
    if bp:
        return {"status": "INVALID", "why": bp}
    res = gr().zero_kg_check(h, r, list(ZERO_KG_TRANSLATE), th_view(), its_past, resolution_files())
    res["status"] = "OK"
    return res


# ============================================================================== gates
def load_thresholds(path: Path) -> dict:
    doc = read_json(path)
    if doc is None:
        raise FileNotFoundError(f"{path}: no thresholds file (make_thresholds.py writes it at STOP 0)")
    if doc.get("schema") != "rt_thresholds_v1":
        raise ValueError(f"{path}: schema {doc.get('schema')!r}")
    return doc


def _frozen_rule(thr: dict, name: str, module_default: dict) -> dict:
    """The rule the thresholds file froze (``rule[name]`` minus its ``text``): the keys of the module's own constant,
    each read FROM THE FILE (a missing key refuses). ``gates`` applies this, never the module constant."""
    blk = (thr.get("rule") or {}).get(name)
    if not isinstance(blk, dict):
        raise ValueError(f"thresholds file has no rule.{name} block")
    missing = [k for k in module_default if k not in blk]
    if missing:
        raise ValueError(f"thresholds file rule.{name} lacks {missing}")
    return {k: blk[k] for k in module_default}


def gates(R: Rows, new_sha: str, tdr_sha: str, thr: dict) -> dict:
    dec0 = thr.get("decision")
    out = {"mode": "gates", "policies": {"new": new_sha, "tdr": tdr_sha}, "thresholds_decision": dec0,
           "G": {"kinds": {}}, "R": {"completion": {}, "solved": {}}, "report": {}}
    if dec0 != "proceed":
        out["verdict"] = {"verdict": "NO_GATE_READ", "action": f"thresholds decision {dec0!r}: steps 8-9 do not run "
                                                               "(PLAN decision rules)"}
        return out
    if thr.get("tdr_sha256") != tdr_sha:
        raise ValueError(f"thresholds were frozen on TDR {str(thr.get('tdr_sha256'))[:8]}, not {tdr_sha[:8]}")
    frozen = (thr.get("sources") or {}).get("sha256") or {}
    now = {n: sha256_file(HERE / n) for n in FROZEN_SCRIPTS}
    if frozen != now:
        raise ValueError("the thresholds file's recorded script sha256s differ from the current scripts "
                         f"(differ: {sorted(n for n in set(frozen) | set(now) if frozen.get(n) != now.get(n))}); the "
                         "rule was frozen with other code: re-run make_thresholds.py --check, or fix the scripts back")
    g_rule = _frozen_rule(thr, "G", G_RULE)
    r_cfg = _frozen_rule(thr, "R_completion", R_COMPLETION_CFG)
    ess = tuple(thr["rule"]["gate_env_seeds"])
    gst = []
    for kind, kd in thr["kinds"].items():
        if kd["status"] != "gated":
            continue
        rid = S.row_id(kind, kd["gate_band"])
        pt = pair_tables(R, rid, ess, new_sha, tdr_sha, "success")
        rec = {"row": rid, "band": kd["gate_band"], "status": pt["status"], "why": pt.get("why")}
        if pt["status"] == "OK":
            rec.update(g_line(pt["a"], pt["b"], rid, g_rule))
            rec["new_row"] = summarize(pt["tables"]["new"], kind)
            rec["tdr_row"] = summarize(pt["tables"]["ref"], kind)
            rec["aggregate"] = _aggregates(pt)
        out["G"]["kinds"][kind] = rec
        gst.append(pt["status"])
    lines = {k: v for k, v in out["G"]["kinds"].items() if v["status"] == "OK"}
    out["G"].update(g_gate(lines, int(thr["rule"]["k"])))
    if any(s != "OK" for s in gst):
        out["G"]["status"] = "INVALID" if "INVALID" in gst else "INCOMPLETE"
    rst = []
    for rid in S.COMPLETION_R:
        row = R.ret[rid]
        pt = pair_tables(R, rid, tuple(row["env_seeds"]), new_sha, tdr_sha, "completion")
        rec = {"status": pt["status"], "why": pt.get("why")}
        if pt["status"] == "OK":
            rec.update(retention_line(pt["a"], pt["b"], rid, cfg=r_cfg))
            if row.get("stressed_resolution") and not rec["pass"]:
                rec["stressed_cell_note"] = (f"shortfall {rec['ref'] - rec['head']:.3f} vs the cell's "
                                             f"{row['stressed_resolution']} stressed-cell resolution (PLAN item 7: "
                                             "read before the HALT)")
        out["R"]["completion"][rid] = rec
        rst.append(rec["status"] if rec["status"] != "OK" else ("PASS" if rec["pass"] else "FAIL"))
    for kind, kd in thr["kinds"].items():
        if kd["status"] != "solved":
            continue
        rid = S.row_id(kind, kd["r_band"])
        pt = pair_tables(R, rid, ess, new_sha, tdr_sha, "success")
        rec = {"row": rid, "band": kd["r_band"], "status": pt["status"], "why": pt.get("why")}
        if pt["status"] == "OK":
            rec.update(retention_line(pt["a"], pt["b"], rid, cfg=r_cfg))
            rec["new_row"] = summarize(pt["tables"]["new"], kind)
            rec["tdr_row"] = summarize(pt["tables"]["ref"], kind)
            rec["aggregate"] = _aggregates(pt)
        out["R"]["solved"][kind] = rec
        rst.append(rec["status"] if rec["status"] != "OK" else ("PASS" if rec["pass"] else "FAIL"))
    zk = zero_kg(R, new_sha, tdr_sha, ITS_PAST_R)
    if zk["status"] != "OK":
        out["R"]["tracking"] = zk
        rst.append(zk["status"])
    else:
        trk = r_tracking(zk)
        if trk["status"] != "PASS":
            zd = zero_kg(R, new_sha, tdr_sha, ITS_PAST_DRIFT_READ)
            trk["drift_reading"] = {"its_past_p2a": ITS_PAST_DRIFT_READ, "status": r_tracking(zd)["status"],
                                    "fails": r_tracking(zd)["fails"],
                                    "note": "PLAN item 7: an R line that fails is read against v1.2's drift term "
                                            "before the HALT (report; the decisive read is drift 0)"}
        out["R"]["tracking"] = trk
        rst.append(trk["status"])
    out["R"]["status"] = ("INVALID" if "INVALID" in rst else "INCOMPLETE" if "INCOMPLETE" in rst
                          else "FAIL" if "FAIL" in rst else "PASS")
    out["report"] = {
        "note": "report only (PLAN Gates): stuck share, r_max, termination reasons, slip, fall classes per row are "
                "in G.kinds / R.solved (new_row, tdr_row, aggregate); TDR's above-envelope step-6 rows below",
        "tdr_above_envelope_step6": {rid: r for rid, r in (thr.get("tdr_baseline_es3") or {}).items()
                                     if r.get("above_envelope")}}
    out["verdict"] = decide(out["G"]["status"], out["R"]["status"])
    return out


def _aggregates(pt: dict) -> dict:
    """The harness aggregates (fall classes, termination reasons, slip ratio median) of a pair's stamps."""
    return {side: {f"es{f['es']}": aggregate_of(Path(f["stamp"])) for f in recs if f.get("stamp")}
            for side, recs in pt["stamps"].items()}


def decide(g: str, r: str) -> dict:
    """PLAN decision rules (the step-7 rows are make_thresholds.py's)."""
    if g in ("INVALID", "INCOMPLETE"):
        return {"verdict": g, "action": "G unread: fix the rows first"}
    if g == "FAIL":
        return {"verdict": "NOT_ACCEPTED", "action": "G fails: not accepted; CPU diagnosis; HALT, the user decides"}
    if r in ("INVALID", "INCOMPLETE"):
        return {"verdict": r, "action": "G passes but R is unread: fix the rows first"}
    if r == "FAIL":
        return {"verdict": "HALT_R", "action": "G passes, R fails: HALT, the user decides (a retention remedy is a "
                                               "new run)"}
    return {"verdict": "ACCEPT", "action": "G and R pass: accept as the rough-terrain teacher; propose the follow-on "
                                           "phase (re-distil, stand re-pair, bake)"}


def baseline(R: Rows, tdr_sha: str, ess=(3,)) -> dict:
    """TDR's step-6 rows: every kind x band at ``ess`` (band choice) + the retention rows (report)."""
    out = {"mode": "baseline", "tdr_sha256": tdr_sha, "kind_env_seeds": list(ess), "kinds": {}, "retention": {}}
    for rid, row in R.kind.items():
        out["kinds"].setdefault(row["kind"], {})[row["band"]] = row_report(R, rid, ess, tdr_sha)
    for rid, row in R.ret.items():
        out["retention"][rid] = row_report(R, rid, tuple(row["env_seeds"]), tdr_sha)
    return out


# ============================================================================== validations
def deep_diff(rec, got, path="", tol=EPS, only_recorded=True) -> list:
    """Every difference between a recorded JSON value and a recomputed one (numbers to ``tol``); a key only in
    ``got`` is ignored when ``only_recorded`` (the record is the reference)."""
    out = []
    if isinstance(rec, dict) and isinstance(got, dict):
        for k in sorted(set(rec) | set(got), key=str):
            if k not in got:
                out.append(f"{path}/{k}: recorded only")
            elif k not in rec:
                if not only_recorded:
                    out.append(f"{path}/{k}: recomputed only")
            else:
                out += deep_diff(rec[k], got[k], f"{path}/{k}", tol, only_recorded)
    elif isinstance(rec, list) and isinstance(got, list):
        if len(rec) != len(got):
            out.append(f"{path}: length {len(rec)} != {len(got)}")
        for i, (a, b) in enumerate(zip(rec, got)):
            out += deep_diff(a, b, f"{path}[{i}]", tol, only_recorded)
    elif isinstance(rec, bool) or isinstance(got, bool) or not (isinstance(rec, (int, float))
                                                                 and isinstance(got, (int, float))):
        if rec != got or isinstance(rec, bool) != isinstance(got, bool):
            out.append(f"{path}: {str(rec)[:80]} != {str(got)[:80]}")
    elif not (math.isclose(rec, got, rel_tol=0.0, abs_tol=tol) or (math.isnan(rec) and math.isnan(got))):
        out.append(f"{path}: {rec!r} != {got!r} (|d| {abs(rec - got):.3g})")
    return out


def _count_numbers(x) -> int:
    if isinstance(x, bool):
        return 0
    if isinstance(x, (int, float)):
        return 1
    if isinstance(x, dict):
        return sum(_count_numbers(v) for v in x.values())
    if isinstance(x, list):
        return sum(_count_numbers(v) for v in x)
    return 0


def _norm(x):
    return json.loads(json.dumps(x, default=_jsonable))


#: the rod-speed G1 cells (rsd_lib S_CELLS + N_CELLS): field_light env file + payload level (rsd_lib.cell)
G1_CELLS = {"T01_rod_speed_0p7_L090": "L090", "T01_rod_speed_0p8_L090": "L090", "T01_rod_speed_0p7_L000": None,
            "T01_rod_speed_1p35_L000": None, "T01_rod_speed_1p35_L090": "L090", "T00_nominal_L000_A": None,
            "T00_nominal_L090_A": "L090", "T00_nominal_L135_A": "L135"}
G1_REF_CELLS = ("T00_nominal_L000_A", "T00_nominal_L090_A", "T00_nominal_L135_A", "T01_rod_speed_0p7_L090",
                "T01_rod_speed_0p8_L090")
G1_TCTL_CELLS = ("T00_nominal_L090_A", "T01_rod_speed_0p7_L090", "T01_rod_speed_0p8_L090")


def g1_cell_row(cell: str) -> dict:
    lv = G1_CELLS[cell]
    return {"id": f"G1_{cell}", "manifest": "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml",
            "scenario": "field_light__omni", "env_file": f"experiments/2026-11-07_rod_speed_dr/envs/{cell}__field_light.env",
            "level_file": None if lv is None else f"experiments/2026-10-16_task4_pipeline/F19/envs/levels/{lv}.env",
            "args": list(S.TEACHER_ARGS)}


def validate_g1(roots=None, g1_path: Path | None = None, dup: str = "earliest") -> dict:
    """(e1) Recompute TDR's recorded rod-speed G1 numbers from its saved stamps, keyed by checkpoint sha256: every
    cell-seed's field_light completion (S + N, and the WATCH copies), the pooled paired completions (N_pooled TDR vs
    T*, x0.7 TDR vs T*, A = TDR vs TCTL x0.7, TCTL vs T* x0.7) through this read-out's retention_line, and the R
    read (gate_read.zero_kg_check at the recorded its_past_p2a 2999) through this read-out's zero_kg inputs. PASS =
    no difference above 1e-12 in any recorded number."""
    g1 = read_json(g1_path or RSD_CAMP / "results" / "g1.json")
    pol = g1["policies"]
    T, O, C = pol["tdr"]["sha256"], pol["old"]["sha256"], (pol.get("tctl") or {}).get("sha256")
    if T != TDR_SHA:
        raise ValueError(f"g1.json tdr sha {T[:8]} is not TDR {TDR_SHA[:8]}")
    if C != TCTL_SHA:
        raise ValueError(f"g1.json tctl sha {str(C)[:8]} is not TCTL {TCTL_SHA[:8]}")
    cells = {c: g1_cell_row(c) for c in G1_CELLS}
    doc = S.load_rows()
    doc = {**doc, "retention_rows": doc["retention_rows"] + [{**r, "cross_r": None} for r in cells.values()]}
    R = Rows(roots or [RSD_LR], doc, dup, shas={T, O, C})
    diffs, compared, got = [], 0, {"completion": {}, "pooled": {}}

    def comp_map(cell, sha, es):
        f = R.found(cells[cell]["id"], es, sha)
        if f["status"] != "OK":
            raise KeySetError(f"G1 {cell} es{es} {sha[:8]}: {f['status']} {f['problems'][:2]} {f['notes'][:2]}")
        return f, completion_of(Path(f["stamp"]))

    stamps_read = []
    for grp in ("S", "N"):
        for cell, rec in g1[grp].items():
            for es_s, sd in rec["seeds"].items():
                f, cm = comp_map(cell, T, int(es_s))
                stamps_read.append(f["stamp"])
                v = share(cm)
                got["completion"][f"{grp}.{cell}.es{es_s}"] = v
                diffs += deep_diff(sd["values"]["field_light_completion"], v, f"/{grp}/{cell}/es{es_s}/completion")
                compared += 1
                w = g1["watch"]["tdr"][cell][f"es{es_s}"]["completion"]
                diffs += deep_diff(w, v, f"/watch/tdr/{cell}/es{es_s}/completion")
                compared += 1
    for who, sha in (("tdr", T), ("old", O)):
        for es_s, v in g1["watch"]["L135_completion"][who].items():
            _, cm = comp_map("T00_nominal_L135_A", sha, int(es_s[2:]))
            diffs += deep_diff(v, share(cm), f"/watch/L135_completion/{who}/{es_s}")
            compared += 1

    def pooled(cell, sa, sb):
        A, B = {}, {}
        for es in (1, 2):
            fa, ca = comp_map(cell, sa, es)
            fb, cb = comp_map(cell, sb, es)
            stamps_read.extend([fa["stamp"], fb["stamp"]])
            A.update({(es, i): v for i, v in ca.items()})
            B.update({(es, i): v for i, v in cb.items()})
        return retention_line(A, B, "field_light")

    pairs = {**{f"N_pooled/{c}": (c, T, O) for c in g1["N_pooled"]},
             "x07_tdr_vs_old": ("T01_rod_speed_0p7_L090", T, O), "A": ("T01_rod_speed_0p7_L090", T, C),
             "tctl_vs_old_x07": ("T01_rod_speed_0p7_L090", C, O)}
    for key, (cell, sa, sb) in pairs.items():
        rec = g1["N_pooled"][key.split("/", 1)[1]] if key.startswith("N_pooled/") else g1[key]
        line = pooled(cell, sa, sb)
        got["pooled"][key] = line
        diffs += deep_diff(rec, _norm(line), f"/{key}")
        compared += _count_numbers(rec)
    zk = zero_kg(R, T, O, g1["its_past_p2a"])
    if zk.get("status") != "OK":
        raise KeySetError(f"G1 R inputs: {zk.get('why')}")
    zk.pop("status", None)
    rec_R = {k: v for k, v in g1["R"].items() if k != "inputs"}
    diffs += deep_diff(rec_R, _norm(zk), "/R")
    compared += _count_numbers(rec_R)
    trk0 = r_tracking(gr().zero_kg_check(*[zero_kg_inputs(R, s)[0] for s in (T, O)], list(ZERO_KG_TRANSLATE),
                                         th_view(), ITS_PAST_R, resolution_files()))
    return {"validation": "e1_g1_reproduction", "g1": str(g1_path or RSD_CAMP / "results" / "g1.json"),
            "keyed_by_sha256": {"tdr": T, "old": O, "tctl": C}, "numbers_compared": compared,
            "differences": diffs, "pass": not diffs, "tolerance": EPS, "stamps_read": sorted(set(stamps_read)),
            "recomputed": got, "R_verdict": zk.get("verdict"),
            "R_at_its_past_0_tracking_status": trk0["status"],
            "R_at_its_past_0_note": "TDR vs T* at drift 0 (the campaign's R form; report only here)"}


SITE_PROBE = r"""
import sys, json, importlib.util
sys.dont_write_bytecode = True
p = sys.argv[1]
sys.path.insert(0, p.rsplit('/', 1)[0])
spec = importlib.util.spec_from_file_location('site_teacher_readout_probe', p)
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
out = {}
for st in json.loads(sys.stdin.read()):
    c = m.crossed_of({'stamp': st}); s = m.success_of({'stamp': st, 'label': st})
    out[st] = {'crossed': {str(k): bool(v) for k, v in c.items()}, 'success': {str(k): bool(v) for k, v in s.items()}}
print(json.dumps(out))
"""


def site_layers(st: Path) -> dict:
    with np.load(Path(st) / "site_rec" / "layers.npz", allow_pickle=False) as z:
        return {"origins": np.asarray(z["terrain_origins"], np.float64).reshape(-1, 3)[:, :2],
                "cross_r": np.asarray(z["tile_cross_r"], np.float64).reshape(-1)}


def crossed_tile_frame(st: Path, tol: float = SITE_SPAWN_TOL_M) -> tuple:
    """SITE's frame for THIS read-out's rule: the origin is the tile centre nearest the spawn (within ``tol``), the
    radius that tile's ``tile_cross_r``; r_max = r_max_from(track, origin). Returns ({env: crossed}, {env: spawn
    offset from the tile centre}, {env: crossed with the SPAWN as origin and the same radius}). The spawn-frame map
    is THIS campaign's own path (``r_max_of`` + ``crossed_from``, raw length == n_steps asserted), at the tile's radius."""
    lay = site_layers(st)
    rmax_spawn = r_max_of(st)
    cro, off, cro_spawn = {}, {}, {}
    for e in episode_list(st):
        i = int(e["env_index"])
        ep = raw_episode(st, i)
        k = int(np.argmin(np.linalg.norm(lay["origins"] - ep["xy"][0], axis=1)))
        d0 = float(np.linalg.norm(lay["origins"][k] - ep["xy"][0]))
        if d0 > tol:
            raise KeySetError(f"{st}: env {i} spawn is {d0:.2f} m from its tile")
        r = lay["cross_r"][k]
        if not np.isfinite(r):
            raise KeySetError(f"{st}: env {i} tile {k} has no crossing radius")
        cro[i] = bool(r_max_from(ep["xy"], lay["origins"][k]) >= r - EPS)
        cro_spawn[i] = crossed_from({i: rmax_spawn[i]}, r)[i]
        off[i] = d0
    return cro, off, cro_spawn


def validate_site(lr: Path = SITE_LR, who: str = "SITE9_tstar", site_camp: Path = SITE_CAMP) -> dict:
    """(e2) On SITE's saved T* site rows (``<lr>/SITE9_tstar/site_*``, every row with site_rec/layers.npz and finite
    tile radii), this read-out's crossed rule at SITE's radii and frame reproduces SITE's own read-out
    (readout.crossed_of / success_of, run read-only in a child process) episode for episode, and its pooled
    completed_and_crossed equals SITE's recorded T* values (results/step9 lowband_new.json ``b``, gates_es12.json
    gate b ``per_row.b``). The rule is not loosened: >= radius - 1e-12 on float64 of the raw float32 track."""
    rows = []
    for sd in sorted((Path(lr) / who).glob("site_*")):
        if sd.name.startswith("site_blockers"):
            continue
        for ed in sorted(sd.glob("seed*")):
            sts = sorted(p for p in ed.iterdir() if p.is_dir() and (p / "site_rec" / "layers.npz").is_file()
                         and _complete(p))
            if sts:
                rows.append((sd.name, int(ed.name[4:]), sts[-1]))
    mine, offs, spawn_agree, per_row_pool = {}, [], [0, 0], collections.defaultdict(dict)
    skipped, wrong_sha = [], []
    for sc, es, st in rows:
        sha = (read_json(st / "run_meta.json", {}) or {}).get("checkpoint_sha256")
        if sha != TSTAR_SHA:                                  # SITE9_tstar is T*'s: the sha, not the dir name, decides
            wrong_sha.append(f"{sc} es{es} {st.name}: checkpoint_sha256 {str(sha)[:8]} != T* {TSTAR_SHA[:8]}")
            continue
        try:
            cro, off, cs = crossed_tile_frame(st)
        except KeySetError as e:
            skipped.append(f"{sc} es{es}: {e}")
            continue
        comp = completion_of(st)
        suc = success_from(comp, cro)
        mine[str(st)] = {"scenario": sc, "es": es, "crossed": cro, "success": suc}
        offs += list(off.values())
        spawn_agree[0] += sum(cs[i] == cro[i] for i in cro)
        spawn_agree[1] += len(cro)
        per_row_pool[sc].update({(es, i): v for i, v in suc.items()})
    py = sys.executable
    proc = subprocess.run([py, "-I", "-c", SITE_PROBE, str(site_camp / "scripts" / "readout.py")],
                          input=json.dumps(sorted(mine)), capture_output=True, text=True,
                          env={**{k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "PYTHON"))},
                               "CUDA_VISIBLE_DEVICES": ""}, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"SITE read-out probe failed: {proc.stderr[-2000:]}")
    theirs = json.loads(proc.stdout.strip().splitlines()[-1])
    ep_diff, n_eps = [], 0
    for st, m in mine.items():
        t = theirs[st]
        for key in ("crossed", "success"):
            a = {int(k): v for k, v in t[key].items()}
            if set(a) != set(m[key]):
                ep_diff.append(f"{m['scenario']} es{m['es']} {key}: env sets differ")
                continue
            bad = [i for i in a if a[i] != m[key][i]]
            n_eps += len(a) if key == "success" else 0
            if bad:
                ep_diff.append(f"{m['scenario']} es{m['es']} {key}: {len(bad)} episodes differ {bad[:6]}")
    rec_diff, rec_n = [], 0
    lb = read_json(site_camp / "results" / "step9" / "lowband_new.json") or {}
    for sc, v in (lb.get("scenarios") or {}).items():
        rv = ((v.get("completed_and_crossed") or {}).get("b"))
        if rv is None or sc not in per_row_pool:
            continue
        rec_diff += deep_diff(rv, share(per_row_pool[sc]), f"/lowband_new/{sc}/completed_and_crossed/b")
        rec_n += 1
    gb = ((read_json(site_camp / "results" / "step9" / "gates_es12.json") or {}).get("b") or {}).get("per_row") or {}
    for sc, v in gb.items():
        if sc in per_row_pool and v.get("b") is not None:
            rec_diff += deep_diff(v["b"], share(per_row_pool[sc]), f"/gates_es12/b/per_row/{sc}/b")
            rec_n += 1
    return {"validation": "e2_site_crossed_reproduction", "rows_read": len(mine), "episodes_compared": n_eps,
            "episode_differences": ep_diff, "recorded_values_compared": rec_n, "recorded_differences": rec_diff,
            "skipped": skipped, "skipped_wrong_sha256": wrong_sha,
            "pass": bool(mine) and not ep_diff and not rec_diff and rec_n > 0
            and spawn_agree[0] == spawn_agree[1] and spawn_agree[1] > 0,
            "frame": "tile centre nearest the spawn (<= 0.5 m), the tile's tile_cross_r",
            "spawn_offset_from_tile_centre_m": distribution(offs),
            "spawn_frame_agreement": {"agree": spawn_agree[0], "n": spawn_agree[1]},
            "pooled_success_by_scenario": {sc: {"success": share(v), "n": len(v)}
                                           for sc, v in sorted(per_row_pool.items())}}


#: (e3) the band-free TDR rows: field_hard (R_field_hard), field_light (R_field_light), x0.7 L090 field_light
CALIBRATION_ROWS = ("R_field_hard", "R_field_light", "R_x07_L090_field_light")
CALIBRATION_RADII = (3.0, 3.2)
CALIBRATION_LINE = 0.95


def calibrate(roots=None, sha: str = TDR_SHA, ess=(1, 2), dup: str = "earliest") -> dict:
    """(e3) PLAN step 4 radius calibration: on TDR's saved band-free rows (es1-2), the share of COMPLETERS whose xy
    r_max from the spawn reaches 3.0 m and 3.2 m must be >= 0.95 on every row (else that radius is lowered before
    step 6)."""
    R = Rows(roots or [RSD_LR], dup=dup, shas={sha})
    out = {"validation": "e3_radius_calibration", "sha256": sha, "env_seeds": list(ess), "line": CALIBRATION_LINE,
           "rows": {}}
    ok = True
    for rid in CALIBRATION_ROWS:
        rm_all, comp_all, recs = {}, {}, []
        for es in ess:
            f = R.found(rid, es, sha)
            recs.append({k: f[k] for k in ("es", "status", "stamp", "problems", "notes")})
            if f["status"] != "OK":
                ok = False
                continue
            st = Path(f["stamp"])
            rm, cm = r_max_of(st), completion_of(st)
            rm_all.update({(es, i): v for i, v in rm.items()})
            comp_all.update({(es, i): v for i, v in cm.items()})
        comp_r = [v for k, v in rm_all.items() if comp_all[k]]
        res = {"stamps": recs, "n": len(rm_all), "completers": len(comp_r), "completion": share(comp_all),
               "r_max_completers": distribution(comp_r), "r_max_all": distribution(rm_all.values()), "shares": {}}
        for rad in CALIBRATION_RADII:
            sh = float(np.mean([v >= rad - EPS for v in comp_r])) if comp_r else None
            per = {}
            for es in ess:
                c = [v for k, v in rm_all.items() if k[0] == es and comp_all[k]]
                per[f"es{es}"] = float(np.mean([v >= rad - EPS for v in c])) if c else None
            res["shares"][f"{rad:g}"] = {"pooled": sh, "per_es": per,
                                         "pass": sh is not None and sh >= CALIBRATION_LINE - EPS}
            ok = ok and res["shares"][f"{rad:g}"]["pass"]
        out["rows"][rid] = res
    out["pass"] = ok
    out["min_share"] = {f"{rad:g}": min((r["shares"][f"{rad:g}"]["pooled"] for r in out["rows"].values()
                                        if r["shares"][f"{rad:g}"]["pooled"] is not None), default=None)
                        for rad in CALIBRATION_RADII}
    return out


# ============================================================================== CLI
def _guard(fn, *a, **kw) -> dict:
    try:
        return fn(*a, **kw)
    except KeySetError as e:
        return {"status": "INVALID", "pass": False, "why": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"status": "ERROR", "pass": False, "why": f"{type(e).__name__}: {e}", "error": traceback.format_exc()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("rows", "baseline", "gates", "validate-g1", "validate-site", "calibrate",
                                     "validate"))
    ap.add_argument("--root", action="append", default=None, help="an lr dir (repeatable); default the campaign's")
    ap.add_argument("--sha", default=None)
    ap.add_argument("--new", default=None)
    ap.add_argument("--tdr", default=TDR_SHA)
    ap.add_argument("--es", type=int, action="append", default=None)
    ap.add_argument("--row", action="append", default=None)
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    ap.add_argument("--g1", type=Path, default=None)
    ap.add_argument("--dup", choices=("earliest", "latest", "error"), default="earliest")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    roots = [Path(r) for r in a.root] if a.root else None
    rc = 0
    if a.mode == "rows":
        if not a.sha:
            ap.error("rows needs --sha")
        R = Rows(roots or [RT_LR], dup=a.dup, shas={a.sha})
        ids = a.row or list(R.all)
        res = {"mode": "rows", "sha256": a.sha,
               "rows": {rid: row_report(R, rid, a.es or (R.all[rid]["env_seeds"]["gate"]
                                                         if isinstance(R.all[rid]["env_seeds"], dict)
                                                         else R.all[rid]["env_seeds"]), a.sha) for rid in ids}}
    elif a.mode == "baseline":
        R = Rows(roots or [RT_LR], dup=a.dup, shas={a.tdr})
        res = baseline(R, a.tdr, tuple(a.es or (3,)))
    elif a.mode == "gates":
        if not a.new:
            ap.error("gates needs --new")
        R = Rows(roots or [RT_LR], dup=a.dup, shas={a.new, a.tdr})
        thr = load_thresholds(a.thresholds)
        res = gates(R, a.new, a.tdr, thr)
        res["thresholds"] = {"path": str(a.thresholds), "sha256": sha256_file(a.thresholds)}
        print(f"RT GATES verdict={res['verdict']['verdict']} G={res['G'].get('status')} R={res['R'].get('status')}")
    else:
        res = {"mode": a.mode, "readout_sha256": sha256_file(Path(__file__)), "rt_spec_sha256": sha256_file(HERE / "rt_spec.py")}
        if a.mode in ("validate-g1", "validate"):
            res["e1"] = _guard(validate_g1, roots, a.g1, a.dup)
        if a.mode in ("validate-site", "validate"):
            res["e2"] = _guard(validate_site)
        if a.mode in ("calibrate", "validate"):
            res["e3"] = _guard(calibrate, roots, TDR_SHA, (1, 2), a.dup)
        for k in ("e1", "e2", "e3"):
            if k in res:
                v = res[k]
                print(f"RT VALIDATE {k} pass={v.get('pass')} " + {
                    "e1": f"compared={v.get('numbers_compared')} diffs={len(v.get('differences') or [])}",
                    "e2": f"rows={v.get('rows_read')} episodes={v.get('episodes_compared')} "
                          f"spawn_frame={(v.get('spawn_frame_agreement') or {}).get('agree')}/"
                          f"{(v.get('spawn_frame_agreement') or {}).get('n')} "
                          f"wrong_sha_skipped={len(v.get('skipped_wrong_sha256') or [])} "
                          f"ep_diffs={len(v.get('episode_differences') or [])} recorded={v.get('recorded_values_compared')} "
                          f"rec_diffs={len(v.get('recorded_differences') or [])}",
                    "e3": f"min_share={v.get('min_share')}"}[k] + (f" why={v.get('why')}" if v.get("why") else ""))
                rc = rc or (0 if v.get("pass") else 1)
    if a.out:
        write_json(a.out, res)
        print(f"RT wrote {a.out}")
    elif a.mode in ("rows", "baseline"):
        print(json.dumps(res, indent=1, default=_jsonable)[:20000])
    return rc


if __name__ == "__main__":
    sys.exit(main())

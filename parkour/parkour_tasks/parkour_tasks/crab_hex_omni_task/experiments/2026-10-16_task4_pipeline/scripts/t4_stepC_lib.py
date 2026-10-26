#!/usr/bin/env python3
"""PLAN step 8 (Phase C preflight): the shared, Kit-free pieces of the step-8 job table, queue and reader.

Imported by ``t4_stepC_jobs.py`` (the job table the queue runs) and ``t4_stepC_read.py`` (the checks and the
report). Nothing here boots Kit, trains or evaluates; it reads stamps / tfevents / driver records and REUSES the TP
tools by path (``TP/pipeline.py`` for the row composition, ``gate_read`` / ``payload_read`` / ``live_watch`` /
``replay_read`` / ``thresholds`` for every statistic), so the step-8 reads are the pipeline's own reads.

Paths (every one absolute):
  PL        the campaign dir             SC   PL/preflight/stepC (the step-8 work dir, also a driver campaign dir)
  OUT       parkour/outputs/omni_plan/task4/stepC (row stamps under OUT/lr/<NAME>/..., distill / stand logs)
The pinned rules these functions implement are PL/preflight/STEPC_RULES.md.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

R = Path(__file__).resolve().parents[7]  # this checkout's root (scripts -> campaign -> experiments -> ... -> repo)
OMNI = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
FWD = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task"
EXP = OMNI / "experiments"
PL = EXP / "2026-10-16_task4_pipeline"
TP = EXP / "tools/pipeline"
SC = PL / "preflight/stepC"
OUT = R / "parkour/outputs/omni_plan/task4/stepC"
P2LR = R / "parkour/outputs/omni_plan/phase2/lr"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
SPEC = PL / "pipeline.yaml"
THRESHOLDS = PL / "thresholds_v0.json"
STATE = SC / "stepC_state.json"
LOGS = R / "parkour/logs/rsl_rl"
STUDENT_EXP = "crab_hex_omni_student"

#: S1 s18 = the p2a head of record (the run-dir copy: it carries params/agent.yaml; byte-identical to the bundle)
S1_CK = LOGS / "crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
S1_SHA = "968cc550989919cdc3fa36ae047379febabee5659cde0ed4d099195acb13c23c"
S1_ITER = 19994
S1_BUNDLED = OMNI / "policy/6_p2a_field/model_19994.pt"
S1_RUN = S1_CK.parent
X10_CK = LOGS / "crab_hex_omni_teacher/2026-10-07_15-34-26_seed16_X10_p2a_A15pB/model_19994.pt"
X11_CK = LOGS / "crab_hex_omni_teacher/2026-10-07_19-05-32_seed17_X11_p2a_A15pB/model_19994.pt"
X60_C6X_CK = LOGS / "crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB/model_14995.pt"
X43_RUN = LOGS / "crab_hex_omni_flat_walk/2026-10-04_10-01-39_seed18_X43_s1_A15pB"
SH_T17_CK = R / "parkour/outputs/omni_plan/stand_height/runs/SH_T17/model_5000.pt"
SH_T17_SHA = "99c377a17b1bf04774afb17871db11c9a9909c14b84951832fb6a0087484154e"
FWD_3A_CK = LOGS / "crab_hex_student/2026-09-08_05-54-01/model_24995.pt"
FWD_3A_RUN = FWD_3A_CK.parent
FWD_2C_RUN = LOGS / "crab_hex_flat_walk/2026-09-07_04-38-50"
FWD_3A_SHAPE_USD = R / "assets/variants/crab_simple__splay15_axis2p5in.usda"
#: the forward phase driver's canary env (FWD/policy/README.md "flat canary of the current head")
FWD_3A_ENV = {"KRABBY_LIN_VEL_X": "0.0:0.35", "KRABBY_TRACK_SIGMA2": "0.1", "KRABBY_TRACK_L1_W": "-1.0",
              "KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "1.0", "KRABBY_STUDENT_MDP": "1"}

_PO = R / "parkour/outputs/omni_plan"
#: recorded stamps the bridges compare against (each verified to exist when this file was written, 2026-10-16)
REF = {
    "REFS_suite_es1": _PO / "phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24",
    "REFS_suite_es2": _PO / "phase2/lr/REFS_S1_s18/suite__omni/seed002/2026-10-09_03-27-23",
    "REFS_walkstop_es1": _PO / "phase2/lr/REFS_S1_s18/walkstop__omni/seed001/2026-10-09_04-11-22",
    "REFS_field_suite_es1": _PO / "phase2/lr/REFS_S1_s18/field_suite__omni/seed001/2026-10-09_04-04-16",
    "REF_field_hard_es1": _PO / "phase2/lr/REF_S1_s18/field_hard__omni/seed001/2026-10-08_05-39-53",
    "REF_field_light_es1": _PO / "phase2/lr/REF_S1_s18/field_light__omni/seed001/2026-10-08_05-26-01",
    "REF_ring_obst_es1": _PO / "phase2/lr/REF_S1_s18/ring_obst__omni/seed001/2026-10-08_05-33-02",
    "REF_field_turn_es1": _PO / "phase2/lr/REF_S1_s18/field_turn__omni/seed001/2026-10-08_05-54-46",
    "P2_side22_es1": _PO / "phase2/lr/P2__S1_s18/side22_25lr__omni/seed001/2026-10-07_14-02-00",
    "ANC_suite_body_es1": _PO / "body_cmd/lr/ANC_S1_s18/suite__omni_body/seed001/2026-10-15_23-56-56",
    "ANC_turnlevels_body_es1": _PO / "body_cmd/lr/ANC_S1_s18/turnlevels__omni_body/seed001/2026-10-16_00-04-01",
    "ANC_grid_body_es1": _PO / "body_cmd/lr/ANC_S1_s18/grid__omni_body/seed001/2026-10-16_00-11-05",
    "TB_c6x_es1": _PO / "ppo_construction/lr/SF__B_rest_s18_tb/turn_band__omni/seed001/2026-10-06_08-41-42",
    "X43_suite_es1": LOGS / "gait_eval/omni_phases/2026-10-01_ppo_construction/seed18_X43_s1_A15pB/suite__omni/seed001/"
                            "2026-10-04_16-15-43",
    "X60_suite_es1": _PO / "phase2/lr/ANC_s18/suite__omni/seed001/2026-10-06_22-48-11",
    "X10_suite_es1": LOGS / "gait_eval/omni_phases/2026-10-07_omni_phase2/seed16_X10_p2a_A15pB/suite__omni/seed001/"
                            "2026-10-07_21-40-30",
    "SH_T17_hwalkstop_es1": _PO / "phase2/lr/SH_T17/hwalkstop__omni/seed001/2026-10-15_00-12-36",
    "SH_T17_hraise_es1": _PO / "phase2/lr/SH_T17/hraise__omni/seed001/2026-10-15_00-26-16",
    "SH_T17z_walkstop_es1": _PO / "phase2/lr/SH_T17z/walkstop__omni/seed001/2026-10-15_00-49-40",
    "FWD3A_slow_es1": LOGS / "gait_eval/phases/2026-09-07_1330_phase_pipeline/seed3_3a_003_A15pB/slow__A15pB/seed001/"
                             "2026-09-08_19-46-08",
}
#: Phase-A (A2) smoke run dirs, from preflight/orchestrator.log (t4_stepA_read.smoke_runs)
A2_LOG = PL / "preflight/orchestrator.log"
LINEAGE_RUN = {
    "s1": X43_RUN,
    "c4": LOGS / "crab_hex_omni_flat_walk/2026-10-05_17-41-16_seed18_X57_c4_A15pB",
    "c5": LOGS / "crab_hex_omni_flat_walk/2026-10-06_00-56-48_seed18_X60_c5_A15pB",
    "c6": LOGS / "crab_hex_omni_flat_walk/2026-10-06_01-30-49_seed18_X60_c6_A15pB",
    "c6x": LOGS / "crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB",
    "p2a": LOGS / "crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB",
}
STAGES6 = ("s1", "c4", "c5", "c6", "c6x", "p2a")
LEVELS = (0, 45, 90, 135, 180, 225, 270)
LEVEL_ROWS = ("suite@1", "suite@2", "field_light@1", "field_light@2")
ZERO_KG_ROWS = ("suite@1", "suite@2", "side22@1", "field_hard@1", "field_hard@2", "field_light@1", "field_light@2",
                "ring_obst@1", "ring_obst@2", "field_turn@1", "field_turn@2")
PILOT_ITERS = 3000
PILOT_OFFSETS = (500, 1000, 1500, 2000, 2500, 3000)
NP_CLAMP = (1500, 3000)
TRANSLATE = ("fwd", "side", "back", "diag")

# ------------------------------------------------------------------------------------------------ loaders
_MODS: dict = {}


def load(path: Path, name: str):
    if name in _MODS:
        return _MODS[name]
    if name in sys.modules:
        _MODS[name] = sys.modules[name]
        return _MODS[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def P():
    """TP/pipeline.py (its Spec / Ctx / row_job / parse_row) with tp_lib configured from the spec's ``paths`` block
    (side22_classify, sh_kin: gate_read / payload_read need it before any Spec is built)."""
    first = "tp_pipeline" not in _MODS and "tp_pipeline" not in sys.modules
    mod = load(TP / "pipeline.py", "tp_pipeline")
    if first or not mod.L.PATHS:
        mod.L.configure_from(SPEC)
    return mod


def TL():
    return P().L


def GR():
    return P().GR


def PR():
    return P().PR()


def LW():
    return P().LW()


def RR():
    return P().RR()


def TH():
    return TL().thresholds_mod()


def BCB():
    return load(EXP / "2026-10-15_body_cmd/scripts/bc_bitwise.py", "t4c_bc_bitwise")


_CTX: dict = {}


def ctx(run: str = "R18"):
    """A read-only pipeline context for ``run`` (R18 = the official seed-18 run's generated env files). Nothing is
    saved: row_job / expect_for are pure on the plan and the spec."""
    if run not in _CTX:
        p = P()
        _CTX[run] = p.Ctx(p.Spec(SPEC), run)
    return _CTX[run]


def th_doc() -> dict:
    return json.loads(THRESHOLDS.read_text())


def thresholds():
    return GR().Thresholds.load(THRESHOLDS)


# ------------------------------------------------------------------------------------------------ io
def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)


_SHA: dict = {}


def sha256(p) -> str:
    p = Path(p)
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def env_text(env: dict) -> str:
    return "".join(f"{k}={env[k]}\n" for k in sorted(env))


def stamp_env(stamp: Path) -> dict:
    return {str(k): str(v) for k, v in (read_json(Path(stamp) / "run_meta.json", {}).get("env_vars") or {}).items()}


def iter_of(p) -> int | None:
    m = re.search(r"model_(\d+)\.pt$", str(p))
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------ stamps
def newest_stamp(root: Path, scenario: str | None, es: int) -> Path | None:
    """The newest stamp dir ``root/<scenario or *>/seed00k/<time>/`` that has a run_meta.json."""
    pat = str(Path(root) / (scenario or "*") / f"seed{int(es):03d}" / "*")
    hits = [Path(p) for p in glob.glob(pat) if (Path(p) / "run_meta.json").is_file()]
    return max(hits, key=lambda p: p.name) if hits else None


def row_stamp(name: str, scenario: str, es: int, ckpt=None) -> Path | None:
    """A COMPLETE gate_row.sh stamp of row NAME (tp_row.json = the runner's completion marker, written last), of the
    checkpoint ``ckpt`` when given."""
    d = newest_stamp(OUT / "lr" / name, scenario, es)
    if d is None or not (d / "tp_row.json").is_file():
        return None
    if ckpt is not None:
        meta = read_json(d / "run_meta.json", {})
        if meta.get("checkpoint_sha256") != sha256(ckpt):
            return None
    return d


def adhoc_stamp(name: str, es: int = 1) -> Path | None:
    """A complete t4_eval_row.sh ad-hoc stamp (OUT/lr/NAME/<any>/seed00k/<time>/ + code_state.json)."""
    d = newest_stamp(OUT / "lr" / name, None, es)
    if d is None:
        return None
    ok = all((d / f).is_file() for f in ("run_meta.json", "scenario_metrics.json", "summary.md", "code_state.json"))
    return d if ok else None


def p2_stamp(name: str, scenario: str, es: int, ckpt_sha: str | None = None) -> Path | None:
    """A complete run_arm.sh --row-eval stamp (phase-2 lr root): complete.json or p2_complete.json."""
    d = newest_stamp(P2LR / name, scenario, es)
    if d is None or not ((d / "complete.json").is_file() or (d / "p2_complete.json").is_file()):
        return None
    if ckpt_sha and read_json(d / "run_meta.json", {}).get("checkpoint_sha256") != ckpt_sha:
        return None
    return d


# ------------------------------------------------------------------------------------------------ comparisons
META_KEYS = ("trained_categories",)


def _strip(obj, keys=META_KEYS):
    if isinstance(obj, dict):
        return {k: _strip(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip(v, keys) for v in obj]
    return obj


def _ref_keys(a, b):
    """``b`` restricted to the keys ``a`` has (recursively, dicts only; lists element-wise)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _ref_keys(a[k], b[k]) for k in a if k in b}
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_ref_keys(x, y) for x, y in zip(a, b)]
    return b


#: raw npz keys the harness writes today that the 2026-09-08 forward 3a stamp predates (C1.7, shared-record compare)
FWD_NEW_RAW_KEYS = ("clock_cmd_speed", "clock_running", "rod_speed")


def bitwise(ref: Path, new: Path, *, raw: bool = True, allow_new_keys=(), ref_keys_only: bool = False) -> dict:
    """The step-8 bridge compare (STEPC_RULES §0): every raw npz key bitwise (``bc_bitwise.compare``) AND
    ``scenario_metrics.json`` aggregate + episodes equal once the METADATA key ``trained_categories`` is dropped from
    both (PLAN left-for-implementation item 3; scoring never reads it). ``raw=False``: aggregates only (stand rows,
    item 10). ``allow_new_keys`` / ``ref_keys_only``: the SHARED-record compare of a reference older than the harness
    (C1.7): raw keys only the new stamp has are allowed when named, and aggregate / episode keys only the new stamp has
    are ignored (both listed in the result)."""
    ref, new = Path(ref), Path(new)
    out = {"ref": str(ref), "new": str(new)}
    if raw:
        c = BCB().compare(ref, new, set(allow_new_keys))
        out.update({k: c.get(k) for k in ("episodes_ref", "episodes_new", "missing_episodes", "extra_episodes")})
        out["n_value_mismatch"] = len(c.get("value_mismatch") or {})
        out["n_key_mismatch"] = len(c.get("key_mismatch") or {})
        out["value_mismatch_sample"] = dict(list((c.get("value_mismatch") or {}).items())[:3])
        raw_ok = (not c["missing_episodes"] and not c["extra_episodes"] and not c["key_mismatch"]
                  and not c["value_mismatch"] and c["episodes_ref"] > 0)
    else:
        raw_ok = True
    a = read_json(ref / "scenario_metrics.json", {})
    b = read_json(new / "scenario_metrics.json", {})
    if ref_keys_only:
        out["new_only_aggregate_keys"] = sorted(set(b.get("aggregate") or {}) - set(a.get("aggregate") or {}))
        out["allowed_new_raw_keys"] = list(allow_new_keys)
        b = {k: _ref_keys(a.get(k), b.get(k)) for k in ("aggregate", "episodes")}
    whole = {k: a.get(k) == b.get(k) for k in ("aggregate", "episodes")}
    stripped = {k: _strip(a.get(k)) == _strip(b.get(k)) for k in ("aggregate", "episodes")}
    out["metrics_equal_whole"] = all(whole.values())
    out["metrics_equal_stripped"] = all(stripped.values()) if raw else stripped["aggregate"]
    out["only_metadata_differs"] = out["metrics_equal_stripped"] and not out["metrics_equal_whole"]
    out["raw_bitwise"] = raw_ok
    out["bitwise"] = bool(raw_ok and out["metrics_equal_stripped"])
    if not out["metrics_equal_stripped"]:
        aa, bb = (a.get("aggregate") or {}), (b.get("aggregate") or {})
        out["aggregate_keys_differ"] = sorted(k for k in set(aa) | set(bb) if _strip(aa.get(k)) != _strip(bb.get(k)))[:12]
    return out


# ------------------------------------------------------------------------------------------------ driver records
def driver_heads(cdir: Path = SC) -> dict:
    """{chain: newest reference record} of a driver campaign dir's state.json."""
    st = read_json(Path(cdir) / "state.json", {}) or {}
    refs = ((st.get("ladder") or {}).get("references") or {})
    return {c: v[-1] for c, v in refs.items() if v}


def driver_head(chain: str, cdir: Path = SC) -> dict | None:
    h = driver_heads(cdir).get(chain)
    return h if h and Path(h.get("ckpt", "")).is_file() else None


def a2_smoke_runs() -> dict:
    """stage -> the Phase-A (A2) smoke run dir (the driver's own orchestrator.log record; never by run name)."""
    out = {}
    if A2_LOG.is_file():
        for m in re.finditer(r"reference PF_(\w+?)_s18 s18 recorded \((X\d+), head (\S+?)\)", A2_LOG.read_text()):
            out[m.group(1)] = Path(m.group(3)).parent
    return out


def saved_ckpts(run_dir: Path) -> list:
    return sorted((i, p) for p in Path(run_dir).glob("model_*.pt") if (i := iter_of(p)) is not None)


def nearest_ckpt(run_dir: Path, target: int):
    """The saved checkpoint nearest to iteration ``target`` (ties -> the earlier), as select_student maps offsets."""
    s = saved_ckpts(run_dir)
    if not s:
        return None
    i, p = min(s, key=lambda ip: (abs(ip[0] - target), ip[0]))
    return {"iter": i, "ckpt": p, "target": target}


def pilot_run() -> Path | None:
    h = driver_head("PP_p3a_s18")
    return Path(h["ckpt"]).parent if h else None


def pilot_ckpt_at(offset: int):
    rd = pilot_run()
    return nearest_ckpt(rd, S1_ITER + offset) if rd else None


def student_run(name: str) -> Path | None:
    """The newest distill run dir ``logs/rsl_rl/crab_hex_omni_student/<date>_NAME`` (train.py --run_name NAME)."""
    hits = sorted(p for p in (LOGS / STUDENT_EXP).glob(f"*_{name}") if p.is_dir())
    return hits[-1] if hits else None


def student_final(name: str, start: int, iters: int) -> Path | None:
    rd = student_run(name)
    if rd is None:
        return None
    p = rd / f"model_{start + iters - 1}.pt"
    return p if p.is_file() else None


# ------------------------------------------------------------------------------------------------ the payload rules
def level_read(stamps: dict) -> dict:
    """payload_read.level_reads on one level's rows (the stage's trained translate categories: p2a's)."""
    tt = [c for c in ctx().plan["stages"]["p2a"]["trained"] if c in TRANSLATE]
    return PR().level_reads({k: Path(v) for k, v in stamps.items()}, th_doc(), trained_translate=tt)


def curve_names(prefix: str, kg: int) -> str:
    return f"{prefix}_L{int(kg):03d}"


def np_rule(reads: dict, R_used: dict) -> dict:
    """DESIGN 2.2 / STEPC_RULES C3.4: ``reads`` {offset: {cat: {survival, ratio}}} at the 90-kg level (suite es1-2
    pooled). The first offset >= 1000 at which EVERY trained translate category's survival and ratio moved by less
    than R (thresholds_v0 R_used: cat.<c>.survival / cat.<c>.ratio) from the previous read; clamped to [1500, 3000]
    (already a multiple of 500). None of them -> 3000 (flagged ``not_converged``)."""
    offs = sorted(int(o) for o in reads)
    steps = []
    hit = None
    for prev, cur in zip(offs, offs[1:]):
        moves, ok = {}, True
        for c, v in reads[cur].items():
            for k in ("survival", "ratio"):
                a, b = (reads[prev].get(c) or {}).get(k), (v or {}).get(k)
                Rk = R_used.get(f"cat.{c}.{k}")
                if a is None or b is None or Rk is None:
                    ok = False
                    moves[f"{c}.{k}"] = None
                    continue
                d = abs(b - a)
                moves[f"{c}.{k}"] = {"delta": d, "R": Rk, "below": d < Rk}
                ok = ok and d < Rk
        steps.append({"from": prev, "to": cur, "all_below_R": ok, "moves": moves})
        if hit is None and ok and cur >= 1000:
            hit = cur
    complete = offs == list(PILOT_OFFSETS)
    n_p = None
    if hit is not None:
        n_p = min(max(hit, NP_CLAMP[0]), NP_CLAMP[1])
    elif complete:
        n_p = NP_CLAMP[1]
    return {"N_p": n_p, "first_settled_read": hit, "not_converged": hit is None and complete, "steps": steps,
            "reads_complete": complete, "offsets": offs}


def suite90_read(stamps: list) -> dict:
    """{cat: {survival, ratio}} pooled over suite es1-2 (n-weighted survival; mean of the env-seed ratio medians)."""
    cats = [TH().suite_categories(Path(s)) for s in stamps]
    out = {}
    for c in TRANSLATE:
        rows = [x.get(c) for x in cats if x.get(c)]
        if not rows:
            continue
        n = sum(r["n"] for r in rows)
        sv = sum(r["survival"] * r["n"] for r in rows) / n if n else None
        rt = [r["ratio"] for r in rows if r.get("ratio") is not None]
        out[c] = {"survival": sv, "ratio": (sum(rt) / len(rt)) if rt else None, "n": n}
    return out


# ------------------------------------------------------------------------------------------------ tfevents
def tf_series(run_dir: Path, tags=None) -> dict:
    return LW().read_series(Path(run_dir), tags)


def tf_walltimes(run_dir: Path, tag: str) -> list:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(int(e.step), float(e.wall_time)) for e in ea.Scalars(tag)]


def s_per_it(run_dir: Path, tag: str, skip: int = 20) -> dict:
    """Median wall seconds per training iteration from the tfevents wall times of a per-iteration tag, skipping the
    first ``skip`` iterations (warm-up)."""
    w = tf_walltimes(run_dir, tag)
    if len(w) < skip + 5:
        return {"s_per_it": None, "n": len(w), "why": f"{len(w)} logged iterations of {tag}"}
    w = w[skip:]
    d = sorted((b[1] - a[1]) / max(1, b[0] - a[0]) for a, b in zip(w, w[1:]) if b[0] > a[0])
    med = d[len(d) // 2]
    span = (w[-1][1] - w[0][1]) / max(1, w[-1][0] - w[0][0])
    return {"s_per_it": med, "s_per_it_span": span, "n": len(w), "h_per_1k": med * 1000 / 3600.0,
            "h_per_1k_span": span * 1000 / 3600.0}


def tf_bitwise(a_run: Path, b_run: Path, *, ignore=("Perf/",), ignore_suffix=("/time",)) -> dict:
    """Every shared scalar tag of two runs bitwise at every shared step (timing tags ignored); first divergence."""
    import numpy as np  # noqa: PLC0415

    A, B = RR().scalars(Path(a_run)), RR().scalars(Path(b_run))
    shared = sorted(t for t in set(A) & set(B) if not t.startswith(ignore) and not t.endswith(ignore_suffix))
    differ = []
    for t in shared:
        st = sorted(set(A[t]) & set(B[t]))
        if not st:
            continue
        va = np.array([A[t][s] for s in st], dtype=np.float32)
        vb = np.array([B[t][s] for s in st], dtype=np.float32)
        if not np.array_equal(va, vb, equal_nan=True):
            i = int(np.argmax(va != vb))
            differ.append({"tag": t, "first_step": int(st[i]),
                           "max_abs": float(np.nanmax(np.abs(va.astype(float) - vb.astype(float))))})
    return {"a": str(a_run), "b": str(b_run), "shared": len(shared), "only_a": sorted(set(A) - set(B))[:20],
            "only_b": sorted(set(B) - set(A))[:20], "differ": differ, "bitwise": not differ and bool(shared)}


# ------------------------------------------------------------------------------------------------ misc
def boot_se_median(vals, n_boot: int = 2000, seed: int = 20261016):
    import numpy as np  # noqa: PLC0415

    v = np.asarray([x for x in vals if x is not None and math.isfinite(x)], float)
    if len(v) < 3:
        return None
    g = np.random.default_rng(seed)
    return float(np.median(v[g.integers(0, len(v), (n_boot, len(v)))], axis=1).std())


def isaac_lines() -> list:
    return TL().isaac_lines()

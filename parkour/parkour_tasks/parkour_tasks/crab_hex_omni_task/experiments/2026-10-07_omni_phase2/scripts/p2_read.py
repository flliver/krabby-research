#!/usr/bin/env python3
"""Omni phase 2 -- the pre-registered reads (CHARTER 5 noise bands and anchor checks, 6.1 per-seed ladder, 6.2 recipe
read, 6.3 field-vs-ring, 6.4 sibling recommendations, 6.5 p2b trigger, 7 ranking inputs). CPU only, read-only.

INPUTS. Records $OUT/arms/<ref>.record.json (scripts/run_arm.sh: run_dir, checkpoint, driver_stamp). Eval stamps
$OUT/lr/<name>/<row>/seed00<k>/<ts>/ (run_arm.sh --row-eval / --lr-eval), each with its code_state.json sidecar.
Names: anchors ANCW_s16 / ANCW_s17 / ANCW_s18 (anchors.yaml; ANC_s* = the ramp-era set, CHARTER 9); arm heads = the reference (P2__A1_s18 ...); intermediate
checkpoints <ref>_m16000 / _m17500 / _m19000. Hand classes: <ARMS>/<name>_lr.json + _lr_along.json read through the
precedent's x6k_read.read_point (feet AND along, fail closed: a missing read is NA, never FAST).

THE READS (one point per evaluated checkpoint):
  completion  the 8-hold rows' schedule_completion_rate (field_hard / field_light / ring_obst), POOLED over env_seeds
              1 and 2: (c1*n1 + c2*n2) / (n1 + n2) -- n = 200 for anchors and arm heads alike (CHARTER 6.0).
  flat list   from suite__omni at env_seed 1 (an arm head: its driver suite stamp; an anchor: its row-eval):
              tracking error |ratio - 1| of fwd / back / side / diag (by_category.<c>.ratio.median; larger = worse),
              turn facing_closed and yaw_progress (medians; smaller = worse), move_turn survival (mean of the
              move_turn / move_turn_r survival_by_hold; smaller = worse), translate survival = every fwd / lat_l / lat_r
              / back / diag hold's survival_by_hold.
  hands       side22_25lr__omni (env_seed 1): FAST only if feet AND along FAST; fall share (verdict.fall_share).
  dose        median |mu| over the run's last 100 logged iterations, cams and rods (x6k_read.tail_reads; <= 1.5 ok).
  jitter      mean over fwd / back / side / diag / turn / move_turn of by_category.<c>.action_jitter.median.
  stand       by_category.stand tripod / steps_per_s / six_down_frac medians and the lift clearance from the raws:
              per stand episode, per foot, p95 - p5 of (foot z - terrain z) over the steady hold window, median over
              the six feet, then median over episodes. (Measured against sf/STAND_STEPPING.md T1: s18 / s16 / s17 c6x
              0.0825 / 0.1232 / 0.0763 vs T1's 0.080 / 0.122 / 0.074; the s18 c6 tap reads 0.024 vs T1 0.035 -- both
              under the 0.05 m line.) The same estimator is applied to every head compared.
  power       mean over the moving-category episodes' steady windows of sum over the ACTUATED joints (run_meta
              action_joint_names) of |tau * qdot| (applied_torque x joint_vel), W.
  exposure    (6.3, one estimator for both arms and both rows) from the env_seed-1 raws: share of steps with the base
              beyond 1.7 m of its spawn, share of steps with any foot over a block or pit (|terrain z - the spawn's
              median terrain z| > 0.02 m) and block steps per episode.
  levels      (6.3, side by side) the run's Metrics/base_parkour/terrain_levels (all tiles) and, when logged,
              terrain_levels_obst (obstacle tiles): mean of the fork's first 20 logged iterations ("start") and of its
              last 100 (the x6k tail window), from the arm's run_dir tfevents.
  field_turn  (7, the "turn band" stand-in -- CHARTER 9 reading: turn_band__omni is not evaluated) field_turn__omni at
              env_seed 1: completion, fall share and the turn_l90 / turn_r90 holds' facing_closed and yaw_progress_rate
              medians (mean of the two holds).

CHARTER 5 NOISE BANDS, per anchor seed and metric: |env_seed 1 - env_seed 2| of that anchor, floored at
  completion / survival: max(|d|, the pooled binomial SE sqrt(p(1-p)/n) at the pooled p and n, 0.04);
  |ratio - 1| and facing_closed: max(|d|, 0.02);  yaw_progress: max(|d|, 0.005).
  The suite metrics (flat list) use s18's band for every seed (s18 also runs its suite at env_seed 2).

CHARTER 6.1 PER SEED vs its own anchor, first match decides:
  INVALID EVAL  a needed stamp lacks KRABBY_OMNI_CLOCK_HIDE=1, KRABBY_STD_PARAM soft, KRABBY_OMNI_SPEED_ELLIPSE
                0.35:0.35:0.25 or KRABBY_OMNI_FACING_OBS_CLIP 0.6; or its row env block (run_meta env_vars_manifest),
                or its sidecar's code_sha / eval_sha / dirty, differs from the anchor's same row; a missing sidecar is
                INVALID, never equal.
  PENDING       a needed read missing or NA (a NA hand class included).
  HAND LOSS     a side hand not FAST, or hold-fall share > 0.15.
  DOSE          median |mu| > 1.5 (cams or rods) over the last 100 iterations.
  FLAT FAIL     a listed metric worse than the anchor by MORE than max(floor, 2 band) (floor 0.10; 0.03 for
                yaw_progress), or a translate hold's survival < 0.95.
  NO GAIN       NOT [field_hard >= anchor + max(0.10, 2 band) AND field_light >= anchor - max(0.03, band) AND
                ring_obst >= anchor - max(0.05, band)].
  COST          a gain, but a listed metric worse by MORE than max(margin, band); margins |ratio - 1| 0.03, fc 0.03,
                yp 0.01, move_turn survival 0.05.
  PASS          otherwise.
CHARTER 6.2: A1 s18 PASS continues, anything else stops (--gate). At 3 seeds: ADOPTABLE (>= 2/3 PASS and no HAND
LOSS) > TRADE-OFF (>= 2/3 gain and one metric named by COST in >= 2/3) > NOT ADOPTABLE (<= 1/3 gain or >= 2/3 HAND
LOSS) > MIXED; any INVALID / PENDING seed -> PENDING.
CHARTER 6.4 siblings vs A1 s18 (paired, n = 1; every verdict stops and asks): S1 / E1 KEEP if both hands FAST, side
feet_ss (hands-averaged lr feet) drops by <= max(0.02, band) [no feet band is defined in CHARTER 5: the 0.02 m/s
floor alone], jitter down >= 25 %, and nothing loses beyond max(margin, band) [the listed metrics at the CHARTER 6.1
margins, and -- this reader's reading of "nothing" -- the obstacle completions at the gain test's hold-the-line
tolerances, field_hard at 0.05]; E1 also needs mean |tau*qdot| down (stand stepping reported). F1 KEEP (low tap) if
A1 s18 still marches (clearance > 0.05 m and stand tripod > 0.4), F1's stand clearance <= 0.05 m and tripod <= 0.4,
and nothing loses; if A1 s18 already taps F1 is NOT RECOMMENDED (no effect to attribute) and reported.
CHARTER 6.5 p2b trigger: gain in >= 2/3 AND a remaining deficit (pooled field_hard < 0.85, or field_hard's walking-hold
tracking_ratio median, seed mean, < 0.7) in >= 2/3 -> stop and ask.
CHARTER 7: intermediate checkpoints (--m-screen): field_hard at env_seed 1 (n = 100) beats the head's by more than
max(0.10, 2 band) -> an env_seed-2 read; it REPLACES the head only if the pooled (n = 200) margin holds with floors
held (no HAND LOSS, no FLAT FAIL vs the anchor). Ranking inputs (floors, obstacle rank-sum, phase-1 rank-sum, jitter
and stand stepping, turn band) are reported for the judges; the ranking is theirs.

    p2_read.py [--json OUT] [--md OUT] [--no-tb] [--kept S1,E1]
    p2_read.py --anchors-check           CHARTER 5 discrimination + inertness + every anchor stamp's
                                         run_meta.checkpoint_sha256 == anchors.yaml: exit 0 ok / 4 stop and ask
    p2_read.py --gate                    A1 s18 (CHARTER 6.2): exit 0 PASS / 4 anything else
    p2_read.py --field-vs-ring           CHARTER 6.3 (descriptive), exit 0
    p2_read.py --sibling S1|E1|F1        CHARTER 6.4: prints the recommendation; exit 4 (always stop and ask)
    p2_read.py --recipe                  CHARTER 6.2 / 6.5: exit 4 when the p2b trigger fires or a seed is PENDING, else 0
    p2_read.py --m-screen REF            names of REF's intermediate checkpoints needing an env_seed-2 field_hard read
    p2_read.py --calibrate RUN_DIR       CHARTER 5 r-bar -> KRABBY_TERRAIN_PROMOTE "<up>:<down>": r-bar = the mean of
                                         Metrics/base_parkour/track_ratio_obst_p50 over the smoke's last 100 logged
                                         iterations (its tfevents; CHARTER 9 operational definition). A per-episode
                                         DUMP file (.json / .npz) is still accepted in place of RUN_DIR.
    p2_read.py --smoke-s5 STAMP|NAME     CHARTER 5 S5 (pre-launch audit B1): the field_light__omni stamp of
                                         run_arm.sh --row-eval <NAME> <s18 head> <s18 env stamp> scenarios_p2.yaml
                                         field_light__omni 1 --episodes 7 (a NAME resolves to its newest seed001
                                         stamp): 7 episodes, the s18 head's sha256, ZERO stand-hold falls (a fall
                                         whose last segment is the prologue or the stand hold; read from the raws,
                                         fail closed). Exit 0 PASS / 4.
    p2_read.py --write-smokes-ok --smokes DIR --s5 STAMP|NAME [--no-charter-log]
                                         writes $OUT/SMOKES_OK only when smokes/<UTC>/smokes.json (scripts/p2_smoke.py)
                                         has S1-S4 PASS on clean code, S3's toe radius <= 1.6 m and S4's numbers,
                                         S5 passes, the CAL1 fork smoke's record shows its dump audit PASS, and the
                                         smokes' and S5's code state equal the LIVE code state. It logs the S3 toe
                                         radius and the S4 numbers in CHARTER 9 (one appended line). Exit 0 / 4.
    p2_read.py --smokes-ok-check         the queue's gate before anything runs: SMOKES_OK present, PASS, complete, and
                                         its code state == the live one. Exit 0 / 4.
    p2_read.py --cal2-gate               the queue's gate before P2__A1_s18 (pre-launch audit M2): P2__CAL2_s18's
                                         record is a smoke, not killed, its dump audit PASS, and its PROMOTE == the
                                         A1 chain's KRABBY_TERRAIN_PROMOTE in candidates.yaml (== CAL2's own). 0 / 4.
    (--out DIR points those four at another $OUT; P2_OUT in the environment does the same for every read.)
    p2_read.py --self-test
"""
from __future__ import annotations

import argparse
import datetime
import functools
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
OUT = Path(os.environ.get("P2_OUT") or R / "parkour/outputs/omni_plan/phase2")
ARMS = OUT / "arms"
LROUT = OUT / "lr"
PREC = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-01_ppo_construction/scripts"
POLICY_MANIFEST = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/manifest.yaml"

FH, FL, RO, FT, FS = "field_hard__omni", "field_light__omni", "ring_obst__omni", "field_turn__omni", "field_suite__omni"
SUITE, SLOW, LR = "suite__omni", "slow__omni", "side22_25lr__omni"
COMP_ROWS = {"field_hard": FH, "field_light": FL, "ring_obst": RO}
SEEDS = (16, 17, 18)
GATE_SEED = 18
HANDS = ("l", "r")
FAST = "FAST"
EPS = 1e-12
#: the CHARTER 6.1 list: key -> (direction, floor, margin); direction +1 = larger is worse, -1 = smaller is worse
LISTED = {"err_fwd": (+1, 0.10, 0.03), "err_back": (+1, 0.10, 0.03), "err_side": (+1, 0.10, 0.03),
          "err_diag": (+1, 0.10, 0.03), "turn_fc": (-1, 0.10, 0.03), "turn_yp": (-1, 0.03, 0.01),
          "mt_surv": (-1, 0.10, 0.05)}
BAND_FLOOR = {"err_fwd": 0.02, "err_back": 0.02, "err_side": 0.02, "err_diag": 0.02, "turn_fc": 0.02,
              "turn_yp": 0.005, "mt_surv": 0.04}
COMP_BAND_FLOOR = 0.04
TRANSLATE_HOLDS = ("fwd", "lat_l", "lat_r", "back", "diag")
TRANSLATE_MIN = 0.95
TRIP = 0.15
GAIN_FH, HOLD_FL, HOLD_RO, HOLD_FH = 0.10, 0.03, 0.05, 0.05
NEEDED_STAMPS = ("suite@1", "lr@1", f"{FH}@1", f"{FH}@2", f"{FL}@1", f"{FL}@2", f"{RO}@1", f"{RO}@2")
#: every stamp whose env / env block / sidecar is compared with the anchor's (CHARTER 6.1 INVALID EVAL): the needed
#: stamps plus field_suite (feeds the CHARTER 7 obstacle rank-sum) and field_turn (the turn-band stand-in)
VALIDITY_STAMPS = NEEDED_STAMPS + (f"{FS}@1", f"{FT}@1")
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
Q_CATS = ("fwd", "back", "side", "diag", "turn", "move_turn")
DISCRIM = (0.15, 0.85)
DISCRIM_GAP = 0.10
P2B_FH, P2B_TRACK = 0.85, 0.7
FEET_DROP, JITTER_CUT = 0.02, 0.75
TAP_CLEAR, TAP_TRIPOD = 0.05, 0.4
MCKPTS = (16000, 17500, 19000)
PAD_R = 1.7
BLOCK_DZ = 0.02
SIBLINGS = ("S1", "E1", "F1")
TURN_HOLDS = ("turn_l90", "turn_r90")
LEVEL_TAGS = {"all": "Metrics/base_parkour/terrain_levels", "obst": "Metrics/base_parkour/terrain_levels_obst"}
LEVEL_START_N, LEVEL_TAIL = 20, 100
#: CHARTER 5 r-bar, operationally (CHARTER 9 log, 2026-10-06): the mean of this tag over the calibration smoke's last
#: CAL_TAIL logged iterations. Each logged value is the median per-episode ep_track_dist / ep_cmd_dist (the ratio
#: promotion_mask gates on; ep_cmd_dist < min_cmd_dist_m excluded) over the ledger window's last exposure_window
#: (1024) obstacle-tile episodes -- on a field run, the field tiles (omni_exposure.OMNI_OBST_RATIO_KEY).
CAL_TAG = "Metrics/base_parkour/track_ratio_obst_p50"
CAL_TAIL = 100


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def X():
    """The precedent's x6k_read (read_point, tail_reads; it loads verdict.py)."""
    return _load(PREC / "x6k_read.py", "_p2_x6k")


# ================================================================================================ stamp reads
def newest_stamp(name: str, row: str, seed: int = 1, lrout: Path = LROUT) -> str | None:
    d = lrout / name / row / f"seed{seed:03d}"
    hits = [p for p in sorted(d.glob("*/")) if stamp_complete(p)]
    return str(hits[-1]) if hits else None


def stamp_complete(p) -> bool:
    """A stamp is complete when the harness wrote complete.json (category_suite rows) or run_arm.sh wrote
    p2_complete.json after a clean exit of a non-suite (8-hold schedule) row."""
    p = Path(p)
    return (p / "complete.json").is_file() or (p / "p2_complete.json").is_file()


def _json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError, TypeError):
        return None


def stamp_validity(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    meta = _json(Path(stamp) / "run_meta.json") or {}
    env = meta.get("env_vars") or {}

    def same(k, want):
        got = env.get(k)
        if got is None:
            return False
        if k == "KRABBY_OMNI_FACING_OBS_CLIP":
            try:
                return abs(float(got) - float(want)) < 1e-9
            except ValueError:
                return False
        return str(got).strip() == want
    sc = _json(Path(stamp) / "code_state.json")
    return {"env4": all(same(k, v) for k, v in ENV4.items()),
            "env4_missing": [k for k, v in ENV4.items() if not same(k, v)],
            "manifest_env": dict(meta.get("env_vars_manifest") or {}),
            "sidecar": (sc.get("code_sha"), sc.get("eval_sha"), bool(sc.get("dirty"))) if sc else None,
            "ckpt_sha": meta.get("checkpoint_sha256")}


def comp_read(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    a = (_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}
    if a.get("schedule_completion_rate") is None:
        return None
    n = int(a.get("n_episodes") or 0)
    return {"rate": float(a["schedule_completion_rate"]), "n": n,
            "falls": int((a.get("termination_reasons") or {}).get("fall", 0)),
            "track": (a.get("tracking_ratio") or {}).get("median")}


def _med(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d.get("median") if isinstance(d, dict) else d


def suite_reads(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    a = ((_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}).get("suite")
    if not a:
        return None
    by, surv = a.get("by_category") or {}, a.get("survival_by_hold") or {}

    def err(c):
        r = _med(by, c, "ratio")
        return None if r is None else abs(r - 1.0)
    mt = [surv.get(h) for h in ("move_turn", "move_turn_r")]
    jit = [_med(by, c, "action_jitter") for c in Q_CATS]
    cats = {}
    for c, holds in ((c, [h for h, cc in (a.get("category_of_hold") or {}).items() if cc == c]) for c in by):
        sv = [surv.get(h) for h in holds if surv.get(h) is not None]
        cats[c] = {"survival": round(sum(sv) / len(sv), 3) if sv else None,
                   "ratio": _med(by, c, "ratio"), "facing_closed": _med(by, c, "facing_closed"),
                   "yaw_progress": _med(by, c, "yaw_progress")}
    return {"err_fwd": err("fwd"), "err_back": err("back"), "err_side": err("side"), "err_diag": err("diag"),
            "turn_fc": _med(by, "turn", "facing_closed"), "turn_yp": _med(by, "turn", "yaw_progress"),
            "mt_surv": (sum(mt) / 2) if None not in mt else None,
            "trans_surv": {h: surv.get(h) for h in TRANSLATE_HOLDS},
            "jitter": (sum(jit) / len(jit)) if None not in jit else None,
            "stand": {k: _med(by, "stand", k) for k in ("tripod", "steps_per_s", "six_down_frac")},
            "categories": cats}


def turn_row_reads(stamp: str | None) -> dict | None:
    """field_turn__omni: completion, fall share and the turn holds' facing_closed / yaw_progress_rate medians."""
    if not stamp:
        return None
    a = (_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}
    n = int(a.get("n_episodes") or 0)
    if not n:
        return None
    fac = a.get("facing_by_hold") or {}
    fc = [_med(fac, h, "facing_closed") for h in TURN_HOLDS]
    yp = [_med(fac, h, "yaw_progress_rate") for h in TURN_HOLDS]
    return {"completion": a.get("schedule_completion_rate"), "n": n,
            "fall_share": int((a.get("termination_reasons") or {}).get("fall", 0)) / n,
            "turn_fc": sum(fc) / 2 if None not in fc else None, "turn_yp": sum(yp) / 2 if None not in yp else None}


def levels_summary(series, start_n: int = LEVEL_START_N, tail: int = LEVEL_TAIL) -> dict | None:
    """(steps, values) -> mean of the first ``start_n`` logged points ("start") and of the last ``tail`` iterations."""
    if series is None or not len(series[0]):
        return None
    pts = sorted(zip((int(x) for x in series[0]), (float(v) for v in series[1])))
    last = pts[-1][0]
    head = [v for _, v in pts[:start_n]]
    tl = [v for st, v in pts if st >= last - tail + 1]
    return {"start": sum(head) / len(head), "tail": sum(tl) / len(tl), "first_step": pts[0][0], "last_step": last}


def terrain_levels(run_dir: str | None) -> dict | None:
    """CHARTER 6.3 terrain levels: all-tile and (when logged) obstacle-tile level start / tail of one run."""
    if not run_dir:
        return None
    ser = X().V.tb_series(Path(run_dir), LEVEL_TAGS)
    return {k: levels_summary(ser.get(k)) for k in LEVEL_TAGS}


# ------------------------------------------------------------------------------------------------ raw estimators
def _raw_episodes(stamp: str, keep):
    import numpy as np  # noqa: PLC0415

    meta = _json(Path(stamp) / "run_meta.json") or {}
    files = {int(f.stem.split("_")[1]): f for f in (Path(stamp) / "raw").glob("episode_*.npz")}
    labels = meta.get("hold_labels") or []
    for e in meta.get("env_index_map") or [{"env_index": i, "label": None} for i in sorted(files)]:
        if keep(e) and e["env_index"] in files:
            yield meta, labels, e, np.load(files[e["env_index"]])


@functools.lru_cache(maxsize=64)
def stand_clearance(stamp: str | None) -> float | None:
    """p95 - p5 of foot height above terrain over the steady stand window; median over feet, then over episodes."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    vals = []
    for _meta, labels, _e, d in _raw_episodes(stamp, lambda e: e.get("label") == "stand"):
        mask = (d["segment_id"] == labels.index("stand")) & d["steady_mask"]
        if mask.sum() < 25:
            continue
        z = d["foot_pos_w"][mask, :, 2] - d["terrain_z"][mask]
        vals.append(float(np.median(np.percentile(z, 95, axis=0) - np.percentile(z, 5, axis=0))))
    return float(np.median(vals)) if vals else None


@functools.lru_cache(maxsize=64)
def actuated_power(stamp: str | None) -> float | None:
    """Mean over moving-category steady steps of sum_j |tau_j * qdot_j| on the actuated joints, W."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    tot, n = 0.0, 0
    for meta, labels, e, d in _raw_episodes(stamp, lambda e: e.get("label") not in (None, "stand", "settle")):
        names, act = meta.get("joint_names") or [], meta.get("action_joint_names") or []
        idx = [names.index(j) for j in act if j in names]
        if not idx:
            return None
        mask = (d["segment_id"] == labels.index(e["label"])) & d["steady_mask"]
        p = np.abs(d["applied_torque"][mask][:, idx] * d["joint_vel"][mask][:, idx]).sum(axis=1)
        tot += float(p.sum()); n += int(mask.sum())
    return tot / n if n else None


@functools.lru_cache(maxsize=64)
def exposure(stamp: str | None) -> dict | None:
    """6.3's one estimator: beyond-pad step share, block/pit step share and block steps per episode."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    beyond, block, bsteps = [], [], []
    for _m, _l, _e, d in _raw_episodes(stamp, lambda e: True):
        xy = d["root_pos_w"][:, :2]
        if len(xy) < 2:
            continue
        beyond.append(float(np.mean(np.linalg.norm(xy - xy[0], axis=1) > PAD_R)))
        ground = float(np.median(d["terrain_z"][0]))
        over = np.any(np.abs(d["terrain_z"] - ground) > BLOCK_DZ, axis=1)
        block.append(float(np.mean(over)))
        bsteps.append(int(over.sum()))
    if not beyond:
        return None
    return {"beyond_pad_frac": float(np.mean(beyond)), "block_step_frac": float(np.mean(block)),
            "block_steps_per_ep": float(np.mean(bsteps)), "n": len(beyond)}


# ================================================================================================ points
def load_point(name: str, *, suite_stamp: str | None = None, run_dir: str | None = None, arms: Path = ARMS,
               lrout: Path = LROUT, with_tb: bool = True, with_raw: bool = False) -> dict:
    """Every read of one evaluated checkpoint (an anchor, an arm head or an intermediate)."""
    st = {"suite@1": suite_stamp or newest_stamp(name, SUITE, 1, lrout), "suite@2": newest_stamp(name, SUITE, 2, lrout),
          "lr@1": newest_stamp(name, LR, 1, lrout), f"{SLOW}@1": newest_stamp(name, SLOW, 1, lrout),
          f"{FT}@1": newest_stamp(name, FT, 1, lrout), f"{FS}@1": newest_stamp(name, FS, 1, lrout)}
    for row in COMP_ROWS.values():
        for s in (1, 2):
            st[f"{row}@{s}"] = newest_stamp(name, row, s, lrout)
    pt = {"name": name, "stamps": st, "valid": {k: stamp_validity(v) for k, v in st.items()},
          "comp": {k: {s: comp_read(st[f"{row}@{s}"]) for s in (1, 2)} for k, row in COMP_ROWS.items()},
          "flat_completion": (comp_read(st[f"{SLOW}@1"]) or {}).get("rate"),
          "suite": suite_reads(st["suite@1"]), "suite2": suite_reads(st["suite@2"]),
          "field_suite": suite_reads(st[f"{FS}@1"]), "field_turn": turn_row_reads(st[f"{FT}@1"]),
          "hands": None, "fall": None, "feet_side": None, "dose_ok": None, "dose": None, "run_dir": run_dir}
    p = X().read_point(str(arms / f"{name}_lr.json"), str(arms / f"{name}_lr_along.json"), name)
    if p:
        pt["hands"], pt["fall"] = p["hands"], p["fall"]
        f = [p["feet"][h] for h in HANDS]
        pt["feet_side"] = sum(f) / 2 if None not in f else None
    if run_dir and with_tb:
        t = X().tail_reads(run_dir)
        pt["dose_ok"], pt["dose"] = t.get("dose_ok"), {k: t.get(k) for k in ("mu_med_cam", "mu_med_rod")}
    if with_raw:
        pt["stand_clearance"] = stand_clearance(st["suite@1"])
        pt["power"] = actuated_power(st["suite@1"])
        pt["exposure"] = {k: exposure(st[f"{row}@1"]) for k, row in (("field_hard", FH), ("ring_obst", RO))}
    return pt


def pooled(pt: dict, key: str) -> tuple[float, int] | None:
    c = (pt.get("comp") or {}).get(key) or {}
    a, b = c.get(1), c.get(2)
    if not a or not b or not (a["n"] and b["n"]):
        return None
    n = a["n"] + b["n"]
    return (a["rate"] * a["n"] + b["rate"] * b["n"]) / n, n


def binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("inf")


def bands(anchor: dict, s18: dict) -> dict:
    """CHARTER 5: completion bands from ``anchor`` (seed 1 vs 2), suite bands from ``s18`` (suite@1 vs suite@2).
    A band that cannot be computed is None (-> PENDING)."""
    out = {}
    for k in COMP_ROWS:
        c = (anchor.get("comp") or {}).get(k) or {}
        pl = pooled(anchor, k)
        out[k] = (max(abs(c[1]["rate"] - c[2]["rate"]), binom_se(*pl), COMP_BAND_FLOOR)
                  if pl and c.get(1) and c.get(2) else None)
    s1, s2 = s18.get("suite") or {}, s18.get("suite2") or {}
    for m, floor in BAND_FLOOR.items():
        a, b = s1.get(m), s2.get(m)
        if a is None or b is None:
            out[m] = None
        elif m == "mt_surv":
            out[m] = max(abs(a - b), binom_se((a + b) / 2, 200), floor)
        else:
            out[m] = max(abs(a - b), floor)
    return out


# ================================================================================================ 6.1 ladder (pure)
def invalid_reasons(pt: dict, ref: dict, keys=VALIDITY_STAMPS) -> list[str]:
    out = []
    for k in keys:
        va, vr = (pt.get("valid") or {}).get(k), (ref.get("valid") or {}).get(k)
        if va is None or vr is None:
            continue                                  # a missing stamp is PENDING's, not INVALID's
        if not va["env4"]:
            out.append(f"{pt['name']} {k}: stamp env lacks {va.get('env4_missing')}")
        if not vr["env4"]:
            out.append(f"{ref['name']} {k}: stamp env lacks {vr.get('env4_missing')}")
        if va["manifest_env"] != vr["manifest_env"]:
            out.append(f"{k}: row env block differs from {ref['name']}'s")
        if va["sidecar"] is None or vr["sidecar"] is None:
            out.append(f"{k}: code-state sidecar missing ({pt['name'] if va['sidecar'] is None else ref['name']})")
        elif va["sidecar"] != vr["sidecar"]:
            out.append(f"{k}: code_sha / eval_sha / dirty differ from {ref['name']}'s ({va['sidecar']} vs {vr['sidecar']})")
    return out


def pending_reasons(pt: dict, anc: dict, bd: dict, need_dose: bool = True) -> list[str]:
    miss = []
    for k in NEEDED_STAMPS:
        if not (pt.get("stamps") or {}).get(k):
            miss.append(f"{pt['name']} {k}")
        if not (anc.get("stamps") or {}).get(k) and k != "lr@1":
            miss.append(f"{anc['name']} {k}")
    for k in COMP_ROWS:
        if pooled(pt, k) is None:
            miss.append(f"{pt['name']} pooled {k}")
        if pooled(anc, k) is None:
            miss.append(f"{anc['name']} pooled {k}")
    for who in (pt, anc):
        s = who.get("suite") or {}
        miss += [f"{who['name']} suite {m}" for m in LISTED if s.get(m) is None]
    ts = (pt.get("suite") or {}).get("trans_surv") or {}
    miss += [f"{pt['name']} survival {h}" for h in TRANSLATE_HOLDS if ts.get(h) is None]
    h = pt.get("hands")
    if not h or any(h.get(x) in (None, "NA") for x in HANDS):
        miss.append(f"{pt['name']} hand classes (NA)")
    if pt.get("fall") is None:
        miss.append(f"{pt['name']} hold-fall share")
    if need_dose and pt.get("dose_ok") is None:
        miss.append(f"{pt['name']} dose read")
    miss += [f"band {k}" for k, v in bd.items() if v is None]
    return miss


def worse_by(pt: dict, ref: dict, m: str) -> float:
    d = LISTED[m][0]
    return d * ((pt["suite"][m]) - (ref["suite"][m]))


def gain_read(pt: dict, anc: dict, bd: dict) -> dict:
    fh, fl, ro = (pooled(pt, k)[0] for k in COMP_ROWS)
    afh, afl, aro = (pooled(anc, k)[0] for k in COMP_ROWS)
    need = afh + max(GAIN_FH, 2 * bd["field_hard"])
    ok_fh = fh >= need - EPS
    ok_fl = fl >= afl - max(HOLD_FL, bd["field_light"]) - EPS
    ok_ro = ro >= aro - max(HOLD_RO, bd["ring_obst"]) - EPS
    return {"gain": ok_fh and ok_fl and ok_ro, "field_hard": [fh, afh, need], "field_light_ok": ok_fl,
            "ring_obst_ok": ok_ro, "field_hard_ok": ok_fh}


def seed_verdict(pt: dict, anc: dict, bd: dict, need_dose: bool = True) -> dict:
    """CHARTER 6.1 for one seed (pure; first match decides). ``bd`` = bands(anchor, s18 anchor)."""
    out = {"name": pt.get("name"), "gain": None, "cost": []}
    inv = invalid_reasons(pt, anc)
    if inv:
        return dict(out, verdict="INVALID EVAL", why=inv)
    pend = pending_reasons(pt, anc, bd, need_dose)
    if pend:
        return dict(out, verdict="PENDING", why=pend)
    g = gain_read(pt, anc, bd)
    out["gain"], out["gain_read"] = g["gain"], g
    out["deltas"] = {m: worse_by(pt, anc, m) for m in LISTED}
    lost = [h for h in HANDS if pt["hands"][h] != FAST]
    if lost or pt["fall"] > TRIP + EPS:
        return dict(out, verdict="HAND LOSS", why=[f"hands {pt['hands']}", f"hold-fall share {pt['fall']:.3f}"])
    if need_dose and pt["dose_ok"] is False:
        return dict(out, verdict="DOSE", why=[f"median |mu| {pt.get('dose')} > 1.5"])
    ff = [f"{m} worse by {out['deltas'][m]:.4f} > max({LISTED[m][1]}, 2x{bd[m]:.4f})" for m in LISTED
          if out["deltas"][m] > max(LISTED[m][1], 2 * bd[m]) + EPS]
    ff += [f"translate survival {h} {v:.3f} < {TRANSLATE_MIN}" for h, v in pt["suite"]["trans_surv"].items()
           if v < TRANSLATE_MIN - EPS]
    if ff:
        return dict(out, verdict="FLAT FAIL", why=ff)
    if not g["gain"]:
        return dict(out, verdict="NO GAIN", why=[f"field_hard {g['field_hard'][0]:.3f} vs need {g['field_hard'][2]:.3f}; "
                                                  f"field_light ok {g['field_light_ok']}; ring_obst ok {g['ring_obst_ok']}"])
    cost = [m for m in LISTED if out["deltas"][m] > max(LISTED[m][2], bd[m]) + EPS]
    if cost:
        return dict(out, verdict="COST", cost=cost, why=[f"{m} worse by {out['deltas'][m]:.4f}" for m in cost])
    return dict(out, verdict="PASS", why=["gain, no flat loss"])


# ================================================================================================ 6.2 / 6.5 (pure)
def recipe_read(vs: dict) -> dict:
    """vs: {seed: seed_verdict} for 16, 17, 18."""
    if any(not vs.get(s) or vs[s]["verdict"] in ("INVALID EVAL", "PENDING") for s in SEEDS):
        return {"reading": "PENDING", "why": "a seed is missing, INVALID EVAL or PENDING"}
    n_pass = sum(vs[s]["verdict"] == "PASS" for s in SEEDS)
    n_hl = sum(vs[s]["verdict"] == "HAND LOSS" for s in SEEDS)
    n_gain = sum(bool(vs[s]["gain"]) for s in SEEDS)
    costs = {}
    for s in SEEDS:
        for m in vs[s].get("cost") or []:
            costs.setdefault(m, []).append(s)
    shared = sorted(m for m, ss in costs.items() if len(ss) >= 2)
    base = {"pass": n_pass, "hand_loss": n_hl, "gain": n_gain, "shared_cost": shared}
    if n_pass >= 2 and n_hl == 0:
        return dict(base, reading="ADOPTABLE")
    if n_gain >= 2 and shared:
        return dict(base, reading="TRADE-OFF")
    if n_gain <= 1 or n_hl >= 2:
        return dict(base, reading="NOT ADOPTABLE")
    return dict(base, reading="MIXED")


def p2b_trigger(vs: dict, pts: dict) -> dict:
    """CHARTER 6.5: gain in >= 2/3 and a remaining deficit in >= 2/3 (stop and ask)."""
    n_gain = sum(bool((vs.get(s) or {}).get("gain")) for s in SEEDS)
    deficit = {}
    for s in SEEDS:
        pt = pts.get(s) or {}
        fh = pooled(pt, "field_hard") if pt else None
        tr = [((pt.get("comp") or {}).get("field_hard") or {}).get(k) for k in (1, 2)] if pt else []
        tr = [x["track"] for x in tr if x and x.get("track") is not None]
        track = sum(tr) / len(tr) if tr else None
        deficit[s] = None if fh is None or track is None else (fh[0] < P2B_FH or track < P2B_TRACK)
    n_def = sum(bool(v) for v in deficit.values())
    return {"trigger": n_gain >= 2 and n_def >= 2, "gain": n_gain, "deficit": deficit,
            "pending": any(v is None for v in deficit.values())}


# ================================================================================================ 6.4 siblings (pure)
def _loses(sib: dict, a1: dict, bd: dict) -> list[str]:
    out = [f"{m} worse by {worse_by(sib, a1, m):.4f}" for m in LISTED
           if worse_by(sib, a1, m) > max(LISTED[m][2], bd[m]) + EPS]
    for k, tol in (("field_hard", HOLD_FH), ("field_light", HOLD_FL), ("ring_obst", HOLD_RO)):
        a, b = pooled(sib, k), pooled(a1, k)
        if a is None or b is None:
            out.append(f"{k} completion unread")
        elif a[0] < b[0] - max(tol, bd[k]) - EPS:
            out.append(f"{k} completion {a[0]:.3f} vs A1 {b[0]:.3f}")
    return out


def sibling_rec(arm: str, sib: dict, a1: dict, bd: dict) -> dict:
    """CHARTER 6.4 recommendation for S1 / E1 / F1 vs A1 s18 (``bd`` = s18's bands). Always stop and ask."""
    out = {"arm": arm, "stop_and_ask": True}
    inv = invalid_reasons(sib, a1)
    if inv:
        return dict(out, rec="INVALID EVAL", why=inv)
    need = [("suite", m) for m in LISTED] + [("hands", None), ("feet_side", None)]
    if arm in ("S1", "E1"):
        need.append(("suite", "jitter"))
    if arm == "E1":
        need.append(("power", None))
    if arm == "F1":
        need += [("stand_clearance", None), ("stand_tripod", None)]
    miss = []
    for who in (sib, a1):
        for k, m in need:
            v = (who.get("suite") or {}).get(m) if k == "suite" else (
                ((who.get("suite") or {}).get("stand") or {}).get("tripod") if k == "stand_tripod" else who.get(k))
            if v is None or (k == "hands" and any(v.get(h) in (None, "NA") for h in HANDS)):
                miss.append(f"{who.get('name')} {k}{'.' + m if m else ''}")
    miss += [f"band {k}" for k, v in bd.items() if v is None]
    if miss:
        return dict(out, rec="PENDING", why=miss)
    loses = _loses(sib, a1, bd)
    report = {"stand": {"sib": dict((sib["suite"].get("stand") or {}), clearance=sib.get("stand_clearance")),
                        "A1": dict((a1["suite"].get("stand") or {}), clearance=a1.get("stand_clearance"))},
              "loses": loses}
    if arm in ("S1", "E1"):
        both = all(sib["hands"][h] == FAST for h in HANDS)
        drop = a1["feet_side"] - sib["feet_side"]
        jit = sib["suite"]["jitter"] <= JITTER_CUT * a1["suite"]["jitter"] + EPS
        conds = {"both hands FAST": both, f"side feet_ss drop {drop:.4f} <= {FEET_DROP}": drop <= FEET_DROP + EPS,
                 f"jitter {sib['suite']['jitter']:.4f} <= 0.75 x {a1['suite']['jitter']:.4f}": jit,
                 "nothing loses": not loses}
        if arm == "E1":
            conds[f"mean |tau qdot| {sib['power']:.2f} < A1 {a1['power']:.2f}"] = sib["power"] < a1["power"]
        keep = all(conds.values())
        return dict(out, rec="KEEP" if keep else "DO NOT KEEP", conditions=conds, report=report)
    a1_tripod, sib_tripod = a1["suite"]["stand"]["tripod"], sib["suite"]["stand"]["tripod"]
    marches = a1["stand_clearance"] > TAP_CLEAR and a1_tripod > TAP_TRIPOD
    if not marches:
        return dict(out, rec="NOT RECOMMENDED", why=["A1 s18 already taps at stand (no effect to attribute)"],
                    report=report)
    conds = {f"F1 stand clearance {sib['stand_clearance']:.3f} <= {TAP_CLEAR}": sib["stand_clearance"] <= TAP_CLEAR + EPS,
             f"F1 stand tripod {sib_tripod:.3f} <= {TAP_TRIPOD}": sib_tripod <= TAP_TRIPOD + EPS,
             "nothing loses": not loses}
    return dict(out, rec="KEEP" if all(conds.values()) else "DO NOT KEEP", conditions=conds, report=report)


# ================================================================================================ 5 anchors (pure)
def anchors_check(anc: dict, s18_manifest: dict | None, shas: dict | None = None) -> dict:
    """CHARTER 5: discrimination per anchor seed, inertness of the s18 suite vs policy/manifest.yaml and, with
    ``shas`` ({seed: anchors.yaml sha256}), every evaluated anchor stamp's run_meta.checkpoint_sha256 (fail closed:
    a stamp without one fails)."""
    fails, rows = [], {}
    for s, want in (shas or {}).items():
        for k, v in sorted(((anc.get(s) or {}).get("valid") or {}).items()):
            if v is not None and v.get("ckpt_sha") != want:
                fails.append(f"s{s} {k}: stamp checkpoint_sha256 {v.get('ckpt_sha')} != anchors.yaml {want[:12]}...")
    for s in SEEDS:
        pt = anc.get(s)
        fh = pooled(pt, "field_hard") if pt else None
        fl = pooled(pt, "field_light") if pt else None
        rows[s] = {"field_hard": fh and fh[0], "field_light": fl and fl[0],
                   "flat_completion": (pt or {}).get("flat_completion")}
        if fh is None or fl is None:
            fails.append(f"s{s}: field_hard / field_light pooled read missing")
            continue
        if not DISCRIM[0] - EPS <= fh[0] <= DISCRIM[1] + EPS:
            fails.append(f"s{s}: field_hard completion {fh[0]:.3f} outside [{DISCRIM[0]}, {DISCRIM[1]}]")
        if fl[0] < fh[0] + DISCRIM_GAP - EPS:
            fails.append(f"s{s}: field_light {fl[0]:.3f} < field_hard {fh[0]:.3f} + {DISCRIM_GAP}")
    inert = None
    cats = (((anc.get(GATE_SEED) or {}).get("suite") or {}).get("categories")) or {}
    if not s18_manifest or not cats:
        fails.append("inertness unread (s18 suite or policy/manifest.yaml categories missing)")
    else:
        diffs = []
        for c, want in s18_manifest.items():
            got = cats.get(c) or {}
            for k, v in want.items():
                if k == "n":
                    continue
                g = got.get(k)
                if g is None or round(float(g), 3) != round(float(v), 3):
                    diffs.append(f"{c}.{k} {g} != {v}")
        inert = not diffs
        if diffs:
            fails.append("s18 suite does not reproduce policy/manifest.yaml: " + "; ".join(diffs))
    return {"ok": not fails, "fails": fails, "rows": rows, "inert": inert}


# ================================================================================================ 7 (pure)
def m_screen(head: dict, ms: dict, anc: dict, bd: dict) -> dict:
    """CHARTER 7 intermediate screen. ms: {iteration: point}. Returns per checkpoint 'needs_seed2' / 'replaces' / ..."""
    out = {}
    h1 = ((head.get("comp") or {}).get("field_hard") or {}).get(1)
    for it, m in ms.items():
        c1 = ((m.get("comp") or {}).get("field_hard") or {}).get(1)
        if not c1 or not h1 or bd.get("field_hard") is None:
            out[it] = {"status": "PENDING"}
            continue
        margin = max(GAIN_FH, 2 * bd["field_hard"])
        if not c1["rate"] > h1["rate"] + margin + EPS:
            out[it] = {"status": "NO", "n100": [c1["rate"], h1["rate"], margin]}
            continue
        pm, ph = pooled(m, "field_hard"), pooled(head, "field_hard")
        if pm is None or ph is None:
            out[it] = {"status": "NEEDS_SEED2", "n100": [c1["rate"], h1["rate"], margin]}
            continue
        floors = []
        if not m.get("hands") or any(m["hands"].get(x) != FAST for x in HANDS) or m.get("fall") is None \
                or m["fall"] > TRIP + EPS:
            floors.append("HAND LOSS / NA")
        s = m.get("suite") or {}
        if any(s.get(k) is None for k in LISTED) or not anc.get("suite"):
            floors.append("flat reads missing")
        else:
            floors += [k for k in LISTED if worse_by(m, anc, k) > max(LISTED[k][1], 2 * bd[k]) + EPS]
            floors += [h for h, v in (s.get("trans_surv") or {}).items() if v is None or v < TRANSLATE_MIN - EPS]
        holds = pm[0] > ph[0] + margin + EPS
        out[it] = {"status": "REPLACES HEAD" if holds and not floors else "NO (n=200)", "n200": [pm[0], ph[0], margin],
                   "floors_broken": floors}
    return out


def rank_inputs(cands: dict, vs: dict) -> dict:
    """CHARTER 7 ranking inputs (the judges rank). cands: {label: point}; vs: {label: seed_verdict}."""
    rows = {}
    for lab, pt in cands.items():
        s = pt.get("suite") or {}
        rows[lab] = {"floors": (vs.get(lab) or {}).get("verdict"),
                     "field_hard": (pooled(pt, "field_hard") or (None,))[0],
                     "field_light": (pooled(pt, "field_light") or (None,))[0],
                     "ring_obst": (pooled(pt, "ring_obst") or (None,))[0],
                     "field_mt_surv": (pt.get("field_suite") or {}).get("mt_surv"),
                     **{m: s.get(m) for m in LISTED}, "jitter": s.get("jitter"),
                     "stand_steps_per_s": (s.get("stand") or {}).get("steps_per_s"),
                     "stand_clearance": pt.get("stand_clearance"),
                     **{f"field_turn_{k}": (pt.get("field_turn") or {}).get(k)
                        for k in ("completion", "fall_share", "turn_fc", "turn_yp")}}

    def ranks(key, higher_better=True):
        vals = {k: v[key] for k, v in rows.items() if v[key] is not None}
        order = sorted(vals, key=lambda k: -vals[k] if higher_better else vals[k])
        out = {}
        for k in order:
            tied = [j for j in order if vals[j] == vals[k]]
            out[k] = sum(order.index(j) + 1 for j in tied) / len(tied)
        return out
    obst = [ranks(k) for k in ("field_hard", "field_light", "ring_obst", "field_mt_surv")]
    ph1 = [ranks(m, higher_better=LISTED[m][0] < 0) for m in LISTED]
    turn = [ranks(k) for k in ("field_turn_turn_fc", "field_turn_turn_yp")]
    for lab in rows:
        o = [r.get(lab) for r in obst]
        p = [r.get(lab) for r in ph1]
        t = [r.get(lab) for r in turn]
        rows[lab]["obstacle_rank_sum"] = sum(o) if None not in o else None
        rows[lab]["phase1_rank_sum"] = sum(p) if None not in p else None
        rows[lab]["turn_rank_sum"] = sum(t) if None not in t else None     # item 5, the field_turn stand-in
    return rows


# ================================================================================================ calibration
def promote_from_rbar(r: float) -> dict:
    """CHARTER 5 step 3: up = 0.9 r-bar, down = 0.5 r-bar, each rounded to 0.05."""
    rnd = lambda x: round(round(x / 0.05) * 0.05, 2)  # noqa: E731
    up, down = rnd(0.9 * r), rnd(0.5 * r)
    return {"r_bar": r, "up": up, "down": down, "promote": f"{up:g}:{down:g}"}


def calibrate(track, cmd, field_tile, valid=None) -> dict:
    """CHARTER 5: r-bar = median per-episode ep_track_dist / ep_cmd_dist on field-tile (valid) episodes;
    up = 0.9 r-bar, down = 0.5 r-bar, each rounded to 0.05."""
    vals = [t / c for t, c, f, v in zip(track, cmd, field_tile, valid if valid is not None else [True] * len(track))
            if f and v and c and c > 0]
    if not vals:
        return {"ok": False, "why": "no valid field-tile episode in the dump"}
    r = statistics.median(vals)
    return dict(ok=True, n=len(vals), **promote_from_rbar(r))


def calibrate_series(series, tail: int = CAL_TAIL) -> dict:
    """CHARTER 5 r-bar from training telemetry (CHARTER 9 operational definition): the mean of
    ``track_ratio_obst_p50`` over the last ``tail`` logged iterations. ``series`` = (steps, values).
    Non-finite points are dropped BEFORE the tail is taken; fewer than ``tail`` finite points fails closed
    (the smoke is >= 300 iterations, so a short series means a missing or late-starting tag)."""
    if series is None or not len(series[0]):
        return {"ok": False, "why": f"no {CAL_TAG} in the run's tfevents"}
    pts = sorted((int(st), float(v)) for st, v in zip(series[0], series[1]) if math.isfinite(float(v)))
    if len(pts) < tail:
        return {"ok": False, "why": f"{CAL_TAG}: {len(pts)} finite logged iterations < {tail}"}
    win = pts[-tail:]
    r = sum(v for _, v in win) / tail
    return dict(ok=True, n=tail, first_step=win[0][0], last_step=win[-1][0], source=CAL_TAG, **promote_from_rbar(r))


def calibrate_run(run_dir: Path) -> dict:
    """:func:`calibrate_series` on a run directory's tfevents (or the directory of a tfevents file)."""
    d = Path(run_dir)
    if d.is_file():
        d = d.parent
    ser = X().V.tb_series(d, {"r": CAL_TAG})
    return dict(calibrate_series(ser.get("r")), run_dir=str(d))


def calibrate_file(path: Path) -> dict:
    """A per-episode dump (.json mapping or .npz) with ep_track_dist + ep_cmd_dist, OR the per-episode
    ``track_ratio`` the parkour term hands its exposure ledger (= tracking_ratio(ep_track_dist, ep_cmd_dist),
    omni_parkour_event.py), and a field-tile flag (field_tile | is_field_tile | is_obst_tile), optionally a validity
    mask (valid | track_ratio_valid)."""
    if path.suffix == ".npz":
        import numpy as np  # noqa: PLC0415
        d = {k: np.asarray(v).tolist() for k, v in np.load(path).items()}
    else:
        d = json.load(open(path))
    tile = next((d[k] for k in ("field_tile", "is_field_tile", "is_obst_tile") if k in d), None)
    valid = next((d[k] for k in ("valid", "track_ratio_valid") if k in d), None)
    if tile is not None and "ep_track_dist" in d and "ep_cmd_dist" in d:
        return calibrate(d["ep_track_dist"], d["ep_cmd_dist"], tile, valid)
    if tile is not None and "track_ratio" in d:
        r = calibrate(d["track_ratio"], [1.0] * len(d["track_ratio"]), tile, valid)
        return dict(r, source="track_ratio")
    return {"ok": False, "why": f"dump lacks ep_track_dist + ep_cmd_dist (or track_ratio) / a field-tile flag: "
                                f"keys {sorted(d)}"}


# ================================================================================================ pre-launch gates
# Pre-launch audit 2026-10-06: B1 (no anchor eval before the CHARTER 5 smokes S1-S5 and the fork smoke pass: the
# queue requires $OUT/SMOKES_OK) and M2 (no A1 s18 launch before a passed CAL2 at A1's PROMOTE values).
SMOKES_DIR = CAMPAIGN / "smokes"
SMOKES_OK = "SMOKES_OK"
SMOKE_CHECKS = ("S1", "S2", "S3", "S4")
S4_KEYS = ("reach_edge_frac", "field_frac_mean", "block_steps_mean", "off_tile_block_steps_mean")
S5_EPISODES = 7
#: the wall-mode S5 (CHARTER 9, 2026-10-06 walls re-tune; the ramp-mode S5 was SMOKE_S5_s18)
S5_NAME = "SMOKE_S5W_s18"
TOE_MAX = 1.6
#: CAL1 / CAL2 = the wall-mode calibration smokes (CHARTER 9, 2026-10-06 walls re-tune). The ramp-mode ones were
#: P2__CAL1b_s18 (a fresh re-run: P2__CAL1_s18 was killed twice) and P2__CAL2_s18.
CAL1, CAL2, A1_GATE = "P2__CALW1_s18", "P2__CALW2_s18", "P2__A1_s18"
DIFF_KEY, EDGES_KEY = "KRABBY_FLAT_TERRAIN_DIFF", "KRABBY_FIELD_EDGES"
#: the 2a preset's values when an overlay does not set them
DIFF_DEFAULT, EDGES_DEFAULT = "0.20:0.70", "ramp"
PROMOTE_KEY = "KRABBY_TERRAIN_PROMOTE"
CODE_KEYS = ("code_sha", "eval_sha", "dirty")


def _placeholder(v) -> bool:
    import re  # noqa: PLC0415
    return bool(re.match(r"(?i)^\s*TBD", str(v))) or "PENDING" in str(v).upper()


def promote_pair(text) -> tuple[float, float] | None:
    try:
        up, down = (float(x) for x in str(text).split(":"))
    except (TypeError, ValueError):
        return None
    return (up, down) if all(math.isfinite(v) for v in (up, down)) else None


def chain_overlay_value(cands: dict, chain: str, key: str):
    for st in reversed(((cands.get("chains") or {}).get(chain) or {}).get("stages") or []):
        ov = st.get("overlay") or {}
        if key in ov:
            return ov[key]
    return None


def a1_terrain(cands: dict | None) -> dict | None:
    """The terrain the A1 gate arm trains on: ``{"difficulty_range": [lo, hi], "wall": bool}`` from its overlay (the
    2a preset's 0.20:0.70 / ramp when unset). None without a candidates file."""
    if not cands:
        return None
    diff = chain_overlay_value(cands, A1_GATE, DIFF_KEY) or DIFF_DEFAULT
    edges = chain_overlay_value(cands, A1_GATE, EDGES_KEY) or EDGES_DEFAULT
    return {"difficulty_range": [float(x) for x in str(diff).split(":")], "wall": str(edges).strip().lower() == "wall"}


def smoked_terrain(doc: dict | None) -> dict | None:
    """The terrain a p2_smoke.py run actually built (its S1 summary), in :func:`a1_terrain`'s shape; None if unread or
    the field sub-terrains disagree on their edges."""
    s1 = (doc or {}).get("s1_summary") or {}
    edges = {v.get("vertical_edges") for k, v in (s1.get("sub_terrains") or {}).items() if str(k).startswith("omni_field")}
    dr = s1.get("difficulty_range")
    if len(edges) != 1 or not isinstance(dr, (list, tuple)) or len(dr) != 2:
        return None
    wall = edges.pop()
    return {"difficulty_range": [float(x) for x in dr], "wall": wall} if isinstance(wall, bool) else None


def terrain_fails(have: dict | None, want: dict | None, what: str) -> list[str]:
    """CHARTER 9 (2026-10-06 walls re-tune): smokes count only for the terrain the arms train on."""
    if want is None:
        return [f"{what}: the A1 terrain is unread (no candidates)"]
    if not have:
        return [f"{what}: smoked terrain unknown (a smoke from before the walls re-tune?)"]
    out = []
    if have.get("wall") is not want["wall"]:
        out.append(f"{what}: smoked edges {'wall' if have.get('wall') else 'ramp'} != A1's "
                   f"{'wall' if want['wall'] else 'ramp'}")
    hd, wd = have.get("difficulty_range") or [], want["difficulty_range"]
    if len(hd) != 2 or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(hd, wd)):
        out.append(f"{what}: smoked difficulty_range {hd} != A1's {wd}")
    return out


def audit_of(rec: dict | None) -> dict | None:
    """The CHARTER 5 dump audit a run_arm.sh smoke record carries (its p2_live.py --smoke-audit JSON line)."""
    raw = (rec or {}).get("smoke_audit")
    if isinstance(raw, dict):
        return raw
    try:
        d = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def smoke_record_fails(rec: dict | None, chain: str) -> list[str]:
    if not rec:
        return [f"no {chain} record"]
    fails = []
    if rec.get("chain") != chain:
        fails.append(f"record chain {rec.get('chain')!r} != {chain}")
    if not rec.get("smoke"):
        fails.append("record is not a calibration smoke")
    if rec.get("killed"):
        fails.append(f"killed: {rec.get('killed')}")
    au = audit_of(rec)
    if not au or au.get("result") != "PASS":
        fails.append(f"dump audit {(au or {}).get('result')!r} (fails {(au or {}).get('fails')})")
    return fails


def cal2_gate(rec: dict | None, cands: dict) -> dict:
    """Pre-launch audit M2 (pure): CAL2 passed its dump audit AT the A1 chain's PROMOTE values."""
    fails = smoke_record_fails(rec, CAL2)
    au = audit_of(rec) or {}
    a1 = chain_overlay_value(cands, A1_GATE, PROMOTE_KEY)
    c2 = chain_overlay_value(cands, CAL2, PROMOTE_KEY)
    pa1, pc2, pau = promote_pair(a1), promote_pair(c2), promote_pair(au.get("promote"))
    if a1 is None or _placeholder(a1) or pa1 is None:
        fails.append(f"A1 {PROMOTE_KEY} not fixed: {a1!r}")
    if pau is None:
        fails.append(f"CAL2 audit PROMOTE unreadable: {au.get('promote')!r}")
    elif pa1 is not None and any(abs(x - y) > 1e-9 for x, y in zip(pau, pa1)):
        fails.append(f"CAL2 audited PROMOTE {au.get('promote')} != A1 {a1}")
    if pc2 is None or (pa1 is not None and any(abs(x - y) > 1e-9 for x, y in zip(pc2, pa1))):
        fails.append(f"CAL2 chain PROMOTE {c2!r} != A1 {a1!r} in candidates.yaml")
    return {"ok": not fails, "fails": fails, "a1_promote": a1, "cal2_promote": c2, "cal2_audit_promote": au.get("promote"),
            "cal2_levels_moved": au.get("levels_moved")}


def resolve_stamp(arg: str, row: str, seed: int = 1, lrout: Path | None = None) -> str | None:
    p = Path(arg)
    if stamp_complete(p):
        return str(p)
    return newest_stamp(arg, row, seed, lrout=lrout or LROUT) if "/" not in arg else None


def smoke_s5_read(stamp: str | None, want_sha: str | None) -> dict:
    """CHARTER 5 S5 (pure IO on one stamp): field_light__omni, 7 episodes, the s18 head, zero stand-hold falls."""
    if not stamp:
        return {"ok": False, "fails": ["no S5 stamp"], "stamp": None}
    import numpy as np  # noqa: PLC0415

    st = Path(stamp)
    meta = _json(st / "run_meta.json") or {}
    sm = _json(st / "scenario_metrics.json") or {}
    fails = []
    if meta.get("scenario_id") != FL:
        fails.append(f"scenario {meta.get('scenario_id')!r} != {FL}")
    eps = sm.get("episodes") or []
    n = int((sm.get("aggregate") or {}).get("n_episodes") or len(eps))
    if n != S5_EPISODES or len(eps) != S5_EPISODES:
        fails.append(f"{n} episodes ({len(eps)} listed), want {S5_EPISODES}")
    if want_sha and meta.get("checkpoint_sha256") != want_sha:
        fails.append(f"checkpoint sha256 {str(meta.get('checkpoint_sha256'))[:12]} != the s18 head {want_sha[:12]}")
    labels = list(meta.get("hold_labels") or [])
    if "stand" not in labels:
        fails.append("no stand hold in the row")
    stand_idx = labels.index("stand") if "stand" in labels else None
    stand_falls, other_falls, unread = [], [], []
    for e in eps:
        if str(e.get("termination_reason")) != "fall":
            continue
        idx = int(e.get("env_index"))
        f = st / "raw" / f"episode_{idx:02d}.npz"
        try:
            seg = int(np.load(f)["segment_id"][-1])
        except (OSError, KeyError, ValueError, IndexError):
            unread.append(idx)
            continue
        (stand_falls if seg in (-1, stand_idx) else other_falls).append(idx)
    if unread:
        fails.append(f"fall episodes without a readable raw segment (fail closed): {unread}")
    if stand_falls:
        fails.append(f"stand-hold falls (prologue or stand): {stand_falls}")
    val = stamp_validity(stamp) or {}
    if not val.get("sidecar"):
        fails.append("no code-state sidecar")
    return {"ok": not fails, "fails": fails, "stamp": str(st), "n_episodes": n, "stand_falls": len(stand_falls),
            "other_falls": len(other_falls), "sidecar": val.get("sidecar"), "env4": val.get("env4")}


def smokes_doc_fails(doc: dict | None) -> list[str]:
    """What keeps a p2_smoke.py smokes.json from counting: S1-S4 PASS, finished, no errors, clean code, the S3 toe
    radius and the four S4 numbers present."""
    if not doc:
        return ["no smokes.json"]
    fails = []
    res = doc.get("results") or {}
    for c in SMOKE_CHECKS:
        if (res.get(c) or {}).get("result") != "PASS":
            fails.append(f"{c} {(res.get(c) or {}).get('result', 'not run')}")
    if not doc.get("finished") or doc.get("errors"):
        fails.append(f"smoke run unfinished or errored: {doc.get('errors')}")
    plan = doc.get("plan") or {}
    if plan.get("reference") != A1_GATE:
        fails.append(f"smoked reference {plan.get('reference')!r} != {A1_GATE}")
    cs = plan.get("code_state") or {}
    if cs.get("dirty") is not False:
        fails.append(f"smoked on a dirty or unknown tree: {cs.get('dirty_paths')}")
    toe = (res.get("S3") or {}).get("toe_radius_max_m")
    if not isinstance(toe, (int, float)) or not toe <= TOE_MAX + 1e-9:
        fails.append(f"S3 toe radius {toe!r} (want <= {TOE_MAX} m)")
    for k in S4_KEYS:
        if not isinstance((res.get("S4") or {}).get(k), (int, float)):
            fails.append(f"S4 {k} missing")
    return fails


def code_state_fails(have: dict | None, live: dict | None, what: str) -> list[str]:
    if not have:
        return [f"{what}: no code state"]
    if not live or live.get("dirty") is not False:
        return [f"live code state dirty or unreadable: {(live or {}).get('dirty_paths')}"]
    return [f"{what} {k} {have.get(k)!r} != live {live.get(k)!r}" for k in CODE_KEYS
            if (bool(have.get(k)) if k == "dirty" else have.get(k)) != (bool(live.get(k)) if k == "dirty" else live.get(k))]


def build_smokes_ok(smokes: Path, s5: dict, cal1: dict | None, live: dict | None,
                    cands: dict | None = None) -> tuple[list[str], dict]:
    doc = _json(Path(smokes) / "smokes.json")
    fails = smokes_doc_fails(doc)
    want = a1_terrain(cands)
    terrain = smoked_terrain(doc)
    fails += terrain_fails(terrain, want, "smokes")
    s5_env = (stamp_validity(s5.get("stamp")) or {}).get("manifest_env") or {}
    if want is not None and (str(s5_env.get(EDGES_KEY, EDGES_DEFAULT)).strip().lower() == "wall") is not want["wall"]:
        fails.append(f"S5: row edges {s5_env.get(EDGES_KEY)!r} != A1's {'wall' if want['wall'] else 'ramp'}")
    fails += [f"S5: {f}" for f in s5.get("fails") or []] if not s5.get("ok") else []
    fails += [f"CAL1 fork smoke: {f}" for f in smoke_record_fails(cal1, CAL1)]
    cs = ((doc or {}).get("plan") or {}).get("code_state")
    fails += code_state_fails(cs, live, "smokes")
    sc = s5.get("sidecar")
    fails += code_state_fails(dict(zip(CODE_KEYS, sc)) if sc else None, live, "S5 sidecar")
    res = (doc or {}).get("results") or {}
    payload = {"result": "FAIL" if fails else "PASS", "fails": fails,
               "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "smokes": str(smokes),
               "s3_toe_radius_max_m": (res.get("S3") or {}).get("toe_radius_max_m"),
               "s3_resets": (res.get("S3") or {}).get("resets"),
               "s4": {k: (res.get("S4") or {}).get(k) for k in S4_KEYS},
               "s4_toe_radius_max_walking_m": (res.get("S4") or {}).get("toe_radius_max_walking_m"),
               "s5": {k: s5.get(k) for k in ("stamp", "n_episodes", "stand_falls", "other_falls")},
               "cal1": {"run_dir": (cal1 or {}).get("run_dir"), "audit": (audit_of(cal1) or {}).get("result")},
               "code_state": {k: (live or {}).get(k) for k in CODE_KEYS},
               "smoked_substitutions": ((doc or {}).get("plan") or {}).get("substituted"),
               "terrain": terrain}
    return fails, payload


def smokes_ok_fails(payload: dict | None, live: dict | None, cands: dict | None = None) -> list[str]:
    if not payload:
        return ["no SMOKES_OK"]
    fails = [] if payload.get("result") == "PASS" and not payload.get("fails") else [f"SMOKES_OK says {payload.get('result')}"]
    toe = payload.get("s3_toe_radius_max_m")
    if not isinstance(toe, (int, float)) or toe > TOE_MAX + 1e-9:
        fails.append(f"S3 toe radius {toe!r}")
    if any(not isinstance((payload.get("s4") or {}).get(k), (int, float)) for k in S4_KEYS):
        fails.append("S4 numbers missing")
    if (payload.get("s5") or {}).get("stand_falls") != 0:
        fails.append("S5 stand falls not 0")
    fails += code_state_fails(payload.get("code_state"), live, "SMOKES_OK")
    fails += terrain_fails(payload.get("terrain"), a1_terrain(cands), "SMOKES_OK")
    return fails


def charter_line(p: dict) -> str:
    s4 = p.get("s4") or {}
    f = lambda v, d=3: "NA" if not isinstance(v, (int, float)) else f"{v:.{d}f}"  # noqa: E731
    cs = p.get("code_state") or {}
    return (f"- {datetime.date.today().isoformat()}, **CHARTER 5 smokes passed; `SMOKES_OK` written before any anchor** "
            f"(`scripts/p2_read.py --write-smokes-ok`).\n"
            f"  - S1-S4: `{Path(p['smokes']).relative_to(CAMPAIGN) if str(p['smokes']).startswith(str(CAMPAIGN)) else p['smokes']}`.\n"
            f"  - S3: max toe radius {f(p.get('s3_toe_radius_max_m'))} m (limit {TOE_MAX} m), over {p.get('s3_resets')} "
            f"resets; walking max (S4) {f(p.get('s4_toe_radius_max_walking_m'))} m.\n"
            f"  - S4: reach_edge {f(s4.get('reach_edge_frac'))}, field_frac {f(s4.get('field_frac_mean'))}, block steps per "
            f"obstacle-tile episode {f(s4.get('block_steps_mean'), 1)}, off-tile block steps "
            f"{f(s4.get('off_tile_block_steps_mean'), 2)}.\n"
            f"  - S5: `{(p.get('s5') or {}).get('stamp')}`, {(p.get('s5') or {}).get('n_episodes')} episodes, "
            f"{(p.get('s5') or {}).get('stand_falls')} stand-hold falls.\n"
            f"  - CAL1 dump audit {(p.get('cal1') or {}).get('audit')}. code_sha {str(cs.get('code_sha'))[:8]}…, "
            f"eval_sha {str(cs.get('eval_sha'))[:8]}….\n")


def _live_code_state(code_root: Path = R) -> dict | None:
    try:
        return _load(HERE / "p2_live.py", "_p2_live_for_read").live_code_state(code_root)
    except Exception as e:  # noqa: BLE001 -- unreadable is a FAIL, never a pass
        return {"dirty": None, "dirty_paths": f"unreadable: {e.__class__.__name__}: {e}"}


# ================================================================================================ loading
def _anchors_yaml() -> dict:
    import yaml  # noqa: PLC0415
    return yaml.safe_load(open(CAMPAIGN / "anchors.yaml"))["anchors"]


def _manifest_s18() -> dict | None:
    try:
        import yaml  # noqa: PLC0415
        m = yaml.safe_load(open(POLICY_MANIFEST))
    except (OSError, ValueError):
        return None
    for st in (m or {}).get("stages") or (m or {}).get("phases") or []:
        if isinstance(st, dict) and str(st.get("sha256", "")).startswith("b40f9a19"):
            return ((st.get("evals") or {}).get("categories")) or None
    return _find_s18(m)


def _find_s18(node):
    if isinstance(node, dict):
        if str(node.get("sha256", "")).startswith("b40f9a19") and isinstance(node.get("evals"), dict):
            return node["evals"].get("categories")
        for v in node.values():
            r = _find_s18(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_s18(v)
            if r:
                return r
    return None


def _rec(ref: str) -> dict | None:
    return _json(ARMS / f"{ref}.record.json")


def load_all(with_tb: bool = True, with_raw: bool = False, arms=("A1", "A0", "S1", "E1", "F1")) -> dict:
    an = _anchors_yaml()
    anc = {int(s): load_point(a["name"], with_tb=False, with_raw=with_raw) for s, a in an.items()}
    heads = {}
    for arm in arms:
        for s in (SEEDS if arm == "A1" else (GATE_SEED,)):
            ref = f"P2__{arm}_s{s}"
            rec = _rec(ref)
            if rec and not rec.get("killed"):
                heads[ref] = load_point(ref, suite_stamp=rec.get("driver_stamp"), run_dir=rec.get("run_dir"),
                                        with_tb=with_tb, with_raw=with_raw)
            else:
                heads[ref] = {"name": ref, "missing": True, "killed": (rec or {}).get("killed")}
    return {"anchors": anc, "heads": heads}


def verdicts(res: dict) -> dict:
    anc = res["anchors"]
    out = {}
    for ref, pt in res["heads"].items():
        s = int(ref.rsplit("_s", 1)[1])
        if pt.get("missing"):
            out[ref] = {"name": ref, "verdict": "PENDING", "why": [f"no record ({pt.get('killed') or 'not run'})"],
                        "gain": None, "cost": []}
            continue
        out[ref] = seed_verdict(pt, anc[s], bands(anc[s], anc[GATE_SEED]))
    return out


# ================================================================================================ self-test
def _sc():
    return ("c" * 64, "e" * 64, False)


def _valid(manifest_env=None, sidecar="default", env4=True):
    return {"env4": env4, "env4_missing": [] if env4 else ["KRABBY_OMNI_CLOCK_HIDE"],
            "manifest_env": manifest_env or {}, "sidecar": _sc() if sidecar == "default" else sidecar}


def _pt(name, fh=(0.50, 0.50), fl=(0.80, 0.80), ro=(0.60, 0.60), err=0.03, fc=0.80, yp=0.10, mt=0.90, jit=0.20,
        hands=("FAST", "FAST"), fall=0.0, dose=True, trans=1.0, feet=0.23, track=0.80, s2=None, **kw):
    def c(r):
        return {"rate": r, "n": 100, "falls": int(round((1 - r) * 100)), "track": track}
    suite = {"err_fwd": err, "err_back": err, "err_side": err, "err_diag": err, "turn_fc": fc, "turn_yp": yp,
             "mt_surv": mt, "trans_surv": {h: trans for h in TRANSLATE_HOLDS}, "jitter": jit,
             "stand": {"tripod": kw.pop("tripod", 0.79), "steps_per_s": 3.2, "six_down_frac": 0.002}}
    pt = {"name": name, "stamps": {k: "/x" for k in NEEDED_STAMPS},
          "valid": {k: _valid() for k in NEEDED_STAMPS},
          "comp": {"field_hard": {1: c(fh[0]), 2: c(fh[1])}, "field_light": {1: c(fl[0]), 2: c(fl[1])},
                   "ring_obst": {1: c(ro[0]), 2: c(ro[1])}},
          "suite": suite, "suite2": s2, "hands": {"l": hands[0], "r": hands[1]}, "fall": fall, "feet_side": feet,
          "dose_ok": dose, "field_suite": {"mt_surv": mt}}
    pt.update(kw)
    return pt


def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    s2 = {"err_fwd": 0.03, "err_back": 0.03, "err_side": 0.03, "err_diag": 0.03, "turn_fc": 0.80, "turn_yp": 0.10,
          "mt_surv": 0.90}
    anc = _pt("ANC_s18", s2=s2)
    bd = bands(anc, anc)
    check("bands: identical seeds take the floors (completion = max(SE, 0.04))",
          (round(bd["field_hard"], 4), bd["err_fwd"], bd["turn_yp"], round(bd["mt_surv"], 4)),
          (0.04, 0.02, 0.005, 0.04))
    wide = _pt("ANC_s16", fh=(0.40, 0.60))
    check("bands: |seed1 - seed2| 0.20 sets the field_hard band", round(bands(wide, anc)["field_hard"], 3), 0.2)
    check("bands: the binomial SE floors a near-0.5 pooled p at n=200 (0.0354 < 0.04 -> 0.04)",
          round(max(binom_se(0.5, 200), 0.04), 4), 0.04)
    check("bands: a missing seed-2 suite -> None (PENDING)", bands(anc, _pt("x"))["err_fwd"], None)

    good = _pt("P2__A1_s18", fh=(0.62, 0.62))
    check("ladder: +0.12 field_hard, no loss -> PASS", seed_verdict(good, anc, bd)["verdict"], "PASS")
    check("ladder: gain exactly anchor + 0.10 -> PASS (>= boundary)",
          seed_verdict(_pt("a", fh=(0.60, 0.60)), anc, bd)["verdict"], "PASS")
    check("ladder: gain +0.09 -> NO GAIN", seed_verdict(_pt("a", fh=(0.59, 0.59)), anc, bd)["verdict"], "NO GAIN")
    check("ladder: gain needs 2 x band when wider (band 0.2 -> +0.40)",
          seed_verdict(_pt("a", fh=(0.85, 0.85)), wide, bands(wide, anc))["verdict"], "NO GAIN")
    check("ladder: field_light 0.04 below -> NO GAIN (hold-the-line 0.04 band... floor max(0.03, 0.04))",
          seed_verdict(_pt("a", fh=(0.62, 0.62), fl=(0.755, 0.755)), anc, bd)["verdict"], "NO GAIN")
    check("ladder: field_light exactly anchor - 0.04 -> PASS (boundary)",
          seed_verdict(_pt("a", fh=(0.62, 0.62), fl=(0.76, 0.76)), anc, bd)["verdict"], "PASS")
    check("ladder: ring_obst 0.06 below -> NO GAIN", seed_verdict(_pt("a", fh=(0.62, 0.62), ro=(0.54, 0.54)), anc,
                                                                  bd)["verdict"], "NO GAIN")
    check("ladder: a missing sidecar -> INVALID EVAL (never equal)",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FH}@2": _valid(sidecar=None)})), anc, bd)["verdict"],
          "INVALID EVAL")
    check("ladder: a different code_sha sidecar -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{"suite@1": _valid(sidecar=("d" * 64, "e" * 64, False))})),
                       anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a dirty sidecar vs a clean anchor -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{"lr@1": _valid(sidecar=("c" * 64, "e" * 64, True))})),
                       anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a stamp without CLOCK_HIDE=1 -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{"suite@1": _valid(env4=False)})), anc, bd)["verdict"],
          "INVALID EVAL")
    check("ladder: a row env block differing from the anchor's -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FH}@1": _valid({"KRABBY_FIELD_PAD_R": "1.8"})})),
                       anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a missing env_seed-2 field_hard -> PENDING",
          seed_verdict(dict(good, comp=dict(good["comp"], field_hard={1: good["comp"]["field_hard"][1], 2: None})),
                       anc, bd)["verdict"], "PENDING")
    check("ladder: a NA hand -> PENDING, never FAST", seed_verdict(_pt("a", fh=(0.62, 0.62), hands=("FAST", "NA")),
                                                                  anc, bd)["verdict"], "PENDING")
    check("ladder: a NA metric -> PENDING", seed_verdict(dict(good, suite=dict(good["suite"], turn_yp=None)), anc,
                                                         bd)["verdict"], "PENDING")
    check("ladder: a missing dose read -> PENDING", seed_verdict(_pt("a", fh=(0.62, 0.62), dose=None), anc, bd)["verdict"],
          "PENDING")
    check("ladder: a missing band -> PENDING", seed_verdict(good, anc, dict(bd, ring_obst=None))["verdict"], "PENDING")
    check("ladder: INVALID outranks PENDING", seed_verdict(dict(_pt("a", dose=None), valid=dict(good["valid"], **{
        "lr@1": _valid(sidecar=None)})), anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a SLOW side hand -> HAND LOSS", seed_verdict(_pt("a", fh=(0.62, 0.62), hands=("FAST", "SLOW")), anc,
                                                               bd)["verdict"], "HAND LOSS")
    check("ladder: hold-fall 0.16 -> HAND LOSS", seed_verdict(_pt("a", fh=(0.62, 0.62), fall=0.16), anc, bd)["verdict"],
          "HAND LOSS")
    check("ladder: hold-fall exactly 0.15 is not HAND LOSS", seed_verdict(_pt("a", fh=(0.62, 0.62), fall=0.15), anc,
                                                                          bd)["verdict"], "PASS")
    check("ladder: dose lost -> DOSE", seed_verdict(_pt("a", fh=(0.62, 0.62), dose=False), anc, bd)["verdict"], "DOSE")
    check("ladder: HAND LOSS outranks DOSE", seed_verdict(_pt("a", hands=("SLOW", "FAST"), dose=False), anc, bd)["verdict"],
          "HAND LOSS")
    check("ladder: |ratio-1| 0.11 worse (> 0.10 floor) -> FLAT FAIL",
          seed_verdict(_pt("a", fh=(0.62, 0.62), err=0.14 + 1e-6), anc, bd)["verdict"], "FLAT FAIL")
    check("ladder: |ratio-1| exactly 0.10 worse -> not FLAT FAIL (COST: > max(0.03, band))",
          seed_verdict(_pt("a", fh=(0.62, 0.62), err=0.13), anc, bd)["verdict"], "COST")
    check("ladder: yaw_progress 0.031 lower (floor 0.03) -> FLAT FAIL",
          seed_verdict(_pt("a", fh=(0.62, 0.62), yp=0.069), anc, bd)["verdict"], "FLAT FAIL")
    check("ladder: yaw_progress 0.02 lower -> COST (margin 0.01)", seed_verdict(_pt("a", fh=(0.62, 0.62), yp=0.08), anc,
                                                                                bd)["verdict"], "COST")
    check("ladder: overshoot (ratio 1.12 vs 1.03) counts as worse tracking error",
          seed_verdict(_pt("a", fh=(0.62, 0.62), err=0.12), anc, bd)["verdict"], "COST")
    check("ladder: a translate hold at 0.94 -> FLAT FAIL", seed_verdict(_pt("a", fh=(0.62, 0.62), trans=0.94), anc,
                                                                        bd)["verdict"], "FLAT FAIL")
    check("ladder: a translate hold at exactly 0.95 passes", seed_verdict(_pt("a", fh=(0.62, 0.62), trans=0.95), anc,
                                                                          bd)["verdict"], "PASS")
    check("ladder: FLAT FAIL outranks NO GAIN", seed_verdict(_pt("a", trans=0.90), anc, bd)["verdict"], "FLAT FAIL")
    check("ladder: move_turn survival 0.06 lower -> COST (margin 0.05)",
          seed_verdict(_pt("a", fh=(0.62, 0.62), mt=0.84), anc, bd)["verdict"], "COST")
    check("ladder: move_turn survival 0.05 lower -> PASS (boundary)",
          seed_verdict(_pt("a", fh=(0.62, 0.62), mt=0.85), anc, bd)["verdict"], "PASS")

    def V(v, gain, cost=()):
        return {"verdict": v, "gain": gain, "cost": list(cost)}
    check("recipe: 3 PASS -> ADOPTABLE", recipe_read({s: V("PASS", True) for s in SEEDS})["reading"], "ADOPTABLE")
    check("recipe: 2 PASS + 1 HAND LOSS -> not ADOPTABLE (MIXED)",
          recipe_read({16: V("PASS", True), 17: V("PASS", True), 18: V("HAND LOSS", True)})["reading"], "MIXED")
    check("recipe: 2 COST on the same metric -> TRADE-OFF",
          recipe_read({16: V("COST", True, ["turn_yp"]), 17: V("COST", True, ["turn_yp", "err_side"]),
                       18: V("PASS", True)})["reading"], "TRADE-OFF")
    check("recipe: 2 COST on different metrics -> MIXED",
          recipe_read({16: V("COST", True, ["turn_yp"]), 17: V("COST", True, ["err_side"]), 18: V("PASS", True)})["reading"],
          "MIXED")
    check("recipe: 1/3 gain -> NOT ADOPTABLE",
          recipe_read({16: V("PASS", True), 17: V("NO GAIN", False), 18: V("NO GAIN", False)})["reading"], "NOT ADOPTABLE")
    check("recipe: 2/3 HAND LOSS -> NOT ADOPTABLE",
          recipe_read({16: V("HAND LOSS", True), 17: V("HAND LOSS", True), 18: V("PASS", True)})["reading"], "NOT ADOPTABLE")
    check("recipe: a PENDING seed -> PENDING", recipe_read({16: V("PASS", True), 17: V("PENDING", None),
                                                            18: V("PASS", True)})["reading"], "PENDING")
    pts = {s: _pt(f"a{s}", fh=(0.70, 0.70)) for s in SEEDS}
    check("p2b: gain 3/3 and field_hard < 0.85 3/3 -> trigger", p2b_trigger({s: V("PASS", True) for s in SEEDS}, pts)["trigger"],
          True)
    pts_hi = {s: _pt(f"a{s}", fh=(0.90, 0.90), track=0.9) for s in SEEDS}
    check("p2b: no deficit -> no trigger", p2b_trigger({s: V("PASS", True) for s in SEEDS}, pts_hi)["trigger"], False)
    pts_tr = {s: _pt(f"a{s}", fh=(0.90, 0.90), track=0.65) for s in SEEDS}
    check("p2b: walking-hold tracking 0.65 < 0.7 is a deficit", p2b_trigger({s: V("PASS", True) for s in SEEDS},
                                                                            pts_tr)["trigger"], True)
    check("p2b: gain 1/3 -> no trigger", p2b_trigger({16: V("PASS", True), 17: V("NO GAIN", False),
                                                      18: V("NO GAIN", False)}, pts)["trigger"], False)

    a1 = _pt("P2__A1_s18", fh=(0.62, 0.62), jit=0.20, feet=0.23, power=100.0, stand_clearance=0.08, tripod=0.79)
    s1 = _pt("P2__S1_s18", fh=(0.62, 0.62), jit=0.14, feet=0.22, power=90.0, stand_clearance=0.08)
    check("sibling S1: jitter -30 %, feet -0.01, nothing lost -> KEEP", sibling_rec("S1", s1, a1, bd)["rec"], "KEEP")
    check("sibling S1: jitter -20 % -> DO NOT KEEP", sibling_rec("S1", dict(s1, suite=dict(s1["suite"], jitter=0.16)),
                                                                 a1, bd)["rec"], "DO NOT KEEP")
    check("sibling S1: jitter exactly -25 % -> KEEP (boundary)",
          sibling_rec("S1", dict(s1, suite=dict(s1["suite"], jitter=0.15)), a1, bd)["rec"], "KEEP")
    check("sibling S1: side feet -0.03 -> DO NOT KEEP", sibling_rec("S1", dict(s1, feet_side=0.20), a1, bd)["rec"],
          "DO NOT KEEP")
    check("sibling S1: a SLOW hand -> DO NOT KEEP", sibling_rec("S1", dict(s1, hands={"l": "FAST", "r": "SLOW"}), a1,
                                                                bd)["rec"], "DO NOT KEEP")
    check("sibling S1: field_hard 0.10 lower -> DO NOT KEEP (nothing loses)",
          sibling_rec("S1", _pt("P2__S1_s18", fh=(0.52, 0.52), jit=0.14), a1, bd)["rec"], "DO NOT KEEP")
    check("sibling S1: a NA hand -> PENDING", sibling_rec("S1", dict(s1, hands={"l": "NA", "r": "FAST"}), a1, bd)["rec"],
          "PENDING")
    check("sibling S1: a missing sidecar -> INVALID EVAL",
          sibling_rec("S1", dict(s1, valid=dict(s1["valid"], **{"suite@1": _valid(sidecar=None)})), a1, bd)["rec"],
          "INVALID EVAL")
    check("sibling E1: power down + S1's conditions -> KEEP", sibling_rec("E1", s1, a1, bd)["rec"], "KEEP")
    check("sibling E1: power not down -> DO NOT KEEP", sibling_rec("E1", dict(s1, power=101.0), a1, bd)["rec"],
          "DO NOT KEEP")
    check("sibling E1: power unread -> PENDING", sibling_rec("E1", dict(s1, power=None), a1, bd)["rec"], "PENDING")
    f1 = _pt("P2__F1_s18", fh=(0.62, 0.62), stand_clearance=0.035, tripod=0.04)
    check("sibling F1: A1 marches, F1 taps low -> KEEP", sibling_rec("F1", f1, a1, bd)["rec"], "KEEP")
    check("sibling F1: clearance exactly 0.05 / tripod 0.4 -> KEEP (boundary)",
          sibling_rec("F1", _pt("P2__F1_s18", fh=(0.62, 0.62), stand_clearance=0.05, tripod=0.4), a1, bd)["rec"], "KEEP")
    check("sibling F1: F1 still marches -> DO NOT KEEP",
          sibling_rec("F1", _pt("P2__F1_s18", fh=(0.62, 0.62), stand_clearance=0.08, tripod=0.7), a1, bd)["rec"],
          "DO NOT KEEP")
    check("sibling F1: A1 already taps -> NOT RECOMMENDED",
          sibling_rec("F1", f1, dict(a1, stand_clearance=0.04), bd)["rec"], "NOT RECOMMENDED")
    check("sibling F1: no clearance read -> PENDING", sibling_rec("F1", dict(f1, stand_clearance=None), a1, bd)["rec"],
          "PENDING")

    man = {"stand": {"survival": 1.0, "n": 100}, "fwd": {"survival": 1.0, "ratio": 1.041, "n": 100}}
    cats = {"stand": {"survival": 1.0, "ratio": None}, "fwd": {"survival": 1.0, "ratio": 1.0413}}
    A = {s: _pt(f"ANC_s{s}", fh=(0.45, 0.45), fl=(0.70, 0.70), suite=dict(_pt("x")["suite"], categories=cats))
         for s in SEEDS}
    check("anchors: in-window and discriminating, inert -> ok", anchors_check(A, man)["ok"], True)
    check("anchors: field_hard 0.90 -> stop", anchors_check({**A, **{16: _pt("x", fh=(0.9, 0.9), fl=(1.0, 1.0))}},
                                                            man)["ok"], False)
    check("anchors: field_hard exactly 0.85 is in the window",
          anchors_check({**A, **{16: _pt("x", fh=(0.85, 0.85), fl=(0.95, 0.95))}}, man)["ok"], True)
    check("anchors: field_light < field_hard + 0.10 -> stop",
          anchors_check({**A, **{17: _pt("x", fh=(0.45, 0.45), fl=(0.54, 0.54))}}, man)["ok"], False)
    check("anchors: the s18 suite off the manifest -> stop (not inert)",
          anchors_check({**A, **{18: dict(A[18], suite=dict(A[18]["suite"], categories=dict(
              cats, fwd={"survival": 1.0, "ratio": 1.043})))}}, man)["ok"], False)
    check("anchors: no manifest -> stop (fail closed)", anchors_check(A, None)["ok"], False)

    head = _pt("P2__A1_s18", fh=(0.50, 0.50))
    m_hi = _pt("m", fh=(0.65, 0.65))
    m_1 = dict(_pt("m1", fh=(0.65, 0.65)), comp=dict(_pt("m1")["comp"], field_hard={1: {"rate": 0.65, "n": 100,
                                                                                          "falls": 35, "track": 0.8}, 2: None}))
    ms = m_screen(head, {16000: _pt("m0", fh=(0.55, 0.55)), 17500: m_1, 19000: m_hi}, anc, bd)
    check("m-screen: +0.05 -> NO; +0.15 seed-1 only -> NEEDS_SEED2; +0.15 at n=200 -> REPLACES HEAD",
          [ms[16000]["status"], ms[17500]["status"], ms[19000]["status"]], ["NO", "NEEDS_SEED2", "REPLACES HEAD"])
    ms2 = m_screen(head, {19000: _pt("m", fh=(0.65, 0.65), hands=("FAST", "SLOW"))}, anc, bd)
    check("m-screen: a broken floor blocks the replacement", ms2[19000]["status"], "NO (n=200)")
    ri = rank_inputs({"s16": _pt("a", fh=(0.7, 0.7)), "s17": _pt("b", fh=(0.6, 0.6)), "s18": _pt("c", fh=(0.6, 0.6))},
                     {})
    check("rank inputs: obstacle rank-sum, ties averaged (fh 1/2.5/2.5 + three 3-way ties at 2)", [ri[k]["obstacle_rank_sum"] for k in ("s16", "s17", "s18")],
          [7.0, 8.5, 8.5])
    cal = calibrate([0.8, 0.9, 0.5, 0.7], [1.0, 1.0, 1.0, 1.0], [1, 1, 0, 1])
    check("calibrate: r-bar = median on field tiles (0.8) -> 0.7:0.4", (round(cal["r_bar"], 3), cal["promote"]),
          (0.8, "0.7:0.4"))
    check("calibrate: no field-tile episode -> not ok", calibrate([0.8], [1.0], [0])["ok"], False)
    check("calibrate: invalid episodes are excluded", calibrate([0.1, 0.8], [1.0, 1.0], [1, 1], [0, 1])["promote"],
          "0.7:0.4")
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "d.json"
        f.write_text(json.dumps({"track_ratio": [0.8, 0.9, 0.5, 0.7], "is_obst_tile": [1, 1, 0, 1],
                                 "track_ratio_valid": [1, 1, 1, 1]}))
        check("calibrate: a per-episode track_ratio dump is accepted", calibrate_file(f).get("promote"), "0.7:0.4")
        f.write_text(json.dumps({"is_obst_tile": [1]}))
        check("calibrate: a dump with no tracking key is refused", calibrate_file(f)["ok"], False)
    ser = ([14996 + i for i in range(300)], [0.6] * 200 + [0.8] * 100)
    cs = calibrate_series(ser)
    check("calibrate (tfevents): r-bar = mean of the last 100 logged its (0.8) -> 0.7:0.4",
          (round(cs["r_bar"], 6), cs["promote"], cs["first_step"]), (0.8, "0.7:0.4", 15196))
    check("calibrate (tfevents): fewer than 100 logged its fails closed",
          calibrate_series(([1, 2], [0.8, 0.8]))["ok"], False)
    check("calibrate (tfevents): no tag fails closed", calibrate_series(None)["ok"], False)
    nan_ser = ([i for i in range(150)], [0.6] * 40 + [float("nan")] * 10 + [0.9] * 100)
    check("calibrate (tfevents): NaN points are dropped, not averaged (0.9 -> 0.8:0.45)",
          calibrate_series(nan_ser)["promote"], "0.8:0.45")
    sha = "b" * 64
    A_sha = {s: dict(A[s], valid={k: dict(_valid(), ckpt_sha=sha) for k in NEEDED_STAMPS}) for s in SEEDS}
    check("anchors: every stamp's checkpoint sha matches anchors.yaml -> ok",
          anchors_check(A_sha, man, {s: sha for s in SEEDS})["ok"], True)
    check("anchors: one stamp of another checkpoint -> stop",
          anchors_check({**A_sha, 17: dict(A_sha[17], valid=dict(A_sha[17]["valid"], **{f"{FH}@2": dict(
              _valid(), ckpt_sha="c" * 64)}))}, man, {s: sha for s in SEEDS})["ok"], False)
    check("anchors: a stamp without a checkpoint sha -> stop (fail closed)",
          anchors_check(A, man, {s: sha for s in SEEDS})["ok"], False)
    check("validity: field_suite / field_turn stamps are compared (a dirty field_suite sidecar -> INVALID EVAL)",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FS}@1": _valid(sidecar=("c" * 64, "e" * 64, True))})),
                       dict(anc, valid=dict(anc["valid"], **{f"{FS}@1": _valid()})), bd)["verdict"], "INVALID EVAL")
    check("validity: a field_turn row env block differing from the anchor's -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FT}@1": _valid({"KRABBY_FIELD_PAD_R": "1.8"})})),
                       dict(anc, valid=dict(anc["valid"], **{f"{FT}@1": _valid()})), bd)["verdict"], "INVALID EVAL")
    lv = levels_summary(([14996 + i for i in range(1000)], [1.0 + i / 999 for i in range(1000)]))
    check("levels: start = first 20 points, tail = last 100 iterations",
          (round(lv["start"], 4), round(lv["tail"], 4), lv["last_step"]),
          (round(1.0 + 9.5 / 999, 4), round(1.0 + 949.5 / 999, 4), 15995))
    check("levels: no series -> None", levels_summary(None), None)
    ft = {"turn_fc": 0.8, "turn_yp": 0.2, "completion": 0.9, "fall_share": 0.05}
    ri2 = rank_inputs({"a": _pt("a", field_turn=ft), "b": _pt("b", field_turn=dict(ft, turn_fc=0.7, turn_yp=0.1))}, {})
    check("rank inputs: field_turn turn holds give the turn rank-sum (item 5 stand-in)",
          [ri2["a"]["turn_rank_sum"], ri2["b"]["turn_rank_sum"], ri2["a"]["field_turn_completion"]], [2.0, 4.0, 0.9])
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================================================ CLI
def _f(x, fmt="{:.3f}"):
    return "--" if x is None else fmt.format(x)


def markdown(res: dict, vs: dict) -> str:
    L = ["# Omni phase 2 -- the pre-registered reading (CHARTER 6)", "",
         "| head | verdict | gain | field_hard (pooled) | field_light | ring_obst | why |", "|---|---|---|---|---|---|---|"]
    for ref, v in vs.items():
        pt = res["heads"].get(ref) or {}
        L.append(f"| {ref} | **{v['verdict']}** | {v.get('gain')} | {_f((pooled(pt, 'field_hard') or (None,))[0])} | "
                 f"{_f((pooled(pt, 'field_light') or (None,))[0])} | {_f((pooled(pt, 'ring_obst') or (None,))[0])} | "
                 f"{'; '.join(str(x) for x in (v.get('why') or []))[:300]} |")
    rr = recipe_read({s: vs.get(f"P2__A1_s{s}") for s in SEEDS})
    L += ["", f"**Recipe read (6.2): {rr['reading']}** {json.dumps({k: v for k, v in rr.items() if k != 'reading'})}"]
    pts = {s: res["heads"].get(f"P2__A1_s{s}") for s in SEEDS}
    L += [f"**p2b trigger (6.5):** {json.dumps(p2b_trigger({s: vs.get(f'P2__A1_s{s}') or {} for s in SEEDS}, pts))}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--kept", default="")
    ap.add_argument("--anchors-check", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--field-vs-ring", action="store_true")
    ap.add_argument("--sibling", choices=SIBLINGS)
    ap.add_argument("--recipe", action="store_true")
    ap.add_argument("--m-screen", metavar="REF")
    ap.add_argument("--calibrate", type=Path)
    ap.add_argument("--smoke-s5", metavar="STAMP|NAME")
    ap.add_argument("--write-smokes-ok", action="store_true")
    ap.add_argument("--smokes", type=Path)
    ap.add_argument("--s5", metavar="STAMP|NAME")
    ap.add_argument("--no-charter-log", action="store_true")
    ap.add_argument("--smokes-ok-check", action="store_true")
    ap.add_argument("--cal2-gate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--code-root", type=Path, default=R)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    out = a.out or OUT
    arms_dir, lr_dir = out / "arms", out / "lr"
    if a.smoke_s5:
        r = smoke_s5_read(resolve_stamp(a.smoke_s5, FL, 1, lr_dir), _anchors_yaml()[GATE_SEED]["sha256"])
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.write_smokes_ok:
        if not a.smokes or not a.s5:
            ap.error("--write-smokes-ok needs --smokes DIR and --s5 STAMP|NAME")
        s5 = smoke_s5_read(resolve_stamp(a.s5, FL, 1, lr_dir), _anchors_yaml()[GATE_SEED]["sha256"])
        import yaml  # noqa: PLC0415
        fails, payload = build_smokes_ok(a.smokes, s5, _json(arms_dir / f"{CAL1}.record.json"),
                                         _live_code_state(a.code_root),
                                         yaml.safe_load(open(CAMPAIGN / "candidates.yaml")))
        if fails:
            print(json.dumps({"ok": False, "fails": fails}, default=str))
            return 4
        out.mkdir(parents=True, exist_ok=True)
        (out / SMOKES_OK).write_text(json.dumps(payload, indent=1, default=str))
        if not a.no_charter_log:
            with open(CAMPAIGN / "CHARTER.md", "a") as fh:
                fh.write(charter_line(payload))
        print(json.dumps({"ok": True, "written": str(out / SMOKES_OK), **payload}, default=str))
        return 0
    if a.smokes_ok_check:
        import yaml  # noqa: PLC0415
        payload = _json(out / SMOKES_OK)
        fails = smokes_ok_fails(payload, _live_code_state(a.code_root), yaml.safe_load(open(CAMPAIGN / "candidates.yaml")))
        print(json.dumps({"ok": not fails, "fails": fails, "smokes_ok": str(out / SMOKES_OK)}, default=str))
        return 0 if not fails else 4
    if a.cal2_gate:
        import yaml  # noqa: PLC0415
        r = cal2_gate(_json(arms_dir / f"{CAL2}.record.json"), yaml.safe_load(open(CAMPAIGN / "candidates.yaml")))
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.calibrate:
        tb = a.calibrate.is_dir() or a.calibrate.name.startswith("events.out.tfevents")
        r = calibrate_run(a.calibrate) if tb else calibrate_file(a.calibrate)
        print(json.dumps(r))
        return 0 if r.get("ok") else 2
    if a.anchors_check:
        an = _anchors_yaml()
        anc = {int(s): load_point(x["name"], with_tb=False) for s, x in an.items()}
        r = anchors_check(anc, _manifest_s18(), {int(s): x["sha256"] for s, x in an.items()})
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.m_screen:
        an = _anchors_yaml()
        s = int(a.m_screen.rsplit("_s", 1)[1])
        anc = load_point(an[s]["name"], with_tb=False)
        s18 = anc if s == GATE_SEED else load_point(an[GATE_SEED]["name"], with_tb=False)
        rec = _rec(a.m_screen) or {}
        head = load_point(a.m_screen, suite_stamp=rec.get("driver_stamp"), with_tb=False)
        ms = {it: load_point(f"{a.m_screen}_m{it}", with_tb=False) for it in MCKPTS}
        r = m_screen(head, ms, anc, bands(anc, s18))
        for it, v in r.items():
            if v["status"] == "NEEDS_SEED2":
                print(f"{a.m_screen}_m{it}")
        print(json.dumps(r, default=str), file=sys.stderr)
        return 0
    with_raw = bool(a.sibling or a.field_vs_ring or a.json or a.md)
    res = load_all(with_tb=not a.no_tb, with_raw=with_raw)
    vs = verdicts(res)
    if a.gate:
        v = vs.get(f"P2__A1_s{GATE_SEED}")
        print(json.dumps(v, default=str))
        return 0 if v and v["verdict"] == "PASS" else 4
    if a.sibling:
        sib, a1 = res["heads"].get(f"P2__{a.sibling}_s18"), res["heads"].get("P2__A1_s18")
        anc = res["anchors"][GATE_SEED]
        if not sib or sib.get("missing") or not a1 or a1.get("missing"):
            print(json.dumps({"arm": a.sibling, "rec": "PENDING", "why": "sibling or A1 s18 not recorded"}))
            return 4
        r = sibling_rec(a.sibling, sib, a1, bands(anc, anc))
        r["seed_verdict_vs_anchor"] = vs.get(f"P2__{a.sibling}_s18", {}).get("verdict")
        print(json.dumps(r, default=str))
        return 4
    if a.field_vs_ring:
        a1, a0 = res["heads"].get("P2__A1_s18") or {}, res["heads"].get("P2__A0_s18") or {}
        out = {}
        for lab, pt in (("A1", a1), ("A0", a0)):
            out[lab] = {k: pooled(pt, k) for k in ("field_hard", "ring_obst")} if not pt.get("missing") else None
            if out[lab] is not None:
                out[lab]["falls_per_ep"] = {k: (sum((pt["comp"][k][s] or {}).get("falls", 0) for s in (1, 2)) /
                                                max(1, sum((pt["comp"][k][s] or {}).get("n", 0) for s in (1, 2))))
                                            for k in ("field_hard", "ring_obst")}
                out[lab]["exposure"] = pt.get("exposure")
                out[lab]["run_dir"] = pt.get("run_dir")
                out[lab]["terrain_levels"] = terrain_levels(pt.get("run_dir")) if not a.no_tb else None
        print(json.dumps(out, default=str))
        return 0
    if a.recipe:
        r = recipe_read({s: vs.get(f"P2__A1_s{s}") for s in SEEDS})
        t = p2b_trigger({s: vs.get(f"P2__A1_s{s}") or {} for s in SEEDS},
                        {s: res["heads"].get(f"P2__A1_s{s}") for s in SEEDS})
        print(json.dumps({"recipe": r, "p2b": t}, default=str))
        return 4 if (t["trigger"] or r["reading"] == "PENDING") else 0
    kept = [k for k in a.kept.split(",") if k]
    cands = {f"A1_s{s}": res["heads"][f"P2__A1_s{s}"] for s in SEEDS if not res["heads"][f"P2__A1_s{s}"].get("missing")}
    cands.update({f"{k}_s18": res["heads"][f"P2__{k}_s18"] for k in kept if not res["heads"][f"P2__{k}_s18"].get("missing")})
    rk = rank_inputs(cands, {lab: vs.get(f"P2__{lab}") for lab in cands})
    md = markdown(res, vs)
    if a.json:
        a.json.write_text(json.dumps({"verdicts": vs, "rank_inputs": rk}, indent=1, default=str))
    if a.md:
        a.md.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""PPO-construction decision support (CPU, read-only): the pre-registered reads and verdicts of CHARTER.md section 4
(plan step 3 item 9), applied mechanically.

MODES
  verdict.py [--records-dir D] [--candidates F] [--json OUT] [--md OUT]
      every run recorded by run_arm.sh (<ARMS>/<ref>.record.json) -> per-run reads, per-bed verdicts, the C-bed
      guard, the pilot, the LR-floor confound, "Q better"; JSON + a markdown table (stdout unless --md).
  verdict.py --check-run RUN_DIR --chain CHAIN --at 200|1000|live [--train-log F] [--candidates F]
      the live kill checks run_arm.sh calls while an arm trains. Exit codes (distinct, so the runner can tell a
      true stop from a continue):
        0 pass or not applicable (C0 / INC0; `live` on a non-4a arm)    2 not enough telemetry yet (retry)
        3 WIRING fail  -> kill, STOP the queue (fix, rerun under a fresh reference; the rerun counts)
        4 DOSE fail    -> kill, STOP the queue (CHARTER 4.3: check wiring; wiring right = stop and ask)
        5 DEAD         -> kill, the queue continues (the run counts as not FAST)
        6 DIVERGED (4a only: KL > 0.1 for 50 iterations, or reward -30 % over 500) -> kill, the queue continues
      `live` = the 4a divergence kills only (polled through the run). The last stdout line is a one-line JSON.
  verdict.py --last-iter RUN_DIR      the last logged training iteration (0 if none).
  verdict.py --self-test              the rules on synthetic inputs covering every branch; exit 0 iff all pass.

THE RULES (CHARTER 4, verbatim intent; the numbers are the module constants below)
  Hand class (4.1): side22_classify.py feet_ss class; a hand is FAST only if along_cmd.py's along-command class is
    FAST too. A feet-FAST hand whose along class is not FAST takes the along class (the stricter read). FAIL CLOSED:
    a feet-FAST hand with NO along class, or a hand with no feet class, is NA (never FAST) and its run is excluded
    from every count as "along read missing: rerun along_cmd.py" until the read exists. Run FAST: two-direction
    beds both hands FAST (both-hands stamp); C bed lat_l FAST (driver stamp).
  Killed and invalid runs: a run killed DEAD (or 4a DIVERGED) counts as not FAST in n and k. A run killed for
    WIRING, or whose post-hoc wiring read fails, is excluded ("rerun required"; its rerun -- a fresh reference
    of the same chain and seed, named in CHARTER 9 -- counts). A DOSE-killed run is excluded and routed to the
    DOSE stop (CHARTER 4.3; the pilot reads DOSE). Two valid runs of one chain and seed: the first recorded
    counts, the other is a re-roll (D18) and is excluded.
  Survival tripwire (4.4), every construction (A/B-type) run: hold-fall share = short / (n + short + static) pooled
    over every lat_* hold of the classifying stamp (side22_classify.py's counts). Reference = median of the bed's C0
    shares excluding C0 runs > 0.25; on two-direction beds max(bed C0 median, C-bed C0 median). Trips above
    max(0.15, reference + 0.10); a tripped run counts as not FAST. No reference -> threshold 0.15, flagged.
  Kills (4.2) for construction arms: wiring at it 200 -- the [KNOB] consumed lines of every new knob in the overlay,
    PPO/effective_std present, Loss/action_bound present for bound arms, PPO/effective_std < 1.0 (window 190-210)
    only for init-0.8 arms that are not bisection arms; the dumped entropy_coef / action_rate weight equal the
    overlay. DEAD at it 1000 (window 950-1050): track_ratio_mean < 0.40 or PPO/effective_std < 0.1.
  Dose (4.3, amended 2026-10-01), bound arms: PPO/mu_med_{cam,rod} <= 1.5 (sat15 reported only) at it 1000 (950-1050) and at the
    final window (2900-2999). A failure at 1000 = kill + wiring check; at the final window only = "dose lost late"
    (the run's class still counts, the flag is reported).
  Two-direction beds (4.5), per non-C0 arm vs the bed's C0, precedence VOID > UNLOCK > PARTIAL > NO-FLIP:
    VOID     C0 FAST in >= 2 runs (2/3 at three seeds, at once; 2/5 after extension) -> "bed not reliably SLOW at
             fresh seeds"; extend C0 only; arm counts descriptive.
    UNLOCK   at 5 seeds each: arm 4/5 with C0 0/5, or arm 5/5 with C0 <= 1/5.
    PARTIAL  ("inconclusive", never "not the limiter") at 5 seeds 2-4/5 without UNLOCK; or any arm hand
             INTERMEDIATE; or a FAST hand in a run not counted FAST; or a non-FAST arm run whose both-hands feet_ss
             (hands averaged) rises model_2500 -> model_2999 by more than both 0.03 m/s and the largest C0 rise.
    NO-FLIP  none of the above, and the arm FAST 0/3 at three seeds or <= 1/5 at five.
    otherwise PENDING (an arm FAST run at three seeds -> extension) or INCOMPLETE (< 3 seeds).
    Extension: any arm run FAST -> that arm + C0 to seeds 14-15; a single C0 FAST run -> every arm + C0.
    NO-FLIP is never issued while an extension is pending (that reads PENDING).
  Step 4 (CHARTER 6): 4a/4b arms (<head>_fixed, <head>_epochkl, <head>_1024) are compared per seed with their
    CONTROL, the step-3 run of the head (<bed>__<head>) at the same seed. A FAST where the control was not ->
    extend that arm + C0 to seeds 14-15; at five seeds each the UNLOCK rule vs C0 applies, and a step-4 UNLOCK is
    listed apart from the step-3 4c triggers ("unlock_step4"). 4a arms also get the 4a wiring read at it 200
    (PPO/lr_applied == KRABBY_PPO_LR for `fixed`, PPO/kl_stop present, PPO/hist_lr == KRABBY_PPO_HIST_LR) and the
    divergence kills. INC0 is never an UNLOCK: its rule is INC0 FAST >= B FAST - 1 -> stop and ask.
  C-bed guard (4.7): PASS A FAST >= 2/3; FAIL A <= 1/3 and C0 >= 2/3 -> run C__B, then the bisection
    (C__A_ent01, C__A_std15, C__bound_only at 2 seeds each, stopping at the first that recovers C0's class = 2/2
    FAST); UNDEFINED otherwise (C0 <= 1/3: "C bed not reliable at fresh seeds"; or A between the bands, e.g.
    2/5 or 3/5, with C0 >= 2/3: neither PASS nor FAIL).
  Q better (4.6), per bed and arm: every arm run's hands-averaged action_jitter (median per hand of the classifying
    stamp's side category, scenario_metrics.json) below every C0 run's (>= 3 runs each), and the arm's mean
    action_fft_freq <= C0's. "Adopt for motion quality" is proposed when Q is better, the C-bed guard PASSes and no
    run of that arm tripped.
  LR floor (4.8), step-3 A/B runs only: share of its 500-2999 with Loss/learning_rate <= 1.0001e-5 > 0.9, or mean
    PPO/lr_applied over 500-2999 < 0.5 x the bed's C0 median of the same mean -> 4a triggered (both numbers
    printed). (A step-4 fixed-LR arm would trip the relative test by construction; it never re-triggers 4a.)
  Pilot (3.1): seed 11 of C__C0, C__A, C_lr__C0, C_lr__A.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
DEFAULT_ARMS = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction/arms")

# ------------------------------------------------------------------------------------------- constants (CHARTER 4)
TRIP_FLOOR = 0.15
TRIP_MARGIN = 0.10
TRIP_COLLAPSE = 0.25          # C0 runs above this are excluded from the tripwire reference
DOSE_MU_MED = 1.5
DOSE_SAT15 = 0.1
W_1000 = (950, 1050)
W_FINAL = (2900, 2999)
W_200 = (190, 210)
WIRING_AT = 210               # tags must have a point at a step <= this
DEAD_TRACK = 0.40
DEAD_STD = 0.1
SIGMA_WIRING_MAX = 1.0
LR_FLOOR = 1.0001e-5
LR_FLOOR_SHARE = 0.9
LR_REL = 0.5
W_LR = (500, 2999)
LATE_RISE_MIN = 0.03
W_HANDCOS = (0, 600)
W_LOCK = (300, 600)
#: 4a divergence kills (plan step 10): KL > 0.1 for 50 consecutive iterations; reward down 30 % over 500.
KILL_KL = 0.1
KILL_KL_ITERS = 50
KILL_REWARD_DROP = 0.30
KILL_REWARD_SPAN = 500
KILL_REWARD_AVG = 50          # each end of the reward comparison is a 50-iteration mean
LR_REL_TOL = 1e-5             # PPO/lr_applied vs the pinned LR (tfevents store float32)
#: past this many iterations beyond a read's window, telemetry still missing is a wiring failure (no endless retry)
MISSING_GRACE = 200

#: --check-run exit codes (run_arm.sh keys its queue on them)
RC_PASS, RC_WAIT, RC_WIRING, RC_DOSE, RC_DEAD, RC_DIVERGED = 0, 2, 3, 4, 5, 6
KILL_KIND = {RC_WIRING: "wiring", RC_DOSE: "dose", RC_DEAD: "dead", RC_DIVERGED: "diverged"}

#: the new knobs that print a "[KNOB] NAME=value consumed" line when applied (plan step 2 item 3)
NEW_KNOBS = ("KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND", "KRABBY_STD_PARAM",
             "KRABBY_INIT_STD", "KRABBY_PPO_SCHEDULE", "KRABBY_PPO_LR", "KRABBY_PPO_DESIRED_KL",
             "KRABBY_PPO_KL_STOP", "KRABBY_PPO_HIST_LR", "KRABBY_PPO_HAND_DIAG", "KRABBY_NUM_ENVS",
             "KRABBY_RSI_CAM_STAGE")
#: the 0/1 flag knobs among them (motion_knobs.parse_flag: "1" on, "0" / blank off)
FLAG_KNOBS = ("KRABBY_PPO_HAND_DIAG", "KRABBY_RSI_CAM_STAGE", "KRABBY_RSI_CAM_STAGE_DEBUG")
#: knobs that make a run a construction (A/B-type) arm: kills, dose, tripwire, LR floor apply
CONSTRUCTION_KNOBS = ("KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND",
                      "KRABBY_STD_PARAM", "KRABBY_INIT_STD", "KRABBY_ENTROPY_COEF", "KRABBY_PPO_SCHEDULE",
                      "KRABBY_PPO_LR", "KRABBY_PPO_DESIRED_KL", "KRABBY_PPO_KL_STOP", "KRABBY_PPO_HIST_LR")
BISECTION = ("C__A_ent01", "C__A_std15", "C__bound_only")
TWO_DIR_BEDS = ("C_lr", "C_lr_rsi", "C_lr_rsi_st")
#: step-4 arm names: 4a / 4b are ``<head>`` + one of these suffixes (any head, any bed); INC0 is 4c's attribution arm
STEP4_SUFFIXES = ("_fixed", "_epochkl", "_1024")
INC0 = "INC0"
#: the step-3 arms whose UNLOCK triggers 4c and whose LR floor triggers 4a
STEP3_ARMS = ("A", "B")


def step4_control(arm: str) -> str | None:
    """The head arm a 4a/4b arm is controlled by (``A_fixed`` -> ``A``, ``C0_1024`` -> ``C0``), else None."""
    for suf in STEP4_SUFFIXES:
        if arm.endswith(suf) and len(arm) > len(suf):
            return arm[: -len(suf)]
    return None


def is_step4(arm: str) -> bool:
    return arm == INC0 or step4_control(arm) is not None


TAGS = {
    "mu_med_cam": "PPO/mu_med_cam", "mu_med_rod": "PPO/mu_med_rod",
    "sat15_cam": "PPO/sat15_cam", "sat15_rod": "PPO/sat15_rod",
    "lock_cam": "PPO/lock_cam", "lock_rod": "PPO/lock_rod",
    "effective_std": "PPO/effective_std", "action_bound": "Loss/action_bound",
    "track": "Metrics/base_parkour/track_ratio_mean", "track_rsi": "Metrics/base_parkour/track_ratio_mean_rsi",
    "lr": "Loss/learning_rate", "lr_applied": "PPO/lr_applied", "hand_cos": "PPO/hand_grad_cos",
    "symmetry": "Loss/symmetry", "kl_mean": "PPO/kl_mean", "kl_stop": "PPO/kl_stop", "hist_lr": "PPO/hist_lr",
    "reward": "Train/mean_reward",
}


# ------------------------------------------------------------------------------------------- small helpers
def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.median(xs)) if xs else None


def tci90(xs):
    """(mean, half-width) of a 90 % t-interval (descriptive only; 3 v 3 MDE ~ 2.75 seed-SD)."""
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if len(xs) < 2:
        return (_mean(xs), None)
    from scipy import stats  # noqa: PLC0415

    m, s = float(np.mean(xs)), float(np.std(xs, ddof=1))
    return (m, float(stats.t.ppf(0.95, len(xs) - 1) * s / math.sqrt(len(xs))))


def split_chain(chain: str) -> tuple[str, str]:
    bed, sep, arm = chain.rpartition("__")
    return (bed, arm) if sep else (chain, "")


def window(series, lo, hi):
    """Mean of a (steps, values) series over steps lo..hi inclusive, or None."""
    if series is None:
        return None
    st, v = series
    m = (st >= lo) & (st <= hi)
    return float(np.nanmean(v[m])) if m.any() else None


def present_by(series, step):
    return series is not None and bool((series[0] <= step).any())


def _yaml_tolerant(path: Path):
    import yaml  # noqa: PLC0415

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: tuple(ld.construct_sequence(node)))
    _L.add_multi_constructor("", lambda ld, suffix, node: None)
    with open(path) as fh:
        return yaml.load(fh, Loader=_L)  # noqa: S506


def load_candidates(path: Path) -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load(Path(path).read_text())


def chain_overlay(cands: dict, chain: str) -> dict:
    ov: dict = {}
    for st in (cands.get("chains") or {}).get(chain, {}).get("stages") or []:
        ov.update({str(k): str(v) for k, v in (st.get("overlay") or {}).items()})
    return ov


def arm_kind(chain: str, overlay: dict) -> dict:
    """What the pre-registered reads apply to for this chain."""
    construction = any(k in overlay for k in CONSTRUCTION_KNOBS)
    bound = "KRABBY_ACTION_BOUND_LOSS" in overlay
    sigma_check = overlay.get("KRABBY_INIT_STD") in ("0.8", "0.80") and chain not in BISECTION
    schedule = overlay.get("KRABBY_PPO_SCHEDULE", "").strip().lower()

    def _num(key):
        try:
            return float(overlay[key]) if key in overlay and str(overlay[key]).strip() else None
        except ValueError:
            return None

    return {"construction": construction, "bound": bound, "sigma_check": sigma_check,
            # a knob set to its OFF spelling prints no consumed line (the flag knobs: "0" / blank = off)
            "new_knobs": [k for k in NEW_KNOBS if k in overlay
                          and not (k in FLAG_KNOBS and str(overlay[k]).strip() in ("", "0"))],
            # step 4a (plan step 10): a non-adaptive schedule arms the 4a wiring read and the divergence kills
            "step4a": schedule in ("fixed", "epoch_kl"), "schedule": schedule or None,
            "ppo_lr": _num("KRABBY_PPO_LR"), "kl_stop": _num("KRABBY_PPO_KL_STOP"),
            "hist_lr": _num("KRABBY_PPO_HIST_LR")}


# ------------------------------------------------------------------------------------------- telemetry
def tb_series(run_dir: Path, tags=None) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    out = {}
    for key, tag in (tags or TAGS).items():
        if tag in have:
            s = ea.Scalars(tag)
            out[key] = (np.array([x.step for x in s]), np.array([x.value for x in s], dtype=float))
    out["_last"] = max((int(v[0].max()) for v in out.values() if len(v[0])), default=0)
    return out


def consumed_lines(train_log: Path | None, knobs) -> dict:
    """{knob: bool} -- an exact "[KNOB] NAME=<value> consumed" line in the train log (the whole line: the
    "[KNOB] NAME=<value> NOT consumed: ..." line of an armed-but-unused knob never counts)."""
    if not knobs:
        return {}
    text = ""
    if train_log and Path(train_log).is_file():
        text = Path(train_log).read_text(errors="replace")
    return {k: bool(re.search(r"^\[KNOB\] " + re.escape(k) + r"=\S+ consumed\s*$", text, re.M)) for k in knobs}


def dump_checks(run_dir: Path, overlay: dict) -> dict:
    """The dumped value of the two EXISTING knobs the arms set (they print no consumed line)."""
    out = {}
    if "KRABBY_ENTROPY_COEF" in overlay:
        p = Path(run_dir) / "params" / "agent.yaml"
        got = None
        if p.is_file():
            got = ((_yaml_tolerant(p) or {}).get("algorithm") or {}).get("entropy_coef")
        out["entropy_coef"] = {"want": float(overlay["KRABBY_ENTROPY_COEF"]), "got": got,
                               "ok": got is not None and abs(float(got) - float(overlay["KRABBY_ENTROPY_COEF"])) < 1e-12}
    if "KRABBY_ACTION_RATE_W" in overlay:
        p = Path(run_dir) / "params" / "env.yaml"
        got = None
        if p.is_file():
            got = (((_yaml_tolerant(p) or {}).get("rewards") or {}).get("reward_action_rate") or {}).get("weight")
        out["action_rate_w"] = {"want": float(overlay["KRABBY_ACTION_RATE_W"]), "got": got,
                                "ok": got is not None and abs(float(got) - float(overlay["KRABBY_ACTION_RATE_W"])) < 1e-12}
    return out


def wiring_read(ser: dict, kind: dict, consumed: dict, dumps: dict) -> dict:
    """CHARTER 4.2 wiring at it 200 (construction arms). ``ok`` None = not enough telemetry yet."""
    if ser.get("_last", 0) < W_200[1]:
        return {"ok": None, "why": f"telemetry only to it {ser.get('_last', 0)}"}
    fails = []
    if not present_by(ser.get("effective_std"), WIRING_AT):
        fails.append("PPO/effective_std absent")
    if kind["bound"] and not present_by(ser.get("action_bound"), WIRING_AT):
        fails.append("Loss/action_bound absent")
    sig200 = window(ser.get("effective_std"), *W_200)
    if kind["sigma_check"] and sig200 is not None and not sig200 < SIGMA_WIRING_MAX:
        fails.append(f"PPO/effective_std {sig200:.3f} >= {SIGMA_WIRING_MAX} at it 200")
    fails += [f"no [KNOB] {k} consumed line" for k, ok in consumed.items() if not ok]
    fails += [f"dumped {k} {d['got']} != {d['want']}" for k, d in dumps.items() if not d["ok"]]
    if kind.get("step4a"):
        fails += step4a_wiring_fails(ser, kind)
    return {"ok": not fails, "fails": fails, "effective_std_200": sig200}


def _rel_ok(got, want) -> bool:
    return got is not None and want is not None and abs(float(got) - want) <= LR_REL_TOL * abs(want)


def step4a_wiring_fails(ser: dict, kind: dict) -> list[str]:
    """Plan step 5 / 10: the 4a iteration-200 assertions, on every logged point through it 210.

    ``fixed``: PPO/lr_applied (the policy optimizer's single param group, averaged per update) == KRABBY_PPO_LR;
    with KRABBY_PPO_KL_STOP set, PPO/kl_stop present; with KRABBY_PPO_HIST_LR set, PPO/hist_lr (the
    history-encoder optimizer's own LR) == that value. ``epoch_kl`` moves the LR, so only the last two apply.
    """
    fails = []

    def upto(key):
        s = ser.get(key)
        if s is None:
            return None
        m = s[0] <= WIRING_AT
        return s[1][m] if m.any() else None

    if kind.get("schedule") == "fixed" and kind.get("ppo_lr") is not None:
        v = upto("lr_applied")
        if v is None:
            fails.append("PPO/lr_applied absent")
        elif not all(_rel_ok(x, kind["ppo_lr"]) for x in v):
            fails.append(f"PPO/lr_applied {float(np.min(v)):.3g}..{float(np.max(v)):.3g} != {kind['ppo_lr']:g}")
    if kind.get("kl_stop") is not None and upto("kl_stop") is None:
        fails.append("PPO/kl_stop absent")
    if kind.get("hist_lr") is not None:
        v = upto("hist_lr")
        if v is None:
            fails.append("PPO/hist_lr absent")
        elif not all(_rel_ok(x, kind["hist_lr"]) for x in v):
            fails.append(f"PPO/hist_lr {float(np.min(v)):.3g}..{float(np.max(v)):.3g} != {kind['hist_lr']:g}")
    return fails


def divergence_read(ser: dict) -> dict:
    """4a kills (plan step 10): PPO/kl_mean > 0.1 on 50 consecutive logged iterations, anywhere so far; or
    Train/mean_reward's 50-iteration mean ending at some iteration t below its mean ending at t - 500 by more
    than 30 % of that earlier mean's magnitude. ``diverged`` None = not enough telemetry for either test."""
    out = {"diverged": None, "kl_run_max": None, "kl_kill_at": None, "reward_kill_at": None}
    kl = ser.get("kl_mean")
    if kl is not None and len(kl[0]):
        order = np.argsort(kl[0])
        st, v = kl[0][order], kl[1][order]
        run = best = 0
        prev = None
        for step, x in zip(st, v):
            if x > KILL_KL:
                run = run + 1 if (run > 0 and step == prev + 1) else 1     # a logging gap restarts the run
            else:
                run = 0
            prev = step
            best = max(best, run)
            if run >= KILL_KL_ITERS and out["kl_kill_at"] is None:
                out["kl_kill_at"] = int(step)
        out["kl_run_max"] = int(best)
        out["diverged"] = out["kl_kill_at"] is not None
    rw = ser.get("reward")
    if rw is not None and len(rw[0]):
        st, v = rw
        for t in np.unique(st[st >= KILL_REWARD_SPAN + KILL_REWARD_AVG - 1]):
            now = v[(st > t - KILL_REWARD_AVG) & (st <= t)]
            then = v[(st > t - KILL_REWARD_SPAN - KILL_REWARD_AVG) & (st <= t - KILL_REWARD_SPAN)]
            if len(now) and len(then):
                r_now, r_then = float(np.mean(now)), float(np.mean(then))
                if r_then != 0.0 and r_now < r_then - KILL_REWARD_DROP * abs(r_then):
                    out["reward_kill_at"] = int(t)
                    out["reward_then_now"] = [r_then, r_now]
                    break
        out["diverged"] = bool(out["diverged"]) or out["reward_kill_at"] is not None
    return out


def dead_read(ser: dict) -> dict:
    tr, sd = window(ser.get("track"), *W_1000), window(ser.get("effective_std"), *W_1000)
    if tr is None and sd is None:
        return {"dead": None, "track_1000": None, "std_1000": None}
    dead = (tr is not None and tr < DEAD_TRACK) or (sd is not None and sd < DEAD_STD)
    return {"dead": bool(dead), "track_1000": tr, "std_1000": sd}


def dose_at(ser: dict, w) -> dict:
    vals = {k: window(ser.get(k), *w) for k in ("mu_med_cam", "mu_med_rod", "sat15_cam", "sat15_rod")}
    if any(v is None for v in vals.values()):
        return {"ok": None, **vals}
    # 2026-10-01 user decision (CHARTER 4.3 amendment, 9 log): the dose is median |mu| <= 1.5 on cams and rods;
    # the share |mu| > 1.5 is reported (sat15_*) but no longer gates. DOSE_SAT15 stays defined for the report only.
    ok = vals["mu_med_cam"] <= DOSE_MU_MED and vals["mu_med_rod"] <= DOSE_MU_MED
    return {"ok": bool(ok), "sat15_gate": False, **vals}


def lr_reads(ser: dict) -> dict:
    out = {"floor_share": None, "lr_applied_mean": None}
    s = ser.get("lr")
    if s is not None:
        m = (s[0] >= W_LR[0]) & (s[0] <= W_LR[1])
        if m.any():
            out["floor_share"] = float(np.mean(s[1][m] <= LR_FLOOR))
    out["lr_applied_mean"] = window(ser.get("lr_applied"), *W_LR)
    return out


def check_run(run_dir: Path, chain: str, at, train_log: Path | None, cands: dict) -> tuple[int, dict]:
    """--check-run: (exit code, summary); the codes are the RC_* constants (module docstring)."""
    ov = chain_overlay(cands, chain)
    if not ov and chain not in (cands.get("chains") or {}):
        return RC_WAIT, {"chain": chain, "why": "chain not in candidates.yaml"}
    kind = arm_kind(chain, ov)
    if not kind["construction"]:
        return RC_PASS, {"chain": chain, "at": at, "result": "n/a (not a construction arm)"}
    if at == "live" and not kind["step4a"]:
        return RC_PASS, {"chain": chain, "at": at, "result": "n/a (not a 4a arm)"}
    ser = tb_series(run_dir)
    last = ser.get("_last", 0)
    if at == "live":
        dv = divergence_read(ser)
        if dv["diverged"] is None:
            return RC_WAIT, {"chain": chain, "at": at, "why": "no PPO/kl_mean or Train/mean_reward yet", **dv}
        return ((RC_DIVERGED, {"chain": chain, "at": at, "result": "FAIL", "kill": "diverged", **dv}) if dv["diverged"]
                else (RC_PASS, {"chain": chain, "at": at, "result": "PASS", **dv}))
    at = int(at)
    if at <= 200:
        w = wiring_read(ser, kind, consumed_lines(train_log, kind["new_knobs"]), dump_checks(run_dir, ov))
        if w["ok"] is None:
            return RC_WAIT, {"chain": chain, "at": at, **w}
        return ((RC_PASS, {"chain": chain, "at": at, "result": "PASS", **w}) if w["ok"]
                else (RC_WIRING, {"chain": chain, "at": at, "result": "FAIL", "kill": "wiring", **w}))
    d = dead_read(ser)
    dose = dose_at(ser, W_1000) if kind["bound"] else {"ok": True, "n/a": "no bound"}
    if d["dead"] is None or dose["ok"] is None:
        if last >= W_1000[1] + MISSING_GRACE:
            return RC_WIRING, {"chain": chain, "at": at, "result": "FAIL", "kill": "wiring", "dead": d, "dose": dose,
                               "why": f"telemetry still missing at it {last} (always-on tags: wiring)"}
        return RC_WAIT, {"chain": chain, "at": at, "dead": d, "dose": dose, "why": "telemetry missing"}
    why = ([f"DEAD (track {d['track_1000']}, std {d['std_1000']})"] if d["dead"] else []) + \
          ([f"DOSE at it 1000 {dose}"] if not dose["ok"] else [])
    # a dose failure is a stop (CHARTER 4.3) and outranks DEAD (which only kills)
    rc = RC_DOSE if not dose["ok"] else (RC_DEAD if d["dead"] else RC_PASS)
    return rc, {"chain": chain, "at": at, "result": "FAIL" if rc else "PASS", "kill": KILL_KIND.get(rc), "why": why,
                "dead": d, "dose": dose}


# ------------------------------------------------------------------------------------------- stamp reads
#: the hand class of a missing read (never FAST; the run is excluded until the read exists)
NA = "NA"


def hand_class(feet, along):
    """CHARTER 4.1: FAST only if both FAST; a feet-FAST hand takes its (non-FAST) along class. FAIL CLOSED: no feet
    class, or a feet-FAST hand with no along class, is NA -- never FAST (a crashed along_cmd.py must not pass a
    feet-only FAST). A feet class other than FAST stands without the along read (along can only make it stricter)."""
    if not feet:
        return NA
    if feet == "FAST":
        return along if along else NA
    return feet


def fall_share(class_json: dict) -> float | None:
    """short / (n + short + static) pooled over every lat_* hold (side22_classify.py's counts)."""
    n = sh = stt = 0
    for lab, r in (class_json.get("labels") or {}).items():
        if not lab.startswith("lat_"):
            continue
        n += int(r.get("n") or 0)
        ex = r.get("excluded") or {}
        sh += int(ex.get("short") or 0)
        stt += int(ex.get("static") or 0)
    tot = n + sh + stt
    return sh / tot if tot else None


def feet_hands_avg(class_json: dict) -> float | None:
    """Hands-averaged feet_ss: mean over hands of mean(lat_h_022, lat_h_025) medians."""
    L = class_json.get("labels") or {}
    hands = []
    for h in ("l", "r"):
        v = [L.get(f"lat_{h}_{s}", {}).get("feet_ss") for s in ("022", "025")]
        v = [x for x in v if x is not None]
        if v:
            hands.append(float(np.mean(v)))
    return float(np.mean(hands)) if hands else None


def q_reads(stamp: str | None) -> dict:
    """Hands-averaged median action_jitter / action_fft_freq of the stamp's side category."""
    if not stamp:
        return {"jitter": None, "fft": None}
    p = Path(stamp) / "scenario_metrics.json"
    if not p.is_file():
        return {"jitter": None, "fft": None}
    side = ((json.loads(p.read_text()).get("aggregate") or {}).get("suite") or {}).get("by_category", {}).get("side") or {}
    out = {}
    for key, name in (("jitter", "action_jitter"), ("fft", "action_fft_freq")):
        bh = side.get("by_hand")
        if isinstance(bh, dict) and bh:
            out[key] = _mean([((bh.get(h) or {}).get(name) or {}).get("median") for h in sorted(bh)])
        else:
            out[key] = (side.get(name) or {}).get("median")
    return out


def _load_json(p):
    return json.loads(Path(p).read_text()) if p and Path(p).is_file() else None


def facts_from_record(rec: dict, cands: dict, with_tb: bool = True) -> dict:
    """Everything the rules need for one run, from run_arm.sh's record and the files it names."""
    ref, chain = rec["ref"], rec["chain"]
    bed, arm = split_chain(chain)
    ov = chain_overlay(cands, chain)
    kind = arm_kind(chain, ov)
    two = bool(rec.get("two_direction"))
    f = {"ref": ref, "chain": chain, "bed": bed, "arm": arm, "seed": int(rec.get("seed") or 0),
         "two_direction": two, "screen_chain": bool(rec.get("screen_chain")), **kind,
         "run_dir": rec.get("run_dir"), "notes": [], "recorded_at": rec.get("recorded_at") or "",
         # run_arm.sh writes a record for a killed run too: {"killed": <reason>, "killed_kind": wiring|dose|dead|
         # diverged, "killed_at": 200|1000|"live"}
         "killed": ({"kind": rec.get("killed_kind"), "at": rec.get("killed_at"), "reason": rec.get("killed")}
                    if rec.get("killed") else None)}
    if two:
        cj, aj = _load_json(rec.get("lr_class_json")), _load_json(rec.get("lr_along_json"))
        stamp = rec.get("lr_stamp")
    else:
        cj, aj = _load_json(rec.get("driver_class_json")), _load_json(rec.get("driver_along_json"))
        stamp = rec.get("driver_stamp")
    c = (cj or {}).get(ref) or {}
    a = (aj or {}).get(ref) if aj else None
    a = a or {}
    f["feet"] = {"l": c.get("class"), "r": c.get("class_r")}
    f["along"] = {"l": a.get("class_along"), "r": a.get("class_along_r")}
    f["hands"] = {"l": hand_class(c.get("class"), a.get("class_along"))}
    if two:
        f["hands"]["r"] = hand_class(c.get("class_r"), a.get("class_along_r"))
    if cj is None and not f["killed"]:
        f["notes"].append("no classifying class json")
    if aj is None and not f["killed"]:
        f["notes"].append("no along-command json: a feet-FAST hand reads NA (rerun along_cmd.py)")
    f["fall_share"] = fall_share(c) if c else None
    f["feet_2999"] = feet_hands_avg(c) if c else None
    c25 = ((_load_json(rec.get("lr2500_class_json")) or {}).get(f"{ref}_m2500")) or {}
    f["feet_2500"] = feet_hands_avg(c25) if c25 else None
    f.update(q_reads(stamp))
    odd = ((_load_json(rec.get("odd_rms_json")) or {}).get(ref)) or {}
    f["odd_l"], f["odd_r"] = odd.get("rod_odd_rms_l"), odd.get("rod_odd_rms_r")
    f["joint_odd_l"], f["joint_odd_r"] = odd.get("joint_odd_rms_l"), odd.get("joint_odd_rms_r")
    if with_tb and rec.get("run_dir") and Path(rec["run_dir"]).is_dir():
        ser = tb_series(Path(rec["run_dir"]))
        tl = Path(rec["train_log"]) if rec.get("train_log") else None
        f["wiring"] = (wiring_read(ser, kind, consumed_lines(tl, kind["new_knobs"]), dump_checks(Path(rec["run_dir"]), ov))
                       if kind["construction"] else None)
        f["dead"] = dead_read(ser) if kind["construction"] else None
        f["dose_1000"] = dose_at(ser, W_1000) if kind["bound"] or not kind["construction"] else None
        f["dose_final"] = dose_at(ser, W_FINAL) if kind["bound"] or not kind["construction"] else None
        f.update(lr_reads(ser))
        f["hand_cos_0_600"] = window(ser.get("hand_cos"), *W_HANDCOS)
        f["lock_300_600"] = _mean([window(ser.get("lock_cam"), *W_LOCK), window(ser.get("lock_rod"), *W_LOCK)])
        f["track_rsi_mean"] = window(ser.get("track_rsi"), 0, 10 ** 9)
        f["symmetry_final"] = window(ser.get("symmetry"), *W_FINAL)
        f["last_iter"] = ser.get("_last")
    return f


# ------------------------------------------------------------------------------------------- the rules
def tripwire_reference(bed_c0: list[dict], c_bed_c0: list[dict], two_direction: bool) -> tuple[float | None, list]:
    notes = []

    def med(runs, label):
        vals = [r["fall_share"] for r in runs if r.get("fall_share") is not None]
        kept = [v for v in vals if v <= TRIP_COLLAPSE]
        if len(kept) < len(vals):
            notes.append(f"{label}: {len(vals) - len(kept)} C0 run(s) above {TRIP_COLLAPSE} excluded")
        return _median(kept)

    ref = med(bed_c0, "bed C0")
    if two_direction:
        cref = med(c_bed_c0, "C-bed C0")
        cands = [x for x in (ref, cref) if x is not None]
        ref = max(cands) if cands else None
    return ref, notes


def trip_threshold(ref: float | None) -> float:
    return max(TRIP_FLOOR, ref + TRIP_MARGIN) if ref is not None else TRIP_FLOOR


def validity(f: dict) -> str | None:
    """Why a run is EXCLUDED from every count (None = it counts). Mutates nothing.

    * killed for wiring -> rerun required (the rerun, a fresh reference of the same chain + seed, counts);
    * killed for dose -> the CHARTER 4.3 stop (wiring check; wiring right = stop and ask);
    * a post-hoc wiring read that failed (the live kill was skipped: AUTO_KILL=0 or a missed poll) -> rerun required;
    * a hand read NA on a run that was not killed -> along/feet read missing (rerun the read, not the run).
    A DEAD or DIVERGED kill is a VALID not-FAST run.
    """
    k = f.get("killed") or {}
    if k.get("kind") == "wiring":
        return f"killed for wiring at it {k.get('at')}: rerun required (a fresh reference, named in CHARTER 9)"
    if k.get("kind") == "dose" and _dose_superseded(f):
        return (f"killed for dose at it {k.get('at')} under the pre-amendment rule; passes the amended median rule "
                "(CHARTER 4.3, 2026-10-01): superseded -- its rerun under a fresh reference counts, not a stop")
    if k.get("kind") == "dose":
        return f"killed for dose at it {k.get('at')}: CHARTER 4.3 stop (check wiring; wiring right = stop and ask)"
    if k.get("kind") not in (None, "dead", "diverged"):
        return f"killed ({k.get('kind')!r}, unknown kind): treat as rerun required"
    if f.get("wiring") and f["wiring"].get("ok") is False and not k:
        return "post-hoc WIRING FAIL at it 200 (" + "; ".join(f["wiring"]["fails"]) + "): rerun required"
    if not k and any(f["hands"].get(h) == NA for h in (("l", "r") if f["two_direction"] else ("l",))):
        return "along read missing (or no class json): rerun along_cmd.py / side22_classify.py on the stamp"
    return None


def apply_run_class(f: dict, thr: float) -> dict:
    """Run FAST (hands) and the tripwire (construction runs). Mutates and returns ``f``."""
    hands = f["hands"]
    hands_fast = all(hands.get(h) == "FAST" for h in (("l", "r") if f["two_direction"] else ("l",)))
    f["trip_threshold"] = thr
    f["tripped"] = bool(f["construction"] and f.get("fall_share") is not None and f["fall_share"] > thr)
    f["hands_fast"] = hands_fast
    f["fast"] = bool(hands_fast and not f["tripped"] and not f.get("killed"))
    f.setdefault("excluded", validity(f))
    flags = []
    if f.get("killed"):
        flags.append(f"KILLED {f['killed'].get('kind')} at it {f['killed'].get('at')}: {f['killed'].get('reason')}")
    if f.get("excluded"):
        flags.append("EXCLUDED: " + f["excluded"])
    if f.get("dose_1000") and f["dose_1000"].get("ok") is False and f["bound"]:
        flags.append("DOSE FAIL at it 1000 (kill + wiring check; a wiring-right fail is a true stop)")
    if f.get("dose_final") and f["dose_final"].get("ok") is False and f["bound"] and \
            not (f.get("dose_1000") and f["dose_1000"].get("ok") is False):
        flags.append("dose lost late (final window); the class still counts")
    if f.get("dead") and f["dead"].get("dead"):
        flags.append("DEAD at it 1000")
    if f.get("wiring") and f["wiring"].get("ok") is False:
        flags.append("WIRING FAIL at it 200: " + "; ".join(f["wiring"]["fails"]))
    f["flags"] = flags
    return f


def two_dir_arm_verdict(arm_runs: list[dict], c0_runs: list[dict]) -> dict:
    """CHARTER 4.5 for one non-C0 arm on one two-direction bed. Runs carry ``fast``, ``hands``, ``feet_2500``,
    ``feet_2999``. Uses seeds 11-15 (the screen + extension); later seeds are ignored here."""
    A = sorted([r for r in arm_runs if 11 <= r["seed"] <= 15], key=lambda r: r["seed"])
    C0 = sorted([r for r in c0_runs if 11 <= r["seed"] <= 15], key=lambda r: r["seed"])
    nA, kA = len(A), sum(r["fast"] for r in A)
    nC, kC = len(C0), sum(r["fast"] for r in C0)
    out = {"n_arm": nA, "k_arm": kA, "n_c0": nC, "k_c0": kC, "reasons": [], "extend": [], "provisional": False}
    # extension recommendations (seeds 14-15 not yet run)
    if kC >= 2:
        if nC < 5:
            out["extend"].append("C0 only to seeds 14-15 (VOID bed; arm seeds 14-15 are the user's call, FYI)")
    elif kC == 1 and (nC < 5 or nA < 5):
        out["extend"].append("every arm + C0 to seeds 14-15 (a single C0 FAST run; a note, not VOID)")
    elif kA >= 1 and (nA < 5 or nC < 5):
        out["extend"].append("this arm + C0 to seeds 14-15 (an arm run FAST)")
    out["provisional"] = bool(out["extend"])
    # C0 late-rise reference
    c0_rises = [r["feet_2999"] - r["feet_2500"] for r in C0 if r.get("feet_2999") is not None and r.get("feet_2500") is not None]
    rise_thr = max([LATE_RISE_MIN] + c0_rises)
    out["late_rise_threshold"] = rise_thr
    if not c0_rises:
        out["reasons"].append(f"NOTE: no C0 model_2500 read on this bed -- the late-rise threshold fell to {LATE_RISE_MIN}")
    # VOID
    if kC >= 2:
        out["verdict"] = "VOID"
        out["reasons"].insert(0, f"C0 FAST {kC}/{nC}: bed not reliably SLOW at fresh seeds; arm {kA}/{nA} descriptive")
        return out
    # UNLOCK
    if nA >= 5 and nC >= 5 and ((kA == 4 and kC == 0) or (kA == 5 and kC <= 1)):
        out["verdict"] = "UNLOCK"
        out["reasons"].insert(0, f"arm {kA}/{nA} vs C0 {kC}/{nC} (Fisher two-sided p <= 0.048)")
        return out
    # PARTIAL
    partial = []
    if nA >= 5 and 2 <= kA <= 4:
        partial.append(f"arm {kA}/{nA} vs C0 {kC}/{nC} does not meet UNLOCK")
    for r in A:
        hs = [h for h in ("l", "r") if r["hands"].get(h) == "INTERMEDIATE"]
        if hs:
            partial.append(f"{r['ref']}: hand(s) {','.join(hs).upper()} INTERMEDIATE")
        if not r["fast"] and any(r["hands"].get(h) == "FAST" for h in ("l", "r")):
            why = "tripwire" if r.get("tripped") and r.get("hands_fast") else "the other hand not FAST"
            partial.append(f"{r['ref']}: a FAST hand in a run not counted FAST ({why})")
        if not r["fast"] and r.get("feet_2999") is not None and r.get("feet_2500") is not None:
            rise = r["feet_2999"] - r["feet_2500"]
            if rise > rise_thr:
                partial.append(f"{r['ref']}: late rise {rise:+.3f} m/s model_2500 -> 2999 > {rise_thr:.3f}")
    if partial:
        out["verdict"] = "PARTIAL"
        out["reasons"] = partial + out["reasons"]
        out["reading"] = "inconclusive (never 'not the limiter'); no 4c; numbers to the user as FYI"
        return out
    # NO-FLIP (never while an extension is pending: then the extension decides -> PENDING below)
    if not out["extend"] and ((nA == 3 and kA == 0 and nC >= 3) or (nA >= 5 and kA <= 1 and nC >= 5)):
        out["verdict"] = "NO-FLIP"
        out["reasons"].insert(0, f"arm {kA}/{nA} vs C0 {kC}/{nC}: no reliable flip at this dose"
                                 + (f" (FAST: {', '.join(r['ref'] for r in A if r['fast'])})" if kA else ""))
        return out
    if nA < 3 or nC < 3:
        out["verdict"] = "INCOMPLETE"
        out["reasons"].insert(0, f"arm {nA} / C0 {nC} seeds of the 3 the screen needs")
    else:
        out["verdict"] = "PENDING"
        out["reasons"].insert(0, f"arm {kA}/{nA} vs C0 {kC}/{nC}: the extension decides"
                                 + (f" ({'; '.join(out['extend'])})" if out["extend"] else ""))
    return out


def c_guard(runs_by_arm: dict) -> dict:
    """CHARTER 4.7 on the C bed: PASS / FAIL / UNDEFINED (+ B and the bisection when FAIL)."""
    def counts(arm, seeds=range(11, 16)):
        rs = [r for r in runs_by_arm.get(arm, []) if r["seed"] in seeds]
        return len(rs), sum(r["fast"] for r in rs)

    nA, kA = counts("A")
    nC, kC = counts("C0")
    out = {"n_A": nA, "k_A": kA, "n_C0": nC, "k_C0": kC, "next": []}
    if nA < 3 or nC < 3:
        out["guard"] = "PENDING"
        out["why"] = f"A {kA}/{nA}, C0 {kC}/{nC}: three seeds each needed"
        return out
    # integer arithmetic: k/n >= 2/3 <=> 3k >= 2n; k/n <= 1/3 <=> 3k <= n
    if 3 * kA >= 2 * nA:
        out["guard"] = "PASS"
        out["why"] = f"A FAST {kA}/{nA} >= 2/3" + (" (C0 itself not reliably FAST: noted)" if 3 * kC <= nC else "")
        return out
    if 3 * kC >= 2 * nC and 3 * kA > nA:
        out["guard"] = "UNDEFINED"
        out["why"] = (f"A FAST {kA}/{nA} is between the bands (PASS needs >= 2/3, FAIL <= 1/3) with C0 {kC}/{nC}: "
                      "the guard does not decide (no B run, no bisection); reported to the user")
        return out
    if 3 * kC >= 2 * nC:
        out["guard"] = "FAIL"
        out["why"] = f"A FAST {kA}/{nA} <= 1/3 with C0 {kC}/{nC} >= 2/3"
        nB, kB = counts("B")
        out["B"] = {"n": nB, "k": kB}
        if nB < 3:
            out["next"].append("run C__B at seeds 11-13")
            return out
        if kB / nB >= 2 / 3:
            out["B"]["recovers"] = True
            out["next"].append("B recovers the stroke on C (bisection not needed)")
            return out
        out["B"]["recovers"] = False
        out["bisection"] = []
        for chain in BISECTION:
            arm = split_chain(chain)[1]
            n, k = counts(arm, seeds=(11, 12))
            st = "pending" if n < 2 else ("RECOVERS" if k == 2 else "does not recover")
            out["bisection"].append({"chain": chain, "n": n, "k": k, "status": st})
            if st == "pending":
                out["next"].append(f"run {chain} at seeds 11-12")
                break
            if st == "RECOVERS":
                out["next"].append(f"{chain} recovers C0's class: the bisection stops here")
                break
        else:
            out["next"].append("no bisection variant recovers C0's class")
        out["note"] = "two-direction A/B continue unchanged (user decision); 4c needs the user's approval"
        return out
    out["guard"] = "UNDEFINED"
    out["why"] = f"A {kA}/{nA}, C0 {kC}/{nC} <= 1/3: C bed not reliable at fresh seeds; the guard does not decide"
    return out


def q_better(arm_runs: list[dict], c0_runs: list[dict]) -> dict:
    a = [r["jitter"] for r in arm_runs if r.get("jitter") is not None]
    c = [r["jitter"] for r in c0_runs if r.get("jitter") is not None]
    af = [r["fft"] for r in arm_runs if r.get("fft") is not None]
    cf = [r["fft"] for r in c0_runs if r.get("fft") is not None]
    out = {"n_arm": len(a), "n_c0": len(c), "jitter_arm_max": max(a) if a else None,
           "jitter_c0_min": min(c) if c else None, "fft_arm_mean": _mean(af), "fft_c0_mean": _mean(cf)}
    if len(a) < 3 or len(c) < 3 or len(af) < 3 or len(cf) < 3:
        out["better"] = None
        return out
    out["better"] = bool(max(a) < min(c) and _mean(af) <= _mean(cf))
    return out


def lr_confound(arm_runs: list[dict], c0_runs: list[dict]) -> dict:
    c0_med = _median([r.get("lr_applied_mean") for r in c0_runs])
    rows, trig = [], []
    for r in arm_runs:
        fs, lm = r.get("floor_share"), r.get("lr_applied_mean")
        t1 = fs is not None and fs > LR_FLOOR_SHARE
        t2 = lm is not None and c0_med is not None and lm < LR_REL * c0_med
        rows.append({"ref": r["ref"], "floor_share": fs, "lr_applied_mean": lm, "floor": t1, "relative": t2})
        if t1 or t2:
            trig.append(r["ref"])
    return {"c0_lr_applied_median": c0_med, "runs": rows, "triggered_by": trig, "triggers_4a": bool(trig)}


def _dose_superseded(r) -> bool:
    """A dose kill made under the pre-amendment rule (the |mu| > 1.5 share) whose own iteration-1000 read passes the
    amended median rule (2026-10-01 12:30 user decision, CHARTER 4.3 and 9): the kill is superseded -- the run stays
    excluded (it has no class; its rerun under a fresh reference counts) but it is NOT a 4.3 stop. Needs the
    tfevents read: with --no-tb (dose_1000 unread) a dose kill stays a stop (fail closed)."""
    k = (r or {}).get("killed") or {}
    return k.get("kind") == "dose" and ((r.get("dose_1000") or {}).get("ok") is True)


def _dose_killed(r) -> bool:
    return bool(r and (r.get("killed") or {}).get("kind") == "dose" and not _dose_superseded(r))


def pilot(by_chain: dict, by_chain_all: dict | None = None) -> dict:
    """Seed 11 of the pilot chains. ``by_chain`` holds the VALID runs; ``by_chain_all`` (default: the same) also
    the excluded ones, so a DOSE-killed A run reads DOSE, not "missing"."""
    by_chain_all = by_chain if by_chain_all is None else by_chain_all

    def get(chain):
        return next((r for r in by_chain.get(chain, []) if r["seed"] == 11), None)

    def dose_kill(chain):
        return next((r for r in by_chain_all.get(chain, []) if r["seed"] == 11 and _dose_killed(r)), None)

    killed = {c: dose_kill(c) for c in ("C__A", "C_lr__A")}
    if any(killed.values()):
        return {"status": "DOSE", "dose_killed": [r["ref"] for r in killed.values() if r],
                "next": "CHARTER 4.3: check wiring; wiring wrong -> fix and rerun (fresh reference); wiring right -> "
                        "'bound did not bind at this coefficient', stop and ask"}
    cc0, ca, lc0, la = get("C__C0"), get("C__A"), get("C_lr__C0"), get("C_lr__A")
    if not all((cc0, ca, lc0, la)):
        return {"status": "PENDING", "missing": [n for n, r in (("C__C0", cc0), ("C__A", ca), ("C_lr__C0", lc0),
                                                                   ("C_lr__A", la)) if r is None]}
    dose_ok = [r.get("dose_1000", {}) and r["dose_1000"].get("ok") for r in (ca, la)]
    out = {"dose_1000_ok": {"C__A": dose_ok[0], "C_lr__A": dose_ok[1]},
           "C__C0_fast": cc0["fast"], "C__A_fast": ca["fast"], "C_lr__C0_fast": lc0["fast"], "C_lr__A_fast": la["fast"]}
    if not all(d is True for d in dose_ok):
        out["status"] = "DOSE"
        out["next"] = "CHARTER 4.3: kill/check wiring; wiring right -> 'bound did not bind at this coefficient', stop and ask"
        return out
    if cc0["fast"] and not ca["fast"]:
        s12 = {c: next((r for r in by_chain.get(c, []) if r["seed"] == 12), None) for c in ("C__C0", "C__A")}
        if s12["C__C0"] is None or s12["C__A"] is None:
            out["status"] = "RETRY"
            out["next"] = "run C__A and C__C0 at seed 12"
        elif s12["C__C0"]["fast"] and not s12["C__A"]["fast"]:
            out["status"] = "GUARD-FAIL"
            out["next"] = "C-bed guard-fail branch (CHARTER 4.7) before any further two-direction A/B run"
        elif not s12["C__C0"]["fast"]:
            out["status"] = "C-UNRELIABLE"
            out["next"] = "record 'C bed not reliable (provisional)' and continue the blocks"
        else:
            out["status"] = "PASS"
            out["next"] = "C__A FAST at seed 12: continue the blocks"
        return out
    if not cc0["fast"]:
        out["status"] = "PASS (C0 not FAST at s11)"
        out["next"] = "continue the blocks; the C-bed reliability is read at three seeds"
        return out
    out["status"] = "PASS"
    out["next"] = "continue the blocks"
    return out


def step4_vs_control(arm_runs: list[dict], control_runs: list[dict]) -> dict:
    """Plan steps 10/11: per seed, the step-4 arm vs the step-3 run of its head at the same seed."""
    ctrl = {r["seed"]: r for r in control_runs}
    rows, flips = [], []
    for r in sorted(arm_runs, key=lambda x: x["seed"]):
        c = ctrl.get(r["seed"])
        row = {"seed": r["seed"], "ref": r["ref"], "fast": r["fast"], "control_ref": c["ref"] if c else None,
               "control_fast": c["fast"] if c else None}
        if c is not None and r["fast"] and not c["fast"]:
            flips.append(r["seed"])
        rows.append(row)
    return {"rows": rows, "flips": flips, "missing_controls": [x["seed"] for x in rows if x["control_ref"] is None]}


def inc0_rule(inc0_runs: list[dict], b_runs: list[dict]) -> dict:
    """Plan step 12 (1): INC0 (incumbent + action_rate 0) attributes a B flip, seeds 11-15."""
    I = [r for r in inc0_runs if 11 <= r["seed"] <= 15]
    B = [r for r in b_runs if 11 <= r["seed"] <= 15]
    nI, kI, nB, kB = len(I), sum(r["fast"] for r in I), len(B), sum(r["fast"] for r in B)
    out = {"n_inc0": nI, "k_inc0": kI, "n_b": nB, "k_b": kB}
    if nI < 5 or nB < 5:
        out["status"] = "PENDING"
        out["why"] = f"INC0 {kI}/{nI}, B {kB}/{nB}: five seeds each needed"
    elif kI >= kB - 1:
        out["status"] = "STOP-AND-ASK"
        out["why"] = f"INC0 FAST {kI}/5 >= B {kB}/5 - 1: the flip belongs to action_rate, not the construction"
    else:
        out["status"] = "B-PACKAGE" if kI > 1 else "B-CONSTRUCTION"
        out["why"] = (f"INC0 {kI}/5 vs B {kB}/5: FB/P1 are " + ("'the B package (construction + action_rate 0) vs C0'"
                      if kI > 1 else "B vs C0 (INC0 <= 1/5)"))
    return out


def _mark_duplicates(facts: list[dict]) -> None:
    """Two VALID runs of one chain + seed: the first recorded counts, the later one is a re-roll (D18).
    A valid run sharing chain + seed with an excluded one is noted as its replacement."""
    groups: dict = {}
    for f in facts:
        groups.setdefault((f["chain"], f["seed"]), []).append(f)
    for runs in groups.values():
        valid = sorted([f for f in runs if not f.get("excluded")], key=lambda f: (f.get("recorded_at") or "", f["ref"]))
        for later in valid[1:]:
            later["excluded"] = (f"duplicate of {valid[0]['ref']} (same chain + seed): a re-roll (D18); "
                                 "the first valid run counts")
        bad = [f["ref"] for f in runs if f.get("excluded") and all(f is not x for x in valid[1:])]
        if valid and bad:
            valid[0]["notes"].append(f"replaces excluded run(s) {', '.join(bad)} (name it in CHARTER 9)")


def evaluate(facts: list[dict]) -> dict:
    """Per-run classes (with the tripwire), per-bed verdicts, guard, pilot, LR confound, Q.

    Excluded runs (:func:`validity`, duplicates) are listed but never counted -- not in n, k, the tripwire
    reference, Q or the LR confound.
    """
    for f in facts:
        if not f.get("excluded"):
            f["excluded"] = validity(f)
    _mark_duplicates(facts)
    valid = [f for f in facts if not f.get("excluded")]
    by_chain: dict = {}
    by_chain_all: dict = {}
    for f in facts:
        by_chain_all.setdefault(f["chain"], []).append(f)
    for f in valid:
        by_chain.setdefault(f["chain"], []).append(f)
    c_bed_c0 = by_chain.get("C__C0", [])
    # tripwire references (valid C0 runs) and run classes (every run, so excluded ones still show a class)
    beds = sorted({f["bed"] for f in facts})
    trip = {}
    for bed in beds:
        bed_runs = [f for f in facts if f["bed"] == bed]
        two = any(f["two_direction"] for f in bed_runs)
        c0 = by_chain.get(f"{bed}__C0", [])
        ref, notes = tripwire_reference(c0, c_bed_c0, two)
        thr = trip_threshold(ref)
        trip[bed] = {"reference": ref, "threshold": thr, "notes": notes + ([] if ref is not None else
                                                                          ["no C0 fall reference: threshold 0.15"])}
        for f in bed_runs:
            apply_run_class(f, thr)
    out = {"runs": facts, "tripwire": trip, "beds": {},
           "excluded": [{"ref": f["ref"], "why": f["excluded"]} for f in facts if f.get("excluded")]}
    # CHARTER 4.3 stops: a dose kill, or a dose failure at it 1000 on a run that was not killed (AUTO_KILL=0)
    out["stops"] = [f"{f['ref']}: dose kill at it 1000 -- check wiring; wiring right = stop and ask" for f in facts
                    if _dose_killed(f)] + \
                   [f"{f['ref']}: DOSE FAIL at it 1000 (not killed) -- check wiring; wiring right = stop and ask"
                    for f in facts if not f.get("killed") and f["bound"] and (f.get("dose_1000") or {}).get("ok") is False]
    # C bed
    c_arms: dict = {}
    for f in valid:
        if f["bed"] == "C":
            c_arms.setdefault(f["arm"], []).append(f)
    guard = c_guard(c_arms) if c_arms else {"guard": "PENDING", "why": "no C-bed runs"}
    out["c_guard"] = guard
    # two-direction beds
    for bed in beds:
        if bed not in TWO_DIR_BEDS:
            continue
        c0 = by_chain.get(f"{bed}__C0", [])
        arms = sorted({f["arm"] for f in facts if f["bed"] == bed and f["arm"] != "C0"})
        res = {}
        for arm in arms:
            runs = by_chain.get(f"{bed}__{arm}", [])
            if arm == INC0:
                res[arm] = {"verdict": "INC0", "step4": True, "inc0": inc0_rule(runs, by_chain.get(f"{bed}__B", [])),
                            "k_arm": sum(r["fast"] for r in runs), "n_arm": len(runs),
                            "k_c0": sum(r["fast"] for r in c0), "n_c0": len(c0), "reasons": [], "extend": [],
                            "q": q_better(runs, c0), "lr_floor": lr_confound([], c0), "tripped": [],
                            "adopt_for_motion_quality": False}
                res[arm]["reasons"].append(res[arm]["inc0"]["why"])
                continue
            v = two_dir_arm_verdict(runs, c0)
            head = step4_control(arm)
            v["step4"] = head is not None
            if head is not None:
                vc = step4_vs_control(runs, by_chain.get(f"{bed}__{head}", []))
                v["control"] = {"chain": f"{bed}__{head}", **vc}
                n5 = len([r for r in runs if 11 <= r["seed"] <= 15]) >= 5 and len([r for r in c0 if 11 <= r["seed"] <= 15]) >= 5
                v["extend"] = ([f"this arm + C0 to seeds 14-15 (FAST where its control was not: seeds {vc['flips']})"]
                               if vc["flips"] and not n5 else [])
                v["provisional"] = bool(v["extend"])
                if v["verdict"] == "VOID":
                    pass
                elif n5:
                    v["verdict"] = "UNLOCK" if v["verdict"] == "UNLOCK" else f"NO UNLOCK ({v['verdict']})"
                else:
                    v["verdict"] = "DESCRIPTIVE"
                    v["reasons"] = ([f"arm {v['k_arm']}/{v['n_arm']} vs C0 {v['k_c0']}/{v['n_c0']} (descriptive: a step-4 "
                                     "arm is judged against its control below)"]
                                    + [r for r in v["reasons"] if r.startswith("NOTE")])
                v["reading"] = ("step-4 arm vs its control (the step-3 run of the head, same seeds): descriptive; a FAST "
                                "where the control was not triggers the extension and, at five seeds, the UNLOCK rule")
            v["q"] = q_better(runs, c0)
            v["lr_floor"] = lr_confound([r for r in runs if r["construction"]] if arm in STEP3_ARMS else [], c0)
            v["tripped"] = [r["ref"] for r in runs if r.get("tripped")]
            v["adopt_for_motion_quality"] = bool(v["q"].get("better") and guard.get("guard") == "PASS" and not v["tripped"]
                                                 and arm in STEP3_ARMS)
            m, h = tci90([r.get("feet_2999") for r in runs])
            mc, hc = tci90([r.get("feet_2999") for r in c0])
            v["descriptive"] = {"feet_ss_mean_ci90": [m, h], "c0_feet_ss_mean_ci90": [mc, hc],
                                "odd_rms_mean_ci90": list(tci90([_mean([r.get("odd_l"), r.get("odd_r")]) for r in runs])),
                                "c0_odd_rms_mean_ci90": list(tci90([_mean([r.get("odd_l"), r.get("odd_r")]) for r in c0]))}
            res[arm] = v
        out["beds"][bed] = res
    # C-bed arms: Q and LR confound too (the guard is the verdict)
    if c_arms:
        res = {}
        for arm, runs in sorted(c_arms.items()):
            if arm == "C0":
                continue
            q = q_better(runs, c_arms.get("C0", []))
            res[arm] = {"q": q, "lr_floor": lr_confound([r for r in runs if r["construction"]] if arm in STEP3_ARMS else [],
                                                        c_arms.get("C0", [])),
                        "tripped": [r["ref"] for r in runs if r.get("tripped")]}
            res[arm]["adopt_for_motion_quality"] = bool(q.get("better") and guard.get("guard") == "PASS"
                                                        and not res[arm]["tripped"] and arm in STEP3_ARMS)
        out["beds"]["C"] = res
    out["pilot"] = pilot(by_chain, by_chain_all)
    out["triggers_4a"] = sorted({ref for bed in out["beds"].values() for v in bed.values()
                                 for ref in v["lr_floor"]["triggered_by"]})
    # 4c triggers: step-3 A/B UNLOCKs only. Step-4 (4a/4b) UNLOCKs are listed apart (plan step 12 carries them
    # into 4c with their own carry rule); INC0 is never an UNLOCK.
    out["unlock"] = [f"{bed}__{arm}" for bed, arms in out["beds"].items() for arm, v in arms.items()
                     if v.get("verdict") == "UNLOCK" and arm in STEP3_ARMS]
    out["unlock_step4"] = [f"{bed}__{arm}" for bed, arms in out["beds"].items() for arm, v in arms.items()
                           if v.get("verdict") == "UNLOCK" and arm not in STEP3_ARMS and arm != INC0]
    out["inc0"] = {bed: arms[INC0]["inc0"] for bed, arms in out["beds"].items() if INC0 in arms}
    return out


# ------------------------------------------------------------------------------------------- output
def _f(x, fmt="{:.3f}"):
    if x is None:
        return "--"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    return fmt.format(x) if isinstance(x, (int, float)) else str(x)


def markdown(res: dict) -> str:
    L = ["# PPO construction -- verdict support (scripts/verdict.py)", ""]
    L += ["## Runs", "",
          "| ref | bed | arm | seed | hand L | hand R | run | fall share | trip thr | tripped | dose 1000 | dose final "
          "| DEAD | wiring | LR floor share | lr_applied | feet 2500 -> 2999 | jitter | fft | rod odd L/R | flags |",
          "|" + "---|" * 21]
    for f in sorted(res["runs"], key=lambda r: (r["bed"], r["arm"], r["seed"])):
        d1, dF = f.get("dose_1000") or {}, f.get("dose_final") or {}
        w = f.get("wiring") or {}
        L.append(f"| {f['ref']} | {f['bed']} | {f['arm']} | {f['seed']} | {f['hands'].get('l')} | {f['hands'].get('r', '')} "
                 f"| {'FAST' if f['fast'] else 'not FAST'} | {_f(f.get('fall_share'))} | {_f(f.get('trip_threshold'))} "
                 f"| {'YES' if f.get('tripped') else ''} | {_f(d1.get('ok'))} | {_f(dF.get('ok'))} "
                 f"| {_f((f.get('dead') or {}).get('dead'))} | {_f(w.get('ok'))} | {_f(f.get('floor_share'), '{:.2f}')} "
                 f"| {_f(f.get('lr_applied_mean'), '{:.2e}')} | {_f(f.get('feet_2500'))} -> {_f(f.get('feet_2999'))} "
                 f"| {_f(f.get('jitter'))} | {_f(f.get('fft'), '{:.2f}')} | {_f(f.get('odd_l'), '{:.2f}')}/{_f(f.get('odd_r'), '{:.2f}')} "
                 f"| {'; '.join(f.get('flags') or [])} |")
    if res.get("excluded"):
        L += ["", "## Excluded runs (never counted)", ""] + [f"- {e['ref']}: {e['why']}" for e in res["excluded"]]
    if res.get("stops"):
        L += ["", "## STOPS (CHARTER 4.3: stop and ask)", ""] + [f"- **{x}**" for x in res["stops"]]
    L += ["", "## Tripwire", ""]
    for bed, t in res["tripwire"].items():
        L.append(f"- {bed}: reference {_f(t['reference'])}, trips above {_f(t['threshold'])}"
                 + (f" ({'; '.join(t['notes'])})" if t["notes"] else ""))
    L += ["", "## Two-direction beds (VOID > UNLOCK > PARTIAL > NO-FLIP)", "",
          "| bed | arm | verdict | arm FAST | C0 FAST | reasons | extend | Q better | adopt (MQ) | 4a trigger |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for bed, arms in res["beds"].items():
        if bed == "C":
            continue
        for arm, v in arms.items():
            if v.get("control"):
                ctl = v["control"]
                v = dict(v, reasons=v["reasons"] + [f"vs control {ctl['chain']}: FAST where the control was not at "
                                                    f"seeds {ctl['flips'] or 'none'}"
                                                    + (f"; no control at seeds {ctl['missing_controls']}"
                                                       if ctl["missing_controls"] else "")])
            L.append(f"| {bed} | {arm}{' (step 4)' if v.get('step4') else ''} | **{v['verdict']}**"
                     f"{' (provisional)' if v.get('provisional') else ''} | {v['k_arm']}/{v['n_arm']} | {v['k_c0']}/{v['n_c0']} "
                     f"| {'; '.join(v['reasons'])} | {'; '.join(v['extend'])} | {_f(v['q'].get('better'))} "
                     f"| {_f(v['adopt_for_motion_quality'])} | {', '.join(v['lr_floor']['triggered_by'])} |")
    g = res["c_guard"]
    L += ["", "## C-bed guard", "", f"- **{g.get('guard')}**: {g.get('why')}"]
    for n in g.get("next") or []:
        L.append(f"- next: {n}")
    for b in g.get("bisection") or []:
        L.append(f"- bisection {b['chain']}: {b['k']}/{b['n']} FAST -> {b['status']}")
    if "C" in res["beds"]:
        for arm, v in res["beds"]["C"].items():
            L.append(f"- C__{arm}: Q better {_f(v['q'].get('better'))}; adopt for motion quality "
                     f"{_f(v['adopt_for_motion_quality'])}; 4a trigger {', '.join(v['lr_floor']['triggered_by']) or '--'}")
    p = res["pilot"]
    L += ["", "## Pilot", "", f"- {p.get('status')}: {p.get('next', '')} {p.get('missing', '')}"]
    L += ["", "## Outcome", "",
          f"- UNLOCK (step 3, the 4c trigger): {', '.join(res['unlock']) or 'none'} (4c needs the C-bed guard PASS or "
          f"the user's approval: guard {g.get('guard')})",
          f"- UNLOCK of a step-4 arm (4c under plan step 12's carry rule; listed apart): "
          f"{', '.join(res.get('unlock_step4') or []) or 'none'}",
          "- INC0: " + ("; ".join(f"{b}: {i['status']} ({i['why']})" for b, i in (res.get("inc0") or {}).items())
                        or "not run"),
          f"- LR-floor confound (4a trigger): {', '.join(res['triggers_4a']) or 'none'}", ""]
    return "\n".join(L) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ------------------------------------------------------------------------------------------- self-test
def _run(ref, *, L="SLOW", R="SLOW", two=True, fall=0.05, f25=None, f29=None, jit=None, fft=None,
         construction=None, bound=None, dose1000=True, dosefinal=True, lr_floor=0.1, lr_mean=1e-4,
         chain=None, seed=None, killed=None, wiring_ok=True, recorded_at=""):
    c, _, s = ref.rpartition("_s")
    chain = chain or c
    bed, arm = split_chain(chain)
    cons = (arm not in ("C0", "INC0", "C0_1024")) if construction is None else construction
    f = {"ref": ref, "chain": chain, "bed": bed, "arm": arm, "seed": int(s) if seed is None else seed,
         "two_direction": two,
         "screen_chain": True, "construction": cons, "bound": cons if bound is None else bound,
         "sigma_check": cons, "new_knobs": [], "hands": {"l": L, **({"r": R} if two else {})},
         "fall_share": fall, "feet_2500": f25, "feet_2999": f29, "jitter": jit, "fft": fft,
         "dose_1000": {"ok": dose1000}, "dose_final": {"ok": dosefinal}, "dead": {"dead": False},
         "wiring": {"ok": wiring_ok, "fails": [] if wiring_ok else ["no [KNOB] X consumed line"]},
         "floor_share": lr_floor, "lr_applied_mean": lr_mean, "notes": [], "recorded_at": recorded_at,
         "killed": ({"kind": killed[0], "at": killed[1], "reason": f"{killed[0]} kill"} if killed else None)}
    return f


def _bed(bed, arm, fast_pattern, **kw):
    """Runs of one bed/arm at seeds 11.. from a pattern string like 'FFS' (F = both hands FAST)."""
    out = []
    for i, ch in enumerate(fast_pattern):
        h = {"F": "FAST", "S": "SLOW", "I": "INTERMEDIATE", "D": "DEAD"}[ch]
        out.append(_run(f"{bed}__{arm}_s{11 + i}", L=h, R=h, **kw))
    return out


def consumed_lines_text(text: str, knobs) -> dict:
    """:func:`consumed_lines` on a string (self-test helper)."""
    import tempfile  # noqa: PLC0415

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(text)
    try:
        return consumed_lines(Path(fh.name), knobs)
    finally:
        Path(fh.name).unlink()


def _check_run_codes() -> dict:
    """check_run's exit codes on synthetic telemetry (self-test helper; tb_series is swapped out)."""
    global tb_series  # noqa: PLW0603
    real = tb_series
    a_ov = {"KRABBY_ACTION_BOUND_LOSS": "0.001", "KRABBY_INIT_STD": "0.8", "KRABBY_STD_PARAM": "soft"}
    cands = {"chains": {"C_lr__A": {"stages": [{"overlay": a_ov}]},
                        "C_lr__A_fixed": {"stages": [{"overlay": dict(a_ov, KRABBY_PPO_SCHEDULE="fixed")}]},
                        "C_lr__C0": {"stages": [{"overlay": {"KRABBY_PPO_HAND_DIAG": "1"}}]}}}
    st = np.arange(0, 1100)

    def ser(track=0.6, mu=1.2, last=1099, std=0.8, kl=0.01):
        n = last + 1
        s_ = np.arange(0, n)
        d = {"track": (s_, np.full(n, track)), "effective_std": (s_, np.full(n, std)),
             "action_bound": (s_, np.full(n, 0.01)), "kl_mean": (s_, np.full(n, kl))}
        if mu is not None:
            d.update({k: (s_, np.full(n, v)) for k, v in (("mu_med_cam", mu), ("mu_med_rod", mu),
                                                          ("sat15_cam", 0.0), ("sat15_rod", 0.0))})
        d["_last"] = last
        return d

    cases = {"wiring": ("C_lr__A", 200, ser(std=1.5)), "dose+dead": ("C_lr__A", 1000, ser(track=0.2, mu=3.0)),
             "dead": ("C_lr__A", 1000, ser(track=0.2)), "pass": ("C_lr__A", 1000, ser()),
             "diverged": ("C_lr__A_fixed", "live", ser(kl=0.5)),
             "missing past grace": ("C_lr__A", 1000, ser(mu=None, last=1300)),
             "missing": ("C_lr__A", 1000, ser(mu=None, last=1060)), "C0 n/a": ("C_lr__C0", 1000, ser(track=0.1)),
             "live on a non-4a arm": ("C_lr__A", "live", ser(kl=0.5))}
    out = {}
    try:
        for name, (chain, at, sr) in cases.items():
            tb_series = (lambda _rd, _tags=None, _s=sr: _s)  # noqa: E731
            out[name] = check_run(Path("/nonexistent"), chain, at, None, cands)[0]
    finally:
        tb_series = real
    del st
    return out


def self_test() -> int:
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
        if not ok:
            fails.append(name)

    cc0 = _bed("C", "C0", "FFF", two=False, fall=0.06)

    def verdict_of(arm_pat, c0_pat, extra=None, arm="A", **kw):
        facts = cc0 + _bed("C_lr", "C0", c0_pat, **kw.get("c0kw", {})) + _bed("C_lr", arm, arm_pat, **kw.get("akw", {}))
        if extra:
            facts += extra
        return evaluate([dict(f) for f in facts])["beds"]["C_lr"][arm]

    print("two-direction beds")
    check("UNLOCK 4/5 vs 0/5", verdict_of("FFFFS", "SSSSS")["verdict"], "UNLOCK")
    check("UNLOCK 5/5 vs 1/5", verdict_of("FFFFF", "SSSSF")["verdict"], "UNLOCK")
    check("4/5 vs 1/5 -> PARTIAL", verdict_of("FFFFS", "SSSSF")["verdict"], "PARTIAL")
    check("3/5 vs 0/5 -> PARTIAL", verdict_of("FFFSS", "SSSSS")["verdict"], "PARTIAL")
    check("VOID at C0 2/3", verdict_of("FFF", "FFS")["verdict"], "VOID")
    v = verdict_of("SSS", "FFS")
    check("VOID at C0 2/3 extends C0 only", v["extend"], ["C0 only to seeds 14-15 (VOID bed; arm seeds 14-15 are the user's call, FYI)"])
    check("VOID at C0 2/5 (1/3 at three seeds)", verdict_of("FFFFS", "SFSSF")["verdict"], "VOID")
    check("C0 1/3 -> not VOID, extend every arm", verdict_of("SSS", "SFS")["extend"],
          ["every arm + C0 to seeds 14-15 (a single C0 FAST run; a note, not VOID)"])
    check("no NO-FLIP while an extension is pending (0/3 vs C0 1/3 -> PENDING)", verdict_of("SSS", "SFS")["verdict"],
          "PENDING")
    check("NO-FLIP 0/3 vs 0/3", verdict_of("SSS", "SSS")["verdict"], "NO-FLIP")
    check("NO-FLIP 1/5 vs 0/5", verdict_of("FSSSS", "SSSSS")["verdict"], "NO-FLIP")
    check("arm 1/3 -> PENDING + extend", (verdict_of("FSS", "SSS")["verdict"], verdict_of("FSS", "SSS")["extend"]),
          ("PENDING", ["this arm + C0 to seeds 14-15 (an arm run FAST)"]))
    check("2 seeds -> INCOMPLETE", verdict_of("SS", "SS")["verdict"], "INCOMPLETE")
    check("PARTIAL via an INTERMEDIATE hand", verdict_of("SIS", "SSS")["verdict"], "PARTIAL")
    one_hand = _bed("C_lr", "A", "SSS")
    one_hand[1]["hands"] = {"l": "FAST", "r": "SLOW"}
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + one_hand])["beds"]["C_lr"]["A"]
    check("PARTIAL via a FAST hand in a non-FAST run", (v["verdict"], "other hand" in " ".join(v["reasons"])), ("PARTIAL", True))
    c0r = _bed("C_lr", "C0", "SSS", f25=0.08, f29=0.10)                 # C0 rises 0.02
    ar = _bed("C_lr", "A", "SSS", f25=0.08, f29=0.08)
    ar[2]["feet_2500"], ar[2]["feet_2999"] = 0.07, 0.12                 # +0.05 > max(0.03, 0.02)
    v = evaluate([dict(f) for f in cc0 + c0r + ar])["beds"]["C_lr"]["A"]
    check("PARTIAL via late rise (+0.05 > max(0.03, C0 0.02))", (v["verdict"], v["late_rise_threshold"]), ("PARTIAL", 0.03))
    c0r2 = _bed("C_lr", "C0", "SSS", f25=0.08, f29=0.08)
    c0r2[0]["feet_2999"] = 0.14                                          # C0 rises 0.06
    v = evaluate([dict(f) for f in cc0 + c0r2 + ar])["beds"]["C_lr"]["A"]
    check("no late-rise PARTIAL when C0 rises more (0.06)", (v["verdict"], round(v["late_rise_threshold"], 3)), ("NO-FLIP", 0.06))
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + _bed("C_lr", "A", "SSS", f29=0.1)])["beds"]["C_lr"]["A"]
    check("missing C0 model_2500 is flagged", any("late-rise threshold fell" in r for r in v["reasons"]), True)

    print("survival tripwire")
    trip_runs = _bed("C_lr", "A", "FSS")
    trip_runs[0]["fall_share"] = 0.30
    res = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS", fall=0.08) + trip_runs])
    r0 = next(r for r in res["runs"] if r["ref"] == "C_lr__A_s11")
    check("tripwire threshold = max(0.15, max(bed C0 0.08, C-bed C0 0.06) + 0.10)", round(res["tripwire"]["C_lr"]["threshold"], 3), 0.18)
    check("tripped FAST-hands run counts not FAST", (r0["tripped"], r0["fast"]), (True, False))
    check("tripped run -> PARTIAL (FAST hand in a run not counted FAST)", res["beds"]["C_lr"]["A"]["verdict"], "PARTIAL")
    c0_collapse = _bed("C_lr", "C0", "SSS", fall=0.05)
    c0_collapse[0]["fall_share"] = 0.40
    res = evaluate([dict(f) for f in _bed("C", "C0", "FFF", two=False, fall=0.02) + c0_collapse + _bed("C_lr", "A", "SSS")])
    check("collapsed C0 (> 0.25) excluded; floor 0.15 binds", round(res["tripwire"]["C_lr"]["threshold"], 3), 0.15)
    res = evaluate([dict(f) for f in _bed("C", "C0", "FFF", two=False, fall=0.09) + _bed("C", "A", "FFF", two=False, fall=0.20)])
    check("C bed: reference = C0 median 0.09 -> thr 0.19; A at 0.20 trips", (round(res["tripwire"]["C"]["threshold"], 3),
          all(r["tripped"] for r in res["runs"] if r["arm"] == "A")), (0.19, True))
    check("C0 runs are never tripped", any(r["tripped"] for r in res["runs"] if r["arm"] == "C0"), False)

    print("C-bed guard")
    g = evaluate([dict(f) for f in cc0 + _bed("C", "A", "FFS", two=False)])["c_guard"]
    check("guard PASS at A 2/3", g["guard"], "PASS")
    g = evaluate([dict(f) for f in cc0 + _bed("C", "A", "FSS", two=False)])["c_guard"]
    check("guard FAIL at A 1/3, C0 3/3 -> run B", (g["guard"], g["next"]), ("FAIL", ["run C__B at seeds 11-13"]))
    facts = cc0 + _bed("C", "A", "SSS", two=False) + _bed("C", "B", "SSS", two=False) + \
        _bed("C", "A_ent01", "SS", two=False) + _bed("C", "A_std15", "FF", two=False)
    g = evaluate([dict(f) for f in facts])["c_guard"]
    check("bisection stops at the first that recovers", [b["status"] for b in g["bisection"]], ["does not recover", "RECOVERS"])
    g = evaluate([dict(f) for f in _bed("C", "C0", "SFS", two=False) + _bed("C", "A", "SSS", two=False)])["c_guard"]
    check("guard UNDEFINED at C0 1/3", g["guard"], "UNDEFINED")
    g = evaluate([dict(f) for f in _bed("C", "C0", "FFFFF", two=False) + _bed("C", "A", "FFSSS", two=False)])["c_guard"]
    check("guard UNDEFINED at A 2/5 with C0 5/5 (between the bands)", g["guard"], "UNDEFINED")
    g = evaluate([dict(f) for f in _bed("C", "C0", "FFFFF", two=False) + _bed("C", "A", "FSSSS", two=False)])["c_guard"]
    check("guard FAIL at A 1/5 with C0 5/5", g["guard"], "FAIL")

    print("dose / DEAD / wiring flags")
    fl = apply_run_class(_run("C_lr__A_s11", dose1000=False), 0.15)["flags"]
    check("dose fail at 1000 flagged", any("DOSE FAIL at it 1000" in x for x in fl), True)
    fl = apply_run_class(_run("C_lr__A_s11", L="FAST", R="FAST", dosefinal=False), 0.15)
    check("dose lost late flagged, class still counts", (any("dose lost late" in x for x in fl["flags"]), fl["fast"]), (True, True))
    ser = {"_last": 1100, "track": (np.arange(900, 1100), np.full(200, 0.35)),
           "effective_std": (np.arange(0, 1100), np.full(1100, 0.8))}
    check("DEAD at it 1000 (track 0.35 < 0.40)", dead_read(ser)["dead"], True)
    ser["track"] = (np.arange(900, 1100), np.full(200, 0.6))
    check("not DEAD at track 0.6, std 0.8", dead_read(ser)["dead"], False)
    kind = {"construction": True, "bound": True, "sigma_check": True, "new_knobs": ["KRABBY_ACTION_BOUND"]}
    ser200 = {"_last": 250, "effective_std": (np.arange(0, 250), np.full(250, 0.8)),
              "action_bound": (np.arange(0, 250), np.full(250, 0.01))}
    check("wiring PASS", wiring_read(ser200, kind, {"KRABBY_ACTION_BOUND": True}, {})["ok"], True)
    check("wiring FAIL: no consumed line", wiring_read(ser200, kind, {"KRABBY_ACTION_BOUND": False}, {})["ok"], False)
    ser_hi = dict(ser200, effective_std=(np.arange(0, 250), np.full(250, 1.5)))
    check("wiring FAIL: sigma 1.5 on an init-0.8 arm", wiring_read(ser_hi, kind, {}, {})["ok"], False)
    check("bisection arm: no sigma check", wiring_read(ser_hi, dict(kind, sigma_check=False), {}, {})["ok"], True)
    check("arm_kind: C__A_std15 skips the sigma check", arm_kind("C__A_std15", {"KRABBY_INIT_STD": "1.5"})["sigma_check"], False)
    check("arm_kind: C__A_ent01 skips the sigma check", arm_kind("C__A_ent01", {"KRABBY_INIT_STD": "0.8"})["sigma_check"], False)
    check("arm_kind: C_lr__A checks sigma", arm_kind("C_lr__A", {"KRABBY_INIT_STD": "0.8"})["sigma_check"], True)
    check("arm_kind: an explicit-off flag needs no consumed line",
          arm_kind("C_lr_rsi__A", {"KRABBY_RSI_CAM_STAGE": "0", "KRABBY_PPO_HAND_DIAG": "1",
                                   "KRABBY_INIT_STD": "0.8"})["new_knobs"], ["KRABBY_INIT_STD", "KRABBY_PPO_HAND_DIAG"])
    check("arm_kind: INC0 is not a construction arm", arm_kind("C_lr__INC0", {"KRABBY_ACTION_RATE_W": "0"})["construction"], False)
    check("wiring not ready before it 210", wiring_read({"_last": 150}, kind, {}, {})["ok"], None)
    check("dose read ok", dose_at({"mu_med_cam": (np.array([1000]), np.array([1.2])), "mu_med_rod": (np.array([1000]), np.array([1.4])),
                                   "sat15_cam": (np.array([1000]), np.array([0.05])), "sat15_rod": (np.array([1000]), np.array([0.1]))},
                                  W_1000)["ok"], True)
    check("dose read fails on mu_med 1.6", dose_at({"mu_med_cam": (np.array([1000]), np.array([1.6])), "mu_med_rod": (np.array([1000]), np.array([1.4])),
                                                   "sat15_cam": (np.array([1000]), np.array([0.05])), "sat15_rod": (np.array([1000]), np.array([0.1]))},
                                                  W_1000)["ok"], False)
    check("dose read ok with sat15 0.27/0.41 (amended 4.3: median only; the C__A_s11 kill case)",
          dose_at({"mu_med_cam": (np.array([1000]), np.array([0.973])), "mu_med_rod": (np.array([1000]), np.array([1.266])),
                   "sat15_cam": (np.array([1000]), np.array([0.274])), "sat15_rod": (np.array([1000]), np.array([0.411]))},
                  W_1000)["ok"], True)

    print("LR floor / Q / pilot")
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS", lr_mean=1e-4) + _bed("C_lr", "A", "SSS", lr_mean=4e-5)])["beds"]["C_lr"]["A"]
    check("LR relative trigger (4e-5 < 0.5 x 1e-4)", v["lr_floor"]["triggers_4a"], True)
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + _bed("C_lr", "A", "SSS", lr_floor=0.95)])["beds"]["C_lr"]["A"]
    check("LR floor-share trigger (0.95 > 0.9)", v["lr_floor"]["triggers_4a"], True)
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + _bed("C_lr", "A", "SSS")])["beds"]["C_lr"]["A"]
    check("no LR trigger", v["lr_floor"]["triggers_4a"], False)
    c0q = _bed("C", "C0", "FFF", two=False, jit=0.12, fft=1.8)
    c0q[0]["jitter"] = 0.11
    aq = _bed("C", "A", "FFF", two=False, jit=0.08, fft=1.7)
    res = evaluate([dict(f) for f in c0q + aq])
    check("Q better (all A jitter < all C0, fft not higher) + adopt proposal",
          (res["beds"]["C"]["A"]["q"]["better"], res["beds"]["C"]["A"]["adopt_for_motion_quality"]), (True, True))
    aq2 = _bed("C", "A", "FFF", two=False, jit=0.08, fft=1.9)
    check("Q not better when fft is higher", evaluate([dict(f) for f in c0q + aq2])["beds"]["C"]["A"]["q"]["better"], False)
    aq3 = _bed("C", "A", "FFF", two=False, jit=0.08, fft=1.7)
    aq3[2]["jitter"] = 0.115
    check("Q not better when one A seed overlaps C0", evaluate([dict(f) for f in c0q + aq3])["beds"]["C"]["A"]["q"]["better"], False)
    pil = _bed("C", "C0", "F", two=False) + _bed("C", "A", "F", two=False) + _bed("C_lr", "C0", "S") + _bed("C_lr", "A", "S")
    check("pilot PASS", evaluate([dict(f) for f in pil])["pilot"]["status"], "PASS")
    pil2 = _bed("C", "C0", "F", two=False) + _bed("C", "A", "S", two=False) + _bed("C_lr", "C0", "S") + _bed("C_lr", "A", "S")
    check("pilot: C__A not FAST while C0 FAST -> seed-12 retry", evaluate([dict(f) for f in pil2])["pilot"]["status"], "RETRY")
    pil3 = pil2 + [_run("C__C0_s12", L="FAST", two=False), _run("C__A_s12", L="SLOW", two=False)]
    check("pilot: fails again -> guard-fail branch", evaluate([dict(f) for f in pil3])["pilot"]["status"], "GUARD-FAIL")
    pil4 = _bed("C", "C0", "F", two=False) + _bed("C", "A", "F", two=False, dose1000=False) + _bed("C_lr", "C0", "S") + _bed("C_lr", "A", "S")
    check("pilot: dose failure -> DOSE", evaluate([dict(f) for f in pil4])["pilot"]["status"], "DOSE")

    print("stamp reads")
    cj = {"labels": {"lat_l_022": {"n": 90, "excluded": {"short": 8, "static": 2}, "feet_ss": 0.2},
                     "lat_l_025": {"n": 95, "excluded": {"short": 5}, "feet_ss": 0.22},
                     "stand": {"n": 100, "excluded": {"short": 50}}}}
    check("fall share pooled over lat_* only", round(fall_share(cj), 4), round(13 / 200, 4))
    check("hands-averaged feet_ss", round(feet_hands_avg(cj), 4), 0.21)
    check("hand class: feet FAST + along SLOW -> SLOW", hand_class("FAST", "SLOW"), "SLOW")
    check("hand class: feet FAST + along FAST -> FAST", hand_class("FAST", "FAST"), "FAST")
    check("hand class: feet INTERMEDIATE + along FAST -> INTERMEDIATE", hand_class("INTERMEDIATE", "FAST"), "INTERMEDIATE")
    check("hand class: feet FAST, no along -> NA (fail closed)", hand_class("FAST", None), "NA")
    check("hand class: feet FAST, empty along -> NA", hand_class("FAST", ""), "NA")
    check("hand class: no feet, along FAST -> NA", hand_class(None, "FAST"), "NA")
    check("hand class: feet SLOW, no along -> SLOW", hand_class("SLOW", None), "SLOW")
    na = _bed("C_lr", "A", "SSS")
    na[0]["hands"] = {"l": "NA", "r": "SLOW"}
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + na])
    check("an NA hand excludes its run (INCOMPLETE, never NO-FLIP)",
          (v["beds"]["C_lr"]["A"]["verdict"], [e["ref"] for e in v["excluded"]]), ("INCOMPLETE", ["C_lr__A_s11"]))

    print("killed / invalid runs")
    k = _bed("C_lr", "A", "SSS")
    k[1] = _run("C_lr__A_s12", L=None, R=None, killed=("dead", 1000))
    k[1]["hands"] = {"l": "NA", "r": "NA"}
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + k])["beds"]["C_lr"]["A"]
    check("a DEAD-killed run counts as not FAST (NO-FLIP 0/3)", (v["verdict"], v["n_arm"], v["k_arm"]), ("NO-FLIP", 3, 0))
    k = _bed("C_lr", "A", "FFF")
    k[0] = _run("C_lr__A_s11", killed=("diverged", "live"))
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSSSS") + k])["beds"]["C_lr"]["A"]
    check("a DIVERGED-killed run counts (not FAST) in n", (v["n_arm"], v["k_arm"]), (3, 2))
    w = _bed("C_lr", "A", "SSS")
    w[0] = _run("C_lr__A_s11", L="FAST", R="FAST", killed=("wiring", 200))
    res = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + w])
    check("a WIRING-killed run is excluded (INCOMPLETE)", (res["beds"]["C_lr"]["A"]["verdict"], len(res["excluded"])),
          ("INCOMPLETE", 1))
    rr = w + [_run("C_lr__A_s11_r2", chain="C_lr__A", seed=11)]
    res = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + rr])
    r2 = next(r for r in res["runs"] if r["ref"] == "C_lr__A_s11_r2")
    check("its rerun (fresh reference, same seed) counts and names what it replaces",
          (res["beds"]["C_lr"]["A"]["verdict"], any("replaces" in n for n in r2["notes"])), ("NO-FLIP", True))
    ph = _bed("C_lr", "A", "FSS")
    ph[0]["wiring"] = {"ok": False, "fails": ["Loss/action_bound absent"]}
    res = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + ph])
    check("a post-hoc wiring FAIL (no live kill) is excluded, never counted",
          (res["beds"]["C_lr"]["A"]["n_arm"], res["beds"]["C_lr"]["A"]["k_arm"]), (2, 0))
    dup = _bed("C_lr", "A", "SSS") + [_run("C_lr__A_s11_x", chain="C_lr__A", seed=11, L="FAST", R="FAST",
                                           recorded_at="2099")]
    res = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + dup])
    check("a second valid run of one chain + seed is a re-roll (excluded)",
          (res["beds"]["C_lr"]["A"]["k_arm"], [e["ref"] for e in res["excluded"]]), (0, ["C_lr__A_s11_x"]))
    pil5 = _bed("C", "C0", "F", two=False) + [_run("C__A_s11", two=False, killed=("dose", 1000), dose1000=False)] + \
        _bed("C_lr", "C0", "S") + _bed("C_lr", "A", "S")
    res = evaluate([dict(f) for f in pil5])
    check("pilot: a DOSE-killed C__A reads DOSE (not PENDING missing) + a stop", (res["pilot"]["status"], len(res["stops"])),
          ("DOSE", 1))
    sup = _bed("C", "C0", "S", two=False) + [_run("C__A_s11", two=False, killed=("dose", 1000), dose1000=True),
                                             _run("C__A_s11_r2", chain="C__A", seed=11, two=False, L="FAST")] + \
        _bed("C_lr", "C0", "S") + _bed("C_lr", "A", "S")
    res = evaluate([dict(f) for f in sup])
    ex = next(e for e in res["excluded"] if e["ref"] == "C__A_s11")
    check("a dose kill that passes the amended median rule is superseded: excluded, no stop, the pilot reads its rerun",
          (len(res["stops"]), "superseded" in ex["why"], res["pilot"]["status"] != "DOSE",
           res["pilot"].get("C__A_fast")), (0, True, True, True))

    print("step 4 / INC0")
    s3 = _bed("C_lr", "A", "SSS")
    s4 = _bed("C_lr", "A_fixed", "SFS")
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS") + s3 + s4])
    a4 = v["beds"]["C_lr"]["A_fixed"]
    check("4a arm compared with its control per seed (flip at s12 -> extend; DESCRIPTIVE)",
          (a4["control"]["flips"], a4["verdict"], bool(a4["extend"]), v["unlock"]), ([12], "DESCRIPTIVE", True, []))
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSSSS") + _bed("C_lr", "A", "SSSSS")
                  + _bed("C_lr", "A_fixed", "FFFFS")])
    check("a step-4 UNLOCK is listed apart from the step-3 4c triggers",
          (v["unlock"], v["unlock_step4"], v["beds"]["C_lr"]["A_fixed"]["verdict"]), ([], ["C_lr__A_fixed"], "UNLOCK"))
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSSSS") + _bed("C_lr", "B", "FFFFF")
                  + _bed("C_lr", "INC0", "FFFFS")])
    check("INC0 4/5 vs B 5/5 -> stop and ask; never an UNLOCK", (v["inc0"]["C_lr"]["status"], v["unlock"]),
          ("STOP-AND-ASK", ["C_lr__B"]))
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSSSS") + _bed("C_lr", "B", "FFFFF")
                  + _bed("C_lr", "INC0", "FSSSS")])
    check("INC0 1/5 vs B 5/5 -> B vs C0", v["inc0"]["C_lr"]["status"], "B-CONSTRUCTION")
    v = evaluate([dict(f) for f in cc0 + _bed("C_lr", "C0", "SSS", lr_mean=3e-4) + _bed("C_lr", "A", "SSS", lr_mean=3e-4)
                  + _bed("C_lr", "A_fixed", "SSS", lr_mean=1e-4)])
    check("a fixed-LR step-4 arm never re-triggers 4a", v["triggers_4a"], [])

    print("live checks")
    check("consumed: exact line", consumed_lines_text("[KNOB] KRABBY_INIT_STD=0.8 consumed\n", ["KRABBY_INIT_STD"]),
          {"KRABBY_INIT_STD": True})
    check("consumed: a NOT consumed line does not count",
          consumed_lines_text("[KNOB] KRABBY_RSI_CAM_STAGE=1 NOT consumed: RSI is off\n", ["KRABBY_RSI_CAM_STAGE"]),
          {"KRABBY_RSI_CAM_STAGE": False})
    st = np.arange(0, 300)
    kl = np.full(300, 0.01)
    kl[100:149] = 0.2
    check("KL > 0.1 for 49 iterations: no kill", divergence_read({"kl_mean": (st, kl)})["diverged"], False)
    kl[100:150] = 0.2
    check("KL > 0.1 for 50 iterations: kill", divergence_read({"kl_mean": (st, kl)})["diverged"], True)
    gap_st = np.concatenate([np.arange(100, 130), np.arange(140, 170)])
    check("a logging gap restarts the KL run", divergence_read({"kl_mean": (gap_st, np.full(60, 0.2))})["diverged"], False)
    st = np.arange(0, 1200)
    rw = np.full(1200, 10.0)
    rw[700:] = 7.5
    check("reward -25 % over 500: no kill", divergence_read({"reward": (st, rw)})["diverged"], False)
    rw[700:] = 6.5
    check("reward -35 % over 500: kill", divergence_read({"reward": (st, rw)})["diverged"], True)
    k4a = {"schedule": "fixed", "ppo_lr": 1e-4, "kl_stop": 0.02, "hist_lr": 3e-4}
    s200 = np.arange(0, 250)
    good = {"lr_applied": (s200, np.full(250, 1e-4)), "kl_stop": (s200, np.zeros(250)),
            "hist_lr": (s200, np.full(250, 3e-4))}
    check("4a wiring: lr_applied / kl_stop / hist_lr as pinned", step4a_wiring_fails(good, k4a), [])
    bad = dict(good, lr_applied=(s200, np.full(250, 1e-5)))
    bad.pop("hist_lr")
    check("4a wiring: wrong LR and no hist_lr tag both fail", len(step4a_wiring_fails(bad, k4a)), 2)
    check("4a wiring: epoch_kl skips the LR equality", step4a_wiring_fails(dict(good, lr_applied=(s200, np.full(250, 5e-4))),
                                                                          dict(k4a, schedule="epoch_kl")), [])
    check("check_run exit codes", _check_run_codes(), {"wiring": 3, "dose+dead": 4, "dead": 5, "pass": 0,
                                                       "diverged": 6, "missing past grace": 3, "missing": 2,
                                                       "C0 n/a": 0, "live on a non-4a arm": 0})

    print(f"self-test: {'PASS' if not fails else 'FAIL'} ({len(fails)} failing)")
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records-dir", type=Path, default=DEFAULT_ARMS)
    ap.add_argument("--candidates", type=Path, default=CAMPAIGN / "candidates.yaml")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--no-tb", action="store_true", help="skip the tfevents reads (stamp reads only)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--last-iter", type=Path, default=None, metavar="RUN_DIR")
    ap.add_argument("--check-run", type=Path, default=None, metavar="RUN_DIR")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--at", default=None, choices=("200", "1000", "live"))
    ap.add_argument("--train-log", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.last_iter is not None:
        try:
            print(tb_series(a.last_iter, {"track": TAGS["track"], "lr": TAGS["lr"]})["_last"])
        except Exception:  # noqa: BLE001 -- the watcher treats anything unreadable as "no progress yet"
            print(0)
        return 0
    cands = load_candidates(a.candidates)
    if a.check_run is not None:
        if not a.chain or a.at is None:
            ap.error("--check-run needs --chain and --at")
        rc, summ = check_run(a.check_run, a.chain, a.at if a.at == "live" else int(a.at), a.train_log, cands)
        print(json.dumps(_jsonable(summ)))
        return rc
    recs = [json.loads(p.read_text()) for p in sorted(a.records_dir.glob("*.record.json"))]
    if not recs:
        print(f"no *.record.json under {a.records_dir}", file=sys.stderr)
        return 2
    facts = [facts_from_record(r, cands, with_tb=not a.no_tb) for r in recs]
    res = evaluate(facts)
    md = markdown(res)
    if a.json:
        a.json.write_text(json.dumps(_jsonable(res), indent=1))
    if a.md:
        a.md.write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Omni phase 2 -- the live kill reads (CHARTER 8) and the calibration-smoke audit (CHARTER 5). CPU only, read-only.

Every read counts iterations FROM THE FORK START (the run's first logged iteration; a fork of model_14995 logs from
14996): "+200" is the window [first+190, first+210], "+1000" is [first+950, first+1050]. Each read that fails STOPS
THE QUEUE and asks (scripts/run_arm.sh: kill with `systemctl --user stop`, touch STOP_QUEUE, NOTIFY line).

  wiring  (+200)  * top-level `curriculum: null` in params/env.yaml (KRABBY_PHASEOUT "" removed every ramp);
                  * clock hidden (Metrics/clock_hidden_frac >= 0.999 over the window, when logged), clock / apex
                    reward weights 0.001, the arm's action_rate weight (dump);
                  * DR present (randomize_rigid_body_mass / _com events, push_by_setting_velocity at +-0.5 m/s) and the
                    element weights of CHARTER 3 (collision -2, edge -0.3, stumble -1, airtime 0.8, stride 0.5);
                  * the arm's geometry sub-terrains present (field: omni_field_terrain and no omni_ring_terrain;
                    A0: the reverse);
                  * Metrics/base_parkour/reach_obst_frac > 0 over the window;
                  * no dropped PPO keys (no "Ignoring unsupported PPO config keys" line in the training log);
                  * the OVERLAY AUDIT: run_candidates.overlay_checks over the chain's WHOLE stage overlay (every CHARTER 3
                    key, PROMOTE included) against params/env.yaml / agent.yaml and the training log -- a key with no
                    registered dump check FAILS (it cannot be shown realised);
                  * TERM LIVENESS: the arm-specific term's Episode_Reward over the window is non-zero (S1, E1, F1) and,
                    for E1 / F1, within 3x of its CHARTER 4 replay magnitude on the same per-max-episode-second basis
                    (replay_gate.json, written when the CHARTER 4 gate is read: {"E1": {"episode_reward": x}, ...});
                    F1's stand-only clause: per-slot telemetry (F1_NONSTAND_TAGS) reading 0 when logged; when none
                    is logged, the CHARTER 9 reading -- stand-only BY CONSTRUCTION: the dumped term's func is
                    parkour_isaaclab.envs.mdp.rewards:reward_stand_stance_quality AND that function's return is
                    multiplied by `stand_mask()` (an AST check of the code root's rewards.py); either failing = FAIL.
  terrain (+1000) on OBSTACLE tiles (OBST_LEVEL_TAGS): the window mean level has risen above its start (mean of the
                  fork's first 20 logged iterations), promotions are non-zero (PROMO_TAGS if logged; otherwise implied
                  by the rise, which no run can show without a promotion), and the mean level is not pinned
                  (<= PIN_EPS above 0 or >= top - PIN_EPS, top = num_rows - 1 from params/env.yaml). Missing
                  obstacle-tile telemetry is WAIT until fork + 1250 (the window end +1050 plus GRACE 200), then FAIL
                  (fail closed: never read off all tiles).
  preflight (before launch; run_arm.sh refuses with exit 7) what the live reads need from the CODE ROOT, so a run is
                  never launched only to be killed by missing telemetry: every arm and CAL2 needs the obstacle-tile
                  level and promotion metrics emitted (the metric keys of OBST_LEVEL_TAGS / PROMO_TAGS as string
                  literals in the non-experiment parkour sources); E1 / F1 need a non-zero replay_gate.json entry;
                  F1 needs the stand-only construction (or a per-slot metric) above.
  collapse (after +300, every poll) rolling-200 Train/mean_episode_length < 0.5 x the SOURCE run's last-200 mean, or
                  rolling-200 Episode_Termination/crab_failure > 0.6.

Exit codes (run_arm.sh's contract): 0 pass / not applicable, 2 not enough telemetry yet (retry next poll), 3 FAIL
(kill + stop the queue). The last stdout line is one JSON object.

    p2_live.py --iters RUN_DIR                                   -> "<first> <last>" logged iterations
    p2_live.py --check RUN_DIR --chain CHAIN --at 200|1000|collapse [--train-log F] [--source RUN_DIR]
               [--candidates F] [--replay F]
    p2_live.py --smoke-audit RUN_DIR --chain CHAIN [--train-log F] [--require-terrain-move]
    p2_live.py --preflight --chain CHAIN [--code-root DIR] [--replay F]
    p2_live.py --code-guard [--code-root DIR] [--lrout DIR] [--anchor-name ANC_s18] [--reanchor]
    p2_live.py --self-test

  code-guard (pre-launch audit M1, 2026-10-06: parallel sessions share the checkout) the LIVE code state of the code
                  root -- run_candidates.code_sha() / eval_sha() / a dirty flag on the code + eval pathspecs, the
                  three fields of the CHARTER 5 code-state sidecar -- against the s18 ANCHOR's sidecar (the newest
                  complete <lrout>/<anchor>/suite__omni/seed001/<ts>/code_state.json, the first anchor row the queue
                  runs). Before any anchor stamp exists it requires dirty = false only. A dirty tree, a difference in
                  any of the three fields, or an anchor stamp without a sidecar REFUSES (exit 7): scripts/run_arm.sh
                  calls it in run_ref (after the preflight) and at the top of every --row-eval / --lr-eval, and
                  stops + NOTIFIES. --reanchor (P2_REANCHOR=1 in run_arm.sh, CHARTER 5 "re-anchoring") skips the
                  comparison and still requires dirty = false, for the first row of a deliberate re-anchor only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
RC_PATH = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py"

RC_PASS, RC_WAIT, RC_FAIL = 0, 2, 3
RC_REFUSE = 7                                     # --code-guard refusal (run_arm.sh's "refused before launch" code)
#: the code-guard reference: the s18 anchor's first queue row (scripts/p2_queue.sh runs suite__omni env_seed 1 first)
ANCHOR_REF_ROW, ANCHOR_REF_SEED = "suite__omni", 1
OUT_DEFAULT = R / "parkour/outputs/omni_plan/phase2"
W200 = (190, 210)
W1000 = (950, 1050)
AT_200, AT_1000, AT_COLLAPSE = 210, 1060, 300     # fork-relative iteration a read first runs at
GRACE = 200                                       # telemetry still missing this far past the window -> FAIL
START_N = 20                                      # "its start": the fork's first 20 logged iterations
PIN_EPS = 0.05
ROLL = 200
COLLAPSE_LEN_FRAC = 0.5
COLLAPSE_FAIL = 0.6
LIVE_RATIO = 3.0
CLOCK_W = 0.001
#: CHARTER 3 "from the 2a preset" element weights (params/env.yaml rewards.<term>.weight)
ELEMENT_WEIGHTS = {"reward_collision": -2.0, "reward_feet_edge": -0.3, "reward_feet_stumble": -1.0,
                   "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5}
DR_EVENTS = ("randomize_rigid_body_mass", "randomize_rigid_body_com", "push_by_setting_velocity")
PUSH = 0.5
FIELD_FN, RING_FN = "omni_field_terrain", "omni_ring_terrain"
#: the arm-specific terms (CHARTER 8 term liveness), = motion_knobs.REUSED_WEIGHT_KNOBS["KRABBY_ACTION_RATE_W"],
#: motion_knobs.POWER_TERM (E1 re-scopes the existing power term to the actuated joints) and
#: motion_knobs.STAND_STANCE_TERM (pinned by tests/unit/test_omni_phase2_tooling.py).
ARM_TERMS = {"S1": "reward_action_rate", "E1": "penalty_mechanical_power", "F1": "reward_stand_stance_quality"}
#: F1's term function (params/env.yaml rewards.<term>.func) and the source the stand-only construction is read from.
F1_FUNC = "parkour_isaaclab.envs.mdp.rewards:reward_stand_stance_quality"
REWARDS_PY = "parkour/parkour_isaaclab/envs/mdp/rewards.py"
REPLAY_TERMS = ("E1", "F1")
#: Stage p2b (P2B_PLAN.md step 1 / CHARTER 10): each lever's new term(s), sign-aware -- the sign an income must
#: carry to count as live (a reward term pays >= 0, a penalty term <= 0). C0 has none (the p2a continuation).
P2B_ARM_TERMS = {"CL": ("reward_foot_clearance", "penalty_swing_min_clearance"), "TG": ("reward_orientation",)}
P2B_TERM_SIGN = {"reward_foot_clearance": 1, "penalty_swing_min_clearance": -1, "reward_orientation": -1}
#: TG's extra floor (P2B_PLAN.md step 1): |Episode_Reward/reward_orientation| at +200 must be >= this multiple of
#: the seed's S1 p2a training tail (that run's term income, which has no obstacle-tile share -- CHARTER 10).
P2B_TG_FLOOR_RATIO = 2.0
#: the p2b intermediate checkpoints' offsets from the fork's first logged iteration (P2B_PLAN.md step 1/4). Round,
#: unlike the phase-2 MCK offsets (1004/2504/4004), because every p2b fork is the same model_19994 head.
P2B_MCK_OFFSETS = (1005, 2505, 4005)
#: TG's floor tail length: the seed's S1 p2a run's last 500 its (P2B_PLAN.md step 1), not the live-read ROLL (200).
P2B_TAIL_N = 500
#: F1's "non-zero only in stand-slot envs" from per-slot telemetry, when logged: any of these tags, reading 0 over the
#: window. None is emitted in the tree (2026-10-06); the stand-only construction check stands in (CHARTER 9).
F1_NONSTAND_TAGS = ("Metrics/stand_stance/nonstand_income", "Metrics/base_velocity/stand_stance_nonstand")
#: obstacle-tile terrain level and promotion telemetry -- these EXACT names (the parkour term's reset log reaches
#: tensorboard as Metrics/base_parkour/k; omni_exposure.OMNI_TERRAIN_WINDOW_KEYS, emitted by
#: OmniParkourEvent.reset on ring AND field runs since 2026-10-06): terrain_levels_obst = mean of terrain_levels over
#: is_obst_tile envs at the reset call (left out when there is none); promote_count = the applied move_up decisions on
#: episodes that ended on an obstacle tile in that reset call (rsl_rl averages the reset calls of an iteration, so a
#: logged value is promotions per reset step, > 0 iff any happened). demote_count is logged too and not read here.
#: The preflight still refuses every arm and CAL2 when a code root lacks them (an all-tile terrain_levels read is
#: never accepted).
METRIC_PREFIX = "Metrics/base_parkour/"
OBST_LEVEL_TAGS = (METRIC_PREFIX + "terrain_levels_obst",)
PROMO_TAGS = (METRIC_PREFIX + "promote_count",)
#: where the preflight looks for the emitted metric keys (relative to the code root; experiments / tests skipped)
SOURCE_DIRS = ("parkour/parkour_isaaclab", "parkour/parkour_tasks/parkour_tasks")
TAGS = {"clock_hidden": "Metrics/clock_hidden_frac", "reach_obst": "Metrics/base_parkour/reach_obst_frac",
        "ep_len": "Train/mean_episode_length", "crab_failure": "Episode_Termination/crab_failure",
        "levels_all": "Metrics/base_parkour/terrain_levels", "std": "PPO/effective_std"}


# ------------------------------------------------------------------------------------------------ small helpers
def arm_of(chain: str) -> str:
    """P2__A1_s18 -> A1; P2__CAL2_s18 -> CAL2; P2__CL_s16 -> CL; P2__TG_s17 -> TG; P2__C0_s18 -> C0."""
    m = re.match(r"^P2__([A-Z]+[0-9]*)_s\d+$", chain)
    return m.group(1) if m else ""


def seed_of(chain: str) -> int | None:
    """P2__CL_s16 -> 16 (the trailing _s<seed>); None if the chain name carries no seed."""
    m = re.search(r"_s(\d+)$", chain)
    return int(m.group(1)) if m else None


def p2b_checkpoints(fork_model_n: int) -> list[int]:
    """The p2b intermediate checkpoint numbers for a fork of ``model_<fork_model_n>.pt`` (P2B_PLAN.md step 1):
    the fork's first logged iteration (``fork_model_n + 1``) plus each of :data:`P2B_MCK_OFFSETS`."""
    first = fork_model_n + 1
    return [first + o for o in P2B_MCK_OFFSETS]


def p2b_replay_magnitude(replay: dict | None, arm: str, term: str):
    """``replay_gate.json``'s ``{"p2b": {arm: {"terms": {term: {"episode_reward": x}}}}}`` entry, or None. A term that
    carries ``episode_reward_live_expect`` (the replay value x its measured training/replay calibration, CHARTER 10
    2026-10-08: TG's tilt runs 1.7-2.9x higher in training than in the eval replay) is read at that value."""
    ent = (((replay or {}).get("p2b") or {}).get(arm) or {}).get("terms", {}).get(term, {})
    return ent.get("episode_reward_live_expect", ent.get("episode_reward"))


def p2b_replay_verdict_ok(replay: dict | None, arm: str) -> bool:
    """CHARTER 10 / P2B_PLAN.md step 2: the arm's p2b replay verdict is PASS, or a recorded user overrule."""
    ent = ((replay or {}).get("p2b") or {}).get(arm) or {}
    return ent.get("verdict") == "PASS" or (ent.get("overrule") or {}).get("by") == "user"


def window(series, lo, hi):
    """Mean of [(step, value)] over lo..hi inclusive, or None."""
    vals = [v for s, v in (series or []) if lo <= s <= hi]
    return sum(vals) / len(vals) if vals else None


def tail_mean(series, n=ROLL):
    vals = [v for _, v in (series or [])][-n:]
    return sum(vals) / len(vals) if vals else None


def first_last(series_by_tag: dict) -> tuple[int | None, int | None]:
    steps = [s for ser in series_by_tag.values() for s, _ in (ser or [])]
    return (min(steps), max(steps)) if steps else (None, None)


def _yaml_tolerant(path: Path):
    import yaml  # noqa: PLC0415

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: tuple(ld.construct_sequence(node)))
    _L.add_multi_constructor("", lambda ld, suffix, node: None)
    with open(path) as fh:
        return yaml.load(fh, Loader=_L)  # noqa: S506


def _find_key(d, key):
    """First value of ``key`` anywhere in a nested dict (depth-first), else None."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def read_tb(run_dir: Path, extra_tags=()) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    want = set(TAGS.values()) | set(OBST_LEVEL_TAGS) | set(PROMO_TAGS) | set(F1_NONSTAND_TAGS) | set(extra_tags) | \
        {f"Episode_Reward/{t}" for t in ARM_TERMS.values()} | \
        {f"Episode_Reward/{t}" for terms in P2B_ARM_TERMS.values() for t in terms}
    return {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in want if t in have}


def load_rc(path: Path | None = None):
    spec = importlib.util.spec_from_file_location("_p2_run_candidates", path or RC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p2_run_candidates"] = mod
    spec.loader.exec_module(mod)
    return mod


def chain_overlay(cands: dict, chain: str) -> dict:
    ov: dict = {}
    for st in ((cands.get("chains") or {}).get(chain) or {}).get("stages") or []:
        ov.update({str(k): str(v) for k, v in (st.get("overlay") or {}).items()})
    return ov


def chain_source_ckpt(cands: dict, chain: str) -> str | None:
    st = (((cands.get("chains") or {}).get(chain) or {}).get("stages") or [{}])[0]
    rf = st.get("resume_from")
    return str(rf["checkpoint"]) if isinstance(rf, dict) and rf.get("checkpoint") else None


def s1_p2a_tail(arms_dir: Path, seed: int, n: int = P2B_TAIL_N) -> dict:
    """TG's floor (P2B_PLAN.md step 1): the seed's S1 p2a training run's last ``n`` its of
    Episode_Reward/reward_orientation (IO; the record's ``run_dir`` comes from scripts/run_arm.sh's write_record).
    ``{}`` when the record or its run dir is unreadable (judge_wiring then fails closed on the missing floor)."""
    try:
        rec = json.load(open(Path(arms_dir) / f"P2__S1_s{seed}.record.json"))
    except (OSError, ValueError):
        return {}
    rd = rec.get("run_dir")
    if not rd or not Path(rd).is_dir():
        return {}
    ser = read_tb(Path(rd), extra_tags=("Episode_Reward/reward_orientation",))
    return {"reward_orientation": tail_mean(ser.get("Episode_Reward/reward_orientation"), n)}


# ------------------------------------------------------------------------------------------------ pure judges
def dump_facts(env: dict | None, agent: dict | None, log_text: str | None) -> dict:
    """The dump-side facts of the wiring read, from the parsed params/env.yaml (+ the training log)."""
    if not env:
        return {"have_env": False}
    rew = env.get("rewards") or {}
    ev = env.get("events") or {}

    def w(term):
        t = rew.get(term)
        return (t or {}).get("weight") if isinstance(t, dict) else None

    push = ((ev.get("push_by_setting_velocity") or {}).get("params") or {}).get("velocity_range") or {}
    subs = _find_key(env.get("scene") or env, "sub_terrains") or {}
    fns = sorted({str((v or {}).get("function", "")).rsplit(":", 1)[-1] for v in subs.values() if isinstance(v, dict)})
    rows = _find_key(env.get("scene") or env, "num_rows")
    funcs = {t: str((rew.get(t) or {}).get("func")) if isinstance(rew.get(t), dict) else None
             for t in ARM_TERMS.values()}
    return {"have_env": True, "term_funcs": funcs, "curriculum_null": "curriculum" in env and env.get("curriculum") is None,
            "weights": {t: w(t) for t in list(ELEMENT_WEIGHTS) + ["reward_clock_schedule", "reward_clock_swing_apex",
                                                                    "reward_action_rate"] + list(ARM_TERMS.values())},
            # present = a cfg block, not merely the key: a disabled DR event dumps as `name: null` (pre-launch
            # audit M3, 2026-10-06), and `e in ev` read that as present
            "dr_events": {e: isinstance(ev.get(e), dict) for e in DR_EVENTS},
            "push_x": list(push.get("x") or []) if isinstance(push, dict) else [],
            "terrain_fns": fns, "num_rows": rows,
            "dropped_ppo_keys": dropped_ppo_keys(log_text)}


#: The runner's documented, deliberately ignored PPO key (the PPO-construction landing allow-list; every B run logs
#: it). Any OTHER key in an "Ignoring unsupported PPO config keys" line is a dropped knob.
PPO_KEY_ALLOW = frozenset({"share_cnn_encoders"})


def dropped_ppo_keys(log_text: str | None) -> list[str]:
    """Keys the runner ignored, minus :data:`PPO_KEY_ALLOW` (pure). A line whose key list cannot be parsed counts as
    dropped (fail closed)."""
    out = []
    for ln in (log_text or "").splitlines():
        if "Ignoring unsupported PPO config keys" not in ln:
            continue
        clean = re.sub(r"\x1b\[[0-9;]*m", "", ln)
        tail = clean.rsplit(":", 1)[-1] if ":" in clean else ""
        keys = [k.strip() for k in tail.split(",") if k.strip()]
        bad = [k for k in keys if k not in PPO_KEY_ALLOW]
        if bad or not keys:
            out.append(clean.strip())
    return out


def _close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def judge_wiring(arm: str, overlay: dict, facts: dict, ser: dict, first: int, audit: list, replay: dict | None,
                 stand_gate: tuple[bool, str] | None = None, ref_tail: dict | None = None) -> dict:
    """CHARTER 8 wiring at +200 (pure). ``audit`` = run_candidates.overlay_checks output [(name, ok)];
    ``stand_gate`` = :func:`stand_gate_structural` of the code root (F1 only; None = not read = FAIL for F1);
    ``ref_tail`` = the seed's S1 p2a training tail (TG only; P2B_PLAN.md step 1, e.g. {"reward_orientation": x})."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    lo, hi = first + W200[0], first + W200[1]
    if last is None or last < hi:
        return {"rc": RC_WAIT, "why": f"telemetry only to it {last} (window {lo}-{hi})"}
    fails, notes = [], []
    if not facts.get("have_env"):
        return {"rc": RC_FAIL, "result": "FAIL", "fails": ["params/env.yaml missing or unreadable"]}
    if not facts["curriculum_null"]:
        fails.append("top-level curriculum is not null (a PHASEOUT ramp survived)")
    ch = window(ser.get(TAGS["clock_hidden"]), lo, hi)
    if ch is not None and ch < 0.999:
        fails.append(f"Metrics/clock_hidden_frac {ch:.3f} < 0.999")
    wts = facts["weights"]
    for t in ("reward_clock_schedule", "reward_clock_swing_apex"):
        if not _close(wts.get(t), CLOCK_W):
            fails.append(f"{t} weight {wts.get(t)} != {CLOCK_W}")
    want_ar = float(overlay.get("KRABBY_ACTION_RATE_W", "0") or 0)
    if not _close(wts.get("reward_action_rate"), want_ar):
        fails.append(f"reward_action_rate weight {wts.get('reward_action_rate')} != {want_ar} (arm {arm})")
    for t, want in ELEMENT_WEIGHTS.items():
        if not _close(wts.get(t), want):
            fails.append(f"{t} weight {wts.get(t)} != {want}")
    fails += [f"DR event {e} absent" for e, ok in facts["dr_events"].items() if not ok]
    px = facts.get("push_x") or []
    if len(px) != 2 or not (_close(px[0], -PUSH) and _close(px[1], PUSH)):
        fails.append(f"push velocity x range {px} != [-{PUSH}, {PUSH}]")
    geom = str(overlay.get("KRABBY_FLAT_TERRAIN_GEOM", "")).strip().lower()
    fns = set(facts.get("terrain_fns") or [])
    want_fn, other = (FIELD_FN, RING_FN) if geom == "field" else (RING_FN, FIELD_FN)
    if want_fn not in fns or other in fns:
        fails.append(f"geometry {geom!r}: sub-terrain functions {sorted(fns)} (want {want_fn}, not {other})")
    ro = window(ser.get(TAGS["reach_obst"]), lo, hi)
    if ro is None:
        fails.append("Metrics/base_parkour/reach_obst_frac not logged")
    elif not ro > 0:
        fails.append(f"reach_obst_frac {ro} not > 0")
    if facts.get("dropped_ppo_keys"):
        fails.append(f"the runner dropped PPO keys: {facts['dropped_ppo_keys'][0][:160]}")
    if not audit:
        fails.append("overlay audit produced no checks")
    fails += [f"overlay audit: {name}" for name, ok in audit if not ok]
    term = ARM_TERMS.get(arm)
    live = {}
    if term:
        er = window(ser.get(f"Episode_Reward/{term}"), lo, hi)
        live = {"term": term, "episode_reward": er}
        if er is None or er == 0.0:
            fails.append(f"term liveness: Episode_Reward/{term} is {er} (must be non-zero)")
        elif arm in REPLAY_TERMS:
            exp = ((replay or {}).get(arm) or {}).get("episode_reward")
            live["replay_expected"] = exp
            if exp in (None, 0, 0.0):
                fails.append(f"term liveness: no CHARTER 4 replay magnitude for {arm} (replay_gate.json)")
            elif er / exp <= 0 or not (1.0 / LIVE_RATIO - 1e-9 <= er / exp <= LIVE_RATIO + 1e-9):
                fails.append(f"term liveness: Episode_Reward/{term} {er:.4g} vs replay {exp:.4g} (outside 3x)")
        if arm == "F1":
            ns = [window(ser.get(t), lo, hi) for t in F1_NONSTAND_TAGS if ser.get(t)]
            if ns:
                if any(v is None or v != 0.0 for v in ns):
                    fails.append(f"term liveness: F1 pays outside the stand slot ({ns})")
            else:                                    # CHARTER 9 reading: stand-only by construction
                func = (facts.get("term_funcs") or {}).get(term)
                if func != F1_FUNC:
                    fails.append(f"term liveness: F1 term func {func!r} != {F1_FUNC} (stand-only construction unshown)")
                if not stand_gate or not stand_gate[0]:
                    fails.append("term liveness: F1 stand-only construction not shown "
                                 f"({stand_gate[1] if stand_gate else 'not read'})")
                if func == F1_FUNC and stand_gate and stand_gate[0]:
                    notes.append(f"F1 stand-only by construction (CHARTER 9): {stand_gate[1]}")
    elif arm in P2B_ARM_TERMS:
        terms = P2B_ARM_TERMS[arm]
        live = {"terms": {}}
        for t in terms:
            er = window(ser.get(f"Episode_Reward/{t}"), lo, hi)
            sign = P2B_TERM_SIGN[t]
            live["terms"][t] = {"episode_reward": er}
            if er is None or er == 0.0 or (sign > 0 and er < 0) or (sign < 0 and er > 0):
                fails.append(f"term liveness: Episode_Reward/{t} is {er} (must be "
                             f"{'non-negative' if sign > 0 else 'non-positive'} and non-zero, arm {arm})")
                continue
            exp = p2b_replay_magnitude(replay, arm, t)
            live["terms"][t]["replay_expected"] = exp
            if exp in (None, 0, 0.0):
                fails.append(f"term liveness: no p2b replay magnitude for {arm}/{t} (replay_gate.json)")
            elif er / exp <= 0 or not (1.0 / LIVE_RATIO - 1e-9 <= er / exp <= LIVE_RATIO + 1e-9):
                fails.append(f"term liveness: Episode_Reward/{t} {er:.4g} vs replay {exp:.4g} (outside 3x)")
        if arm == "TG":
            er = live["terms"].get("reward_orientation", {}).get("episode_reward")
            tail = (ref_tail or {}).get("reward_orientation")
            live["ref_tail"] = tail
            if tail in (None, 0, 0.0):
                fails.append("term liveness: no seed S1 p2a run tail for TG's floor (reward_orientation)")
            elif er is not None and abs(er) < P2B_TG_FLOOR_RATIO * abs(tail):
                fails.append(f"term liveness: TG |Episode_Reward/reward_orientation| {er:.4g} < "
                             f"{P2B_TG_FLOOR_RATIO}x the seed's S1 p2a tail {tail:.4g}")
    else:
        notes.append(f"arm {arm}: no arm-specific term (liveness n/a)")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "notes": notes, "reach_obst": ro, "clock_hidden": ch, "liveness": live, "window": [lo, hi]}


def stand_gate_structural(rewards_py: Path) -> tuple[bool, str]:
    """F1's stand-only construction, read from source (CPU, no import): in ``reward_stand_stance_quality`` a name is
    bound to an expression calling ``.stand_mask()`` and every ``return`` is a product with that name as a factor."""
    import ast  # noqa: PLC0415

    try:
        tree = ast.parse(Path(rewards_py).read_text())
    except (OSError, SyntaxError) as e:
        return False, f"{rewards_py} unreadable ({e.__class__.__name__})"
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == ARM_TERMS["F1"]), None)
    if fn is None:
        return False, f"no def {ARM_TERMS['F1']} in {rewards_py}"
    gates = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "stand_mask"
                     for c in ast.walk(n.value)) for t in n.targets if isinstance(t, ast.Name)}
    if not gates:
        return False, "no name bound to a stand_mask() call"

    def factors(e):
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
            return factors(e.left) + factors(e.right)
        return [e]
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if not rets:
        return False, "no return"
    for r in rets:
        names = {f.id for f in factors(r.value) if isinstance(f, ast.Name)} if r.value is not None else set()
        if not names & gates:
            return False, f"a return (line {r.lineno}) is not multiplied by the stand mask {sorted(gates)}"
    return True, f"every return of {fn.name} is multiplied by {sorted(gates)} (= stand_mask())"


def emitted_metric_keys(code_root: Path, keys) -> dict:
    """{key: [files]} -- the metric keys found as string literals in the non-experiment parkour sources."""
    import ast  # noqa: PLC0415

    want, found = set(keys), {k: [] for k in keys}
    for d in SOURCE_DIRS:
        for f in sorted((Path(code_root) / d).rglob("*.py")):
            if "experiments" in f.parts or "tests" in f.parts:
                continue
            try:
                txt = f.read_text(errors="replace")
            except OSError:
                continue
            if not any(k in txt for k in want):
                continue
            try:
                lits = {n.value for n in ast.walk(ast.parse(txt)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            except SyntaxError:
                continue
            for k in want & lits:
                found[k].append(str(f.relative_to(code_root)))
    return found


def judge_preflight(arm: str, emitted: dict, replay: dict | None, stand_gate: tuple[bool, str] | None) -> dict:
    """What the live reads need before a launch (pure). ``emitted`` = :func:`emitted_metric_keys` output."""
    fails, notes = [], []
    if arm != "CAL1":                                 # every arm (+1000 terrain read) and CAL2 (levels must move)
        for tag in OBST_LEVEL_TAGS + PROMO_TAGS:
            key = tag[len(METRIC_PREFIX):]
            if not emitted.get(key):
                fails.append(f"metric {tag} is not emitted by the code root (no '{key}' literal in {list(SOURCE_DIRS)}): "
                             f"the +1000 terrain read would fail closed")
    if arm in REPLAY_TERMS:
        ent = (replay or {}).get(arm) or {}
        exp = ent.get("episode_reward")
        if not isinstance(exp, (int, float)) or exp == 0:
            fails.append(f"no CHARTER 4 replay magnitude for {arm} in replay_gate.json ({exp!r})")
        # CHARTER 4: a FAIL stops and asks -- the arm may launch only on a PASS or a recorded user overrule
        # ({"overrule": {"by": "user", "date": ..., "note": ...}} written by hand after the user's decision).
        ov = ent.get("overrule") or {}
        if ent.get("verdict") != "PASS" and ov.get("by") != "user":
            fails.append(f"CHARTER 4 replay verdict for {arm} is {ent.get('verdict')!r} with no recorded user overrule")
    if arm in P2B_ARM_TERMS:
        # CHARTER 10 / P2B_PLAN.md step 2: CL/TG may launch only on a p2b replay-gate PASS or a recorded user
        # overrule, and only once every one of the arm's terms has its own non-zero magnitude recorded.
        for t in P2B_ARM_TERMS[arm]:
            exp = p2b_replay_magnitude(replay, arm, t)
            if not isinstance(exp, (int, float)) or exp == 0:
                fails.append(f"no p2b replay magnitude for {arm}/{t} in replay_gate.json ({exp!r})")
        if not p2b_replay_verdict_ok(replay, arm):
            ent = ((replay or {}).get("p2b") or {}).get(arm) or {}
            fails.append(f"p2b replay verdict for {arm} is {ent.get('verdict')!r} with no recorded user overrule")
    if arm == "F1":
        tags = [t for t in F1_NONSTAND_TAGS if (emitted.get(t.rsplit("/", 1)[-1]))]
        if tags:
            notes.append(f"F1 per-slot telemetry emitted: {tags}")
        elif not stand_gate or not stand_gate[0]:
            fails.append(f"F1 stand-only construction not shown ({stand_gate[1] if stand_gate else 'not read'})")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails, "notes": notes,
            "arm": arm}


def judge_terrain(ser: dict, first: int, num_rows) -> dict:
    """CHARTER 8 terrain at +1000 (pure), on obstacle-tile telemetry only."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    lo, hi = first + W1000[0], first + W1000[1]
    if last is None or last < hi:
        return {"rc": RC_WAIT, "why": f"telemetry only to it {last}"}
    tag = next((t for t in OBST_LEVEL_TAGS if ser.get(t)), None)
    if tag is None:
        if last >= hi + GRACE:
            return {"rc": RC_FAIL, "result": "FAIL",
                    "fails": [f"no obstacle-tile terrain-level telemetry ({list(OBST_LEVEL_TAGS)}) by it {last}"]}
        return {"rc": RC_WAIT, "why": "no obstacle-tile terrain-level tag yet"}
    lv = ser[tag]
    start_vals = [v for _, v in sorted(lv)[:START_N]]
    start = sum(start_vals) / len(start_vals)
    at = window(lv, lo, hi)
    fails = []
    if at is None:
        return {"rc": RC_WAIT, "why": f"{tag} has no point in {lo}-{hi}"}
    risen = at > start
    if not risen:
        fails.append(f"obstacle-tile mean level {at:.3f} has not risen above its start {start:.3f}")
    ptag = next((t for t in PROMO_TAGS if ser.get(t)), None)
    if ptag:
        pv = [v for s, v in ser[ptag] if first <= s <= hi]
        promos = sum(pv) > 0
        promo_src = ptag
    else:
        promos = risen
        promo_src = "implied by the rise (no promotion tag logged)"
    if not promos:
        fails.append(f"no promotions ({promo_src})")
    try:
        top = int(num_rows) - 1
    except (TypeError, ValueError):
        top = None
        fails.append(f"num_rows unreadable from params/env.yaml ({num_rows!r}): cannot test the top pin")
    if at <= PIN_EPS:
        fails.append(f"mean level {at:.3f} pinned at 0")
    if top is not None and at >= top - PIN_EPS:
        fails.append(f"mean level {at:.3f} pinned at the top level {top}")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "tag": tag, "start": start, "at_1000": at, "promotions": promo_src, "top": top}


def judge_collapse(ser: dict, first: int, source_tail_len) -> dict:
    """CHARTER 8 collapse (pure): after +300 only."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    if last is None or last < first + AT_COLLAPSE:
        return {"rc": RC_PASS, "result": "n/a (before +300)"}
    el = [(s, v) for s, v in ser.get(TAGS["ep_len"]) or [] if s >= first]
    cf = [(s, v) for s, v in ser.get(TAGS["crab_failure"]) or [] if s >= first]
    if not el or not cf or source_tail_len in (None, 0):
        return {"rc": RC_WAIT, "why": "episode length / crab_failure / source tail missing"}
    r_len, r_cf = tail_mean(el), tail_mean(cf)
    fails = []
    if r_len < COLLAPSE_LEN_FRAC * source_tail_len:
        fails.append(f"rolling-200 episode length {r_len:.1f} < 0.5 x source tail {source_tail_len:.1f}")
    if r_cf > COLLAPSE_FAIL:
        fails.append(f"rolling-200 crab_failure {r_cf:.3f} > {COLLAPSE_FAIL}")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "ep_len_roll": r_len, "crab_failure_roll": r_cf, "source_tail": source_tail_len}


def judge_smoke(audit: list, facts: dict, ser: dict, overlay: dict, require_move: bool) -> dict:
    """CHARTER 5 calibration-smoke dump audit (pure): every overlay key realised (PROMOTE included), no dropped PPO
    key; with ``require_move`` the obstacle-tile terrain levels must move (max != min over the run)."""
    fails = [f"overlay audit: {n}" for n, ok in audit if not ok]
    if not audit:
        fails.append("overlay audit produced no checks")
    if not any("KRABBY_TERRAIN_PROMOTE" in n for n, _ in audit):
        fails.append("KRABBY_TERRAIN_PROMOTE not audited")
    if facts.get("dropped_ppo_keys"):
        fails.append("the runner dropped PPO keys")
    moved = None
    if require_move:
        tag = next((t for t in OBST_LEVEL_TAGS if ser.get(t)), None)
        if tag is None:
            fails.append("no obstacle-tile terrain-level telemetry: cannot show the levels move")
        else:
            vals = [v for _, v in ser[tag]]
            moved = max(vals) - min(vals) > 1e-6
            if not moved:
                fails.append(f"{tag} never moved ({vals[0]:.3f})")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "promote": overlay.get("KRABBY_TERRAIN_PROMOTE"), "levels_moved": moved}


# ------------------------------------------------------------------------------------------------ IO drivers
def _gather(run_dir: Path, train_log):
    envp, agp = run_dir / "params" / "env.yaml", run_dir / "params" / "agent.yaml"
    env = _yaml_tolerant(envp) if envp.is_file() else None
    agent = _yaml_tolerant(agp) if agp.is_file() else None
    log = Path(train_log).read_text(errors="replace") if train_log and Path(train_log).is_file() else None
    return env, agent, log


def check(a) -> tuple[int, dict]:
    import yaml  # noqa: PLC0415

    cands = yaml.safe_load(open(a.candidates))
    if a.chain not in (cands.get("chains") or {}):
        return RC_WAIT, {"why": f"chain {a.chain} not in {a.candidates}"}
    arm = arm_of(a.chain)
    ov = chain_overlay(cands, a.chain)
    ser = read_tb(a.check)
    first, last = first_last(ser)
    if first is None:
        return RC_WAIT, {"why": "no telemetry yet"}
    if a.at == "200":
        env, agent, log = _gather(a.check, a.train_log)
        if last < first + W200[1]:
            return RC_WAIT, {"why": f"telemetry only to it {last}"}
        rc_mod = load_rc()
        audit = rc_mod.overlay_checks({"ckpt": str(a.check / "model_audit.pt"), "log": a.train_log}, ov)
        replay = json.load(open(a.replay)) if a.replay and Path(a.replay).is_file() else None
        sg = stand_gate_structural(Path(a.code_root) / REWARDS_PY) if arm == "F1" else None
        seed = seed_of(a.chain)
        ref_tail = s1_p2a_tail(a.arms, seed) if arm == "TG" and seed is not None else None
        res = judge_wiring(arm, ov, dump_facts(env, agent, log), ser, first, audit, replay, sg, ref_tail)
    elif a.at == "1000":
        env, _, _ = _gather(a.check, None)
        res = judge_terrain(ser, first, _find_key((env or {}).get("scene") or env or {}, "num_rows"))
    else:
        src = a.source or (R / chain_source_ckpt(cands, a.chain)).parent
        src_ser = read_tb(Path(src))
        res = judge_collapse(ser, first, tail_mean(src_ser.get(TAGS["ep_len"])))
    res.update(chain=a.chain, at=a.at, first=first, last=last)
    return int(res["rc"]), res


def smoke_audit(a) -> tuple[int, dict]:
    import yaml  # noqa: PLC0415

    cands = yaml.safe_load(open(a.candidates))
    ov = chain_overlay(cands, a.chain)
    env, agent, log = _gather(a.smoke_audit, a.train_log)
    audit = load_rc().overlay_checks({"ckpt": str(a.smoke_audit / "model_audit.pt"), "log": a.train_log}, ov)
    res = judge_smoke(audit, dump_facts(env, agent, log), read_tb(a.smoke_audit), ov, a.require_terrain_move)
    res.update(chain=a.chain)
    return int(res["rc"]), res


def preflight(a) -> tuple[int, dict]:
    import yaml  # noqa: PLC0415

    cands = yaml.safe_load(open(a.candidates))
    if a.chain not in (cands.get("chains") or {}):
        return RC_FAIL, {"result": "FAIL", "fails": [f"chain {a.chain} not in {a.candidates}"]}
    arm = arm_of(a.chain)
    keys = [t[len(METRIC_PREFIX):] for t in OBST_LEVEL_TAGS + PROMO_TAGS] + [t.rsplit("/", 1)[-1] for t in F1_NONSTAND_TAGS]
    emitted = emitted_metric_keys(Path(a.code_root), keys)
    replay = json.load(open(a.replay)) if a.replay and Path(a.replay).is_file() else None
    sg = stand_gate_structural(Path(a.code_root) / REWARDS_PY) if arm == "F1" else None
    res = judge_preflight(arm, emitted, replay, sg)
    res.update(chain=a.chain, emitted={k: v for k, v in emitted.items() if v})
    return int(res["rc"]), res


# ------------------------------------------------------------------------------------------------ code guard (M1)
CODE_STATE_KEYS = ("code_sha", "eval_sha", "dirty")


def anchor_reference_sidecar(lrout: Path, name: str) -> tuple[str, dict | None, str | None]:
    """The code-guard reference: ``(state, sidecar, stamp)`` of the newest COMPLETE ``<lrout>/<name>/suite__omni/
    seed001/<ts>`` stamp. ``state`` is ``none`` (no anchor stamp yet), ``missing`` (a stamp with no readable
    code_state.json -- fails closed) or ``ok``."""
    d = Path(lrout) / name / ANCHOR_REF_ROW / f"seed{ANCHOR_REF_SEED:03d}"
    hits = sorted(p for p in d.glob("*/") if (p / "complete.json").is_file()) if d.is_dir() else []
    if not hits:
        return "none", None, None
    st = hits[-1]
    try:
        data = json.load(open(st / "code_state.json"))
    except (OSError, ValueError):
        return "missing", None, str(st)
    return ("ok", data, str(st)) if isinstance(data, dict) else ("missing", None, str(st))


def judge_code_guard(live: dict, ref_state: str, ref: dict | None, reanchor: bool = False) -> dict:
    """Pre-launch audit M1 (pure). ``live`` = {code_sha, eval_sha, dirty, dirty_paths} of the code root now; ``ref`` =
    the s18 anchor's sidecar. Refuses a dirty tree always; with an anchor, any difference in code_sha / eval_sha /
    dirty; an anchor stamp without a sidecar (never "equal")."""
    fails, notes = [], []
    if live.get("dirty") or live.get("dirty") is None:
        fails.append(f"code/eval pathspecs dirty (or unreadable): {live.get('dirty_paths')}")
    if reanchor:
        notes.append("re-anchor: the anchor comparison is skipped (dirty = false still required)")
    elif ref_state == "none":
        notes.append("no anchor stamp yet: dirty = false required, nothing to compare")
    elif ref_state == "missing" or not ref:
        fails.append("the s18 anchor stamp has no code_state.json sidecar (fail closed)")
    else:
        for k in CODE_STATE_KEYS:
            a, b = live.get(k), ref.get(k)
            if k == "dirty":
                a, b = bool(a), bool(b)
            if a != b:
                fails.append(f"{k}: live {a!r} != anchor {b!r}")
    return {"rc": RC_REFUSE if fails else RC_PASS, "result": "REFUSE" if fails else "PASS", "fails": fails,
            "notes": notes, "ref_state": ref_state, "live": {k: live.get(k) for k in CODE_STATE_KEYS},
            "anchor": {k: (ref or {}).get(k) for k in CODE_STATE_KEYS} if ref else None}


def live_code_state(code_root: Path) -> dict:
    rc = load_rc(Path(code_root) / RC_PATH.relative_to(R))
    dirty = rc.dirty_paths(Path(code_root))
    return {"code_sha": rc.code_sha(Path(code_root)), "eval_sha": rc.eval_sha(Path(code_root)), "dirty": bool(dirty),
            "dirty_paths": dirty}


def _anchor_name(seed: int = 18) -> str:
    import yaml  # noqa: PLC0415

    try:
        return str(yaml.safe_load(open(CAMPAIGN / "anchors.yaml"))["anchors"][seed]["name"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        # fail closed (review of the CHARTER 9 walls re-tune): a guessed name could resolve to the ramp-era ANC_s*
        # stamps, which carry today's code state and would PASS against a stale reference
        raise RuntimeError(f"anchors.yaml unreadable, no code-guard reference: {e.__class__.__name__}: {e}") from e


def code_guard(a) -> tuple[int, dict]:
    lrout = Path(a.lrout) if a.lrout else Path(os.environ.get("P2_OUT") or OUT_DEFAULT) / "lr"
    name = a.anchor_name or _anchor_name()
    try:
        live = live_code_state(Path(a.code_root))
    except Exception as e:  # noqa: BLE001 -- an unreadable code state refuses, never passes
        live = {"dirty": None, "dirty_paths": f"code state unreadable: {e.__class__.__name__}: {e}"}
    state, ref, stamp = anchor_reference_sidecar(lrout, name)
    res = judge_code_guard(live, state, ref, a.reanchor)
    res.update(anchor_name=name, anchor_stamp=stamp, code_root=str(a.code_root))
    return int(res["rc"]), res


# ------------------------------------------------------------------------------------------------ self-test
def _ser(first=14996, n=1100, **const):
    return {tag: [(first + i, v) for i in range(n)] for tag, v in const.items()}


def _facts(**over):
    f = {"have_env": True, "curriculum_null": True,
         "weights": dict(ELEMENT_WEIGHTS, reward_clock_schedule=0.001, reward_clock_swing_apex=0.001,
                         reward_action_rate=0.0, reward_stand_stance_quality=None, penalty_mechanical_power=0.0),
         "term_funcs": {"reward_stand_stance_quality": F1_FUNC},
         "dr_events": {e: True for e in DR_EVENTS}, "push_x": [-0.5, 0.5], "terrain_fns": ["omni_field_terrain",
                                                                                          "omni_flat_terrain"],
         "num_rows": 10, "dropped_ppo_keys": []}
    f.update(over)
    return f


def _self_test() -> int:
    ok = True

    def check_(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    F0 = 14996
    ov_a1 = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "0"}
    base = {TAGS["reach_obst"]: 0.3, TAGS["clock_hidden"]: 1.0}
    audit_ok = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    s = _ser(n=215, **base)
    check_("wiring: a clean A1 passes", judge_wiring("A1", ov_a1, _facts(), s, F0, audit_ok, None)["rc"], RC_PASS)
    check_("wiring: telemetry short of +210 waits", judge_wiring("A1", ov_a1, _facts(), _ser(n=150, **base), F0,
                                                                 audit_ok, None)["rc"], RC_WAIT)
    check_("wiring: counted from the fork start (14996+210), not from 0",
           judge_wiring("A1", ov_a1, _facts(), _ser(first=F0, n=205, **base), F0, audit_ok, None)["rc"], RC_WAIT)
    check_("wiring: curriculum not null fails", judge_wiring("A1", ov_a1, _facts(curriculum_null=False), s, F0,
                                                             audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: a failed overlay-audit key fails", judge_wiring("A1", ov_a1, _facts(), s, F0,
                                                                    audit_ok + [("overlay X", False)], None)["rc"], RC_FAIL)
    check_("wiring: an empty audit fails closed", judge_wiring("A1", ov_a1, _facts(), s, F0, [], None)["rc"], RC_FAIL)
    check_("wiring: reach_obst 0 fails", judge_wiring("A1", ov_a1, _facts(), _ser(n=215, **dict(base, **{
        TAGS["reach_obst"]: 0.0})), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: ring sub-terrains under a field arm fail",
           judge_wiring("A1", ov_a1, _facts(terrain_fns=["omni_ring_terrain"]), s, F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: A0 ring passes with ring sub-terrains",
           judge_wiring("A0", {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_ACTION_RATE_W": "0"},
                        _facts(terrain_fns=["omni_ring_terrain"]), s, F0, audit_ok, None)["rc"], RC_PASS)
    check_("wiring: a dropped PPO key fails", judge_wiring("A1", ov_a1, _facts(dropped_ppo_keys=["Ignoring unsupported "
                                                           "PPO config keys: kl_stop"]), s, F0, audit_ok, None)["rc"], RC_FAIL)
    _w = "\x1b[33m18:12:10 [on_policy_runner_with_extractor.py] WARNING: Ignoring unsupported PPO config keys for PPOWithExtractor: "
    check_("dropped keys: the allow-listed share_cnn_encoders alone is not a drop",
           len(dropped_ppo_keys(_w + "share_cnn_encoders\x1b[0m")), 0)
    check_("dropped keys: any other key is a drop", len(dropped_ppo_keys(_w + "share_cnn_encoders, kl_stop\x1b[0m")), 1)
    check_("wiring: clock 1.0 (an un-blanked preset) fails", judge_wiring("A1", ov_a1, _facts(weights=dict(
        _facts()["weights"], reward_clock_schedule=1.0)), s, F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: element weight off (airtime 0.4) fails", judge_wiring("A1", ov_a1, _facts(weights=dict(
        _facts()["weights"], reward_feet_air_time_positive=0.4)), s, F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: a missing DR event fails", judge_wiring("A1", ov_a1, _facts(dr_events={
        "randomize_rigid_body_mass": True, "randomize_rigid_body_com": False, "push_by_setting_velocity": True}),
        s, F0, audit_ok, None)["rc"], RC_FAIL)
    ov_s1 = dict(ov_a1, KRABBY_ACTION_RATE_W="-0.05")
    w_s1 = dict(_facts()["weights"], reward_action_rate=-0.05)
    s_s1 = _ser(n=215, **dict(base, **{"Episode_Reward/reward_action_rate": -0.004}))
    check_("wiring: S1 live action_rate term passes", judge_wiring("S1", ov_s1, _facts(weights=w_s1), s_s1, F0,
                                                                   audit_ok, None)["rc"], RC_PASS)
    check_("wiring: S1 with a dead action_rate term fails", judge_wiring("S1", ov_s1, _facts(weights=w_s1),
                                                                         _ser(n=215, **dict(base, **{"Episode_Reward/reward_action_rate": 0.0})),
                                                                         F0, audit_ok, None)["rc"], RC_FAIL)
    s_e1 = _ser(n=215, **dict(base, **{"Episode_Reward/penalty_mechanical_power": -0.03}))
    check_("wiring: E1 within 3x of the replay passes", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                                     {"E1": {"episode_reward": -0.02}})["rc"], RC_PASS)
    check_("wiring: E1 at 3.5x the replay fails", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                               {"E1": {"episode_reward": -0.03 / 3.5}})["rc"], RC_FAIL)
    check_("wiring: E1 at exactly 3x passes (boundary)", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                                      {"E1": {"episode_reward": -0.01}})["rc"], RC_PASS)
    check_("wiring: E1 of the wrong sign fails", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                              {"E1": {"episode_reward": 0.03}})["rc"], RC_FAIL)
    check_("wiring: E1 with no replay record fails closed", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                                         None)["rc"], RC_FAIL)
    s_f1 = _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance_quality": 0.05}))
    gate_ok, gate_bad = (True, "stand_mask() factor"), (False, "no stand_mask() factor")
    check_("wiring: F1 without per-slot telemetry or a construction read fails closed",
           judge_wiring("F1", ov_a1, _facts(), s_f1, F0, audit_ok, {"F1": {"episode_reward": 0.04}})["rc"], RC_FAIL)
    check_("wiring: F1 stand-only by construction (no per-slot telemetry) passes",
           judge_wiring("F1", ov_a1, _facts(), s_f1, F0, audit_ok, {"F1": {"episode_reward": 0.04}}, gate_ok)["rc"],
           RC_PASS)
    check_("wiring: F1 with an ungated term source fails",
           judge_wiring("F1", ov_a1, _facts(), s_f1, F0, audit_ok, {"F1": {"episode_reward": 0.04}}, gate_bad)["rc"],
           RC_FAIL)
    check_("wiring: F1 whose dumped func is another function fails",
           judge_wiring("F1", ov_a1, _facts(term_funcs={"reward_stand_stance_quality": "x:other"}), s_f1, F0, audit_ok,
                        {"F1": {"episode_reward": 0.04}}, gate_ok)["rc"], RC_FAIL)
    check_("wiring: F1 under the old term name reads a dead term and fails",
           judge_wiring("F1", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance": 0.05})),
                        F0, audit_ok, {"F1": {"episode_reward": 0.04}}, gate_ok)["rc"], RC_FAIL)
    s_f1b = _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance_quality": 0.05, F1_NONSTAND_TAGS[0]: 0.0}))
    check_("wiring: F1 paying only at stand passes", judge_wiring("F1", ov_a1, _facts(), s_f1b, F0, audit_ok,
                                                                  {"F1": {"episode_reward": 0.04}})["rc"], RC_PASS)
    s_f1c = _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance_quality": 0.05, F1_NONSTAND_TAGS[0]: 0.01}))
    check_("wiring: F1 paying outside the stand slot fails", judge_wiring("F1", ov_a1, _facts(), s_f1c, F0, audit_ok,
                                                                          {"F1": {"episode_reward": 0.04}})["rc"], RC_FAIL)

    # ------------------------------------------------------------------------------------------- stage p2b (CHARTER 10)
    cl_replay = {"p2b": {"CL": {"terms": {"reward_foot_clearance": {"episode_reward": 0.02},
                                          "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}
    s_cl = _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": 0.025,
                                       "Episode_Reward/penalty_swing_min_clearance": -0.012}))
    check_("wiring: CL both terms live and within 3x passes",
           judge_wiring("CL", ov_a1, _facts(), s_cl, F0, audit_ok, cl_replay)["rc"], RC_PASS)
    check_("wiring: CL with a dead clearance term fails",
           judge_wiring("CL", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": 0.0,
                        "Episode_Reward/penalty_swing_min_clearance": -0.012})), F0, audit_ok, cl_replay)["rc"], RC_FAIL)
    check_("wiring: CL with the clearance reward negative (wrong sign) fails",
           judge_wiring("CL", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": -0.02,
                        "Episode_Reward/penalty_swing_min_clearance": -0.012})), F0, audit_ok, cl_replay)["rc"], RC_FAIL)
    check_("wiring: CL with the swing penalty positive (wrong sign) fails",
           judge_wiring("CL", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": 0.02,
                        "Episode_Reward/penalty_swing_min_clearance": 0.012})), F0, audit_ok, cl_replay)["rc"], RC_FAIL)
    check_("wiring: CL at 3.5x the replay fails",
           judge_wiring("CL", ov_a1, _facts(), s_cl, F0, audit_ok,
                        {"p2b": {"CL": {"terms": {"reward_foot_clearance": {"episode_reward": 0.025 / 3.5},
                                                  "penalty_swing_min_clearance": {"episode_reward": -0.012}}}}})["rc"],
           RC_FAIL)
    check_("wiring: CL with no p2b replay record fails closed",
           judge_wiring("CL", ov_a1, _facts(), s_cl, F0, audit_ok, None)["rc"], RC_FAIL)
    tg_replay = {"p2b": {"TG": {"terms": {"reward_orientation": {"episode_reward": -0.05}}}}}
    s_tg = _ser(n=215, **dict(base, **{"Episode_Reward/reward_orientation": -0.045}))
    check_("wiring: TG live, >= 2x the seed tail and within 3x of the replay passes",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail={"reward_orientation": -0.02})["rc"],
           RC_PASS)
    check_("wiring: TG under 2x the seed's S1 p2a tail fails",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail={"reward_orientation": -0.03})["rc"],
           RC_FAIL)
    check_("wiring: TG at exactly 2x the tail passes (boundary)",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail={"reward_orientation": -0.0225})["rc"],
           RC_PASS)
    check_("wiring: TG with no seed tail fails closed",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail=None)["rc"], RC_FAIL)
    check_("wiring: TG dead term fails", judge_wiring("TG", ov_a1, _facts(), _ser(n=215, **dict(
        base, **{"Episode_Reward/reward_orientation": 0.0})), F0, audit_ok, tg_replay,
        ref_tail={"reward_orientation": -0.02})["rc"], RC_FAIL)
    check_("wiring: C0 has no arm-specific term (no fails from it)",
           judge_wiring("C0", ov_a1, _facts(), s, F0, audit_ok, None)["rc"], RC_PASS)
    check_("p2b_checkpoints: a model_19994 fork", p2b_checkpoints(19994), [21000, 22500, 24000])
    check_("seed_of", [seed_of("P2__CL_s16"), seed_of("P2__TG_s17"), seed_of("ANCW_s18")], [16, 17, 18])

    def lv(start, end, n=1100):
        return [(F0 + i, start + (end - start) * i / (n - 1)) for i in range(n)]
    T = OBST_LEVEL_TAGS[0]
    check_("terrain: a rising, unpinned obstacle level passes", judge_terrain({T: lv(1.0, 2.5)}, F0, 10)["rc"], RC_PASS)
    check_("terrain: before +1050 waits", judge_terrain({T: lv(1.0, 2.5, n=900)}, F0, 10)["rc"], RC_WAIT)
    check_("terrain: a flat level fails (not risen, no promotions)", judge_terrain({T: lv(1.0, 1.0)}, F0, 10)["rc"], RC_FAIL)
    check_("terrain: pinned at 0 fails", judge_terrain({T: lv(0.0, 0.04)}, F0, 10)["rc"], RC_FAIL)
    check_("terrain: pinned at the top (9) fails", judge_terrain({T: lv(8.9, 9.0)}, F0, 10)["rc"], RC_FAIL)
    check_("terrain: a promotion tag of zeros fails", judge_terrain({T: lv(1.0, 2.0), PROMO_TAGS[0]: [(F0 + i, 0.0)
                                                                                                    for i in range(1100)]}, F0, 10)["rc"], RC_FAIL)
    allt = {TAGS["levels_all"]: lv(1.0, 2.0)}
    check_("terrain: all-tile levels only (no obstacle tag) waits inside the grace", judge_terrain(allt, F0, 10)["rc"], RC_WAIT)
    check_("terrain: ... and fails closed past it", judge_terrain({TAGS["levels_all"]: lv(1.0, 2.0, n=1300)}, F0, 10)["rc"],
           RC_FAIL)
    EL, CF = TAGS["ep_len"], TAGS["crab_failure"]
    check_("collapse: before +300 is n/a", judge_collapse(_ser(n=250, **{EL: 100.0, CF: 0.9}), F0, 1800.0)["rc"], RC_PASS)
    check_("collapse: a healthy run passes", judge_collapse(_ser(n=400, **{EL: 1500.0, CF: 0.2}), F0, 1800.0)["rc"], RC_PASS)
    check_("collapse: episode length < 0.5 x source fails", judge_collapse(_ser(n=400, **{EL: 880.0, CF: 0.2}), F0,
                                                                           1800.0)["rc"], RC_FAIL)
    check_("collapse: episode length = 0.5 x source passes (boundary)",
           judge_collapse(_ser(n=400, **{EL: 900.0, CF: 0.2}), F0, 1800.0)["rc"], RC_PASS)
    check_("collapse: crab_failure 0.61 fails", judge_collapse(_ser(n=400, **{EL: 1500.0, CF: 0.61}), F0, 1800.0)["rc"],
           RC_FAIL)
    check_("collapse: no source tail waits", judge_collapse(_ser(n=400, **{EL: 1500.0, CF: 0.2}), F0, None)["rc"], RC_WAIT)
    check_("smoke: a full audit with PROMOTE realised and moving levels passes",
           judge_smoke(audit_ok, _facts(), {T: lv(1.0, 1.5)}, {"KRABBY_TERRAIN_PROMOTE": "0.75:0.4"}, True)["rc"], RC_PASS)
    check_("smoke: levels that never move fail", judge_smoke(audit_ok, _facts(), {T: lv(1.0, 1.0)}, {}, True)["rc"], RC_FAIL)
    check_("smoke: PROMOTE not audited fails", judge_smoke([("overlay KRABBY_OMNI_FACING=0.2:1.2", True)], _facts(), {}, {},
                                                           False)["rc"], RC_FAIL)
    check_("arm_of", [arm_of("P2__A1_s18"), arm_of("P2__CAL2_s18"), arm_of("P2__E1_s18"), arm_of("P2__CL_s16"),
                      arm_of("P2__TG_s17"), arm_of("P2__C0_s18")], ["A1", "CAL2", "E1", "CL", "TG", "C0"])
    em_ok = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    check_("preflight: A1 with both obstacle metrics emitted passes", judge_preflight("A1", em_ok, None, None)["rc"],
           RC_PASS)
    check_("preflight: A1 without the obstacle-level metric is refused",
           judge_preflight("A1", {"promote_count": ["x.py"]}, None, None)["rc"], RC_FAIL)
    check_("preflight: CAL2 without the promotion metric is refused",
           judge_preflight("CAL2", {"terrain_levels_obst": ["x.py"]}, None, None)["rc"], RC_FAIL)
    check_("preflight: CAL1 needs no obstacle telemetry", judge_preflight("CAL1", {}, None, None)["rc"], RC_PASS)
    check_("preflight: E1 without a replay magnitude is refused", judge_preflight("E1", em_ok, None, None)["rc"], RC_FAIL)
    check_("preflight: E1 with a replay magnitude passes",
           judge_preflight("E1", em_ok, {"E1": {"episode_reward": -0.02, "verdict": "PASS"}}, None)["rc"], RC_PASS)
    check_("preflight E1: a FAIL verdict without a user overrule refuses",
           judge_preflight("E1", em_ok, {"E1": {"episode_reward": -0.02, "verdict": "FAIL"}}, None)["rc"], RC_FAIL)
    check_("preflight E1: a FAIL verdict with a recorded user overrule passes",
           judge_preflight("E1", em_ok, {"E1": {"episode_reward": -0.02, "verdict": "FAIL",
                                                "overrule": {"by": "user", "date": "2026-10-06"}}}, None)["rc"], RC_PASS)
    check_("preflight: F1 with replay + construction passes",
           judge_preflight("F1", em_ok, {"F1": {"episode_reward": 0.04, "verdict": "PASS"}}, gate_ok)["rc"], RC_PASS)
    check_("preflight: F1 with an ungated term is refused",
           judge_preflight("F1", em_ok, {"F1": {"episode_reward": 0.04}}, gate_bad)["rc"], RC_FAIL)
    check_("preflight: CL without a p2b replay magnitude is refused", judge_preflight("CL", em_ok, None, None)["rc"],
           RC_FAIL)
    check_("preflight: CL with both p2b terms + a PASS verdict passes",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "PASS", "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02},
               "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}, None)["rc"], RC_PASS)
    check_("preflight: CL missing one p2b term is refused",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "PASS", "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02}}}}}, None)["rc"], RC_FAIL)
    check_("preflight: CL verdict FAIL without a user overrule is refused",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "FAIL", "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02},
               "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}, None)["rc"], RC_FAIL)
    check_("preflight: CL verdict FAIL with a recorded user overrule passes",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "FAIL", "overrule": {"by": "user"}, "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02},
               "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}, None)["rc"], RC_PASS)
    check_("preflight: TG with its p2b term + a PASS verdict passes",
           judge_preflight("TG", em_ok, {"p2b": {"TG": {"verdict": "PASS", "terms": {
               "reward_orientation": {"episode_reward": -0.05}}}}}, None)["rc"], RC_PASS)
    check_("preflight: TG without a p2b replay magnitude is refused", judge_preflight("TG", em_ok, None, None)["rc"],
           RC_FAIL)
    check_("preflight: C0 needs no p2b replay magnitude", judge_preflight("C0", em_ok, None, None)["rc"], RC_PASS)
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=Path)
    ap.add_argument("--check", type=Path)
    ap.add_argument("--smoke-audit", type=Path)
    ap.add_argument("--chain")
    ap.add_argument("--at", choices=("200", "1000", "collapse"))
    ap.add_argument("--train-log", default=None)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=CAMPAIGN / "candidates.yaml")
    ap.add_argument("--replay", type=Path, default=CAMPAIGN / "replay_gate.json")
    ap.add_argument("--require-terrain-move", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--code-root", type=Path, default=R)
    ap.add_argument("--code-guard", action="store_true")
    ap.add_argument("--lrout", type=Path, default=None)
    ap.add_argument("--anchor-name", default=None)
    ap.add_argument("--reanchor", action="store_true")
    ap.add_argument("--arms", type=Path, default=None,
                     help="p2b (TG): the arms dir to read the seed's P2__S1_s<seed>.record.json from "
                          "(default $P2_OUT/arms)")
    ap.add_argument("--p2b-checkpoints", type=int, default=None, metavar="FORK_MODEL_N",
                     help="print the p2b intermediate checkpoint numbers for a fork of model_<FORK_MODEL_N>.pt")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.arms is None:
        a.arms = Path(os.environ.get("P2_OUT") or OUT_DEFAULT) / "arms"
    if a.p2b_checkpoints is not None:
        print(" ".join(str(n) for n in p2b_checkpoints(a.p2b_checkpoints)))
        return 0
    if a.self_test:
        return _self_test()
    if a.code_guard:
        rc, res = code_guard(a)
        print(json.dumps(res, default=str))
        return rc
    if a.iters:
        f, l = first_last(read_tb(a.iters))
        print(f"{f if f is not None else 0} {l if l is not None else 0}")
        return 0
    if a.check:
        if not a.chain or not a.at:
            ap.error("--check needs --chain and --at")
        rc, res = check(a)
    elif a.smoke_audit:
        if not a.chain:
            ap.error("--smoke-audit needs --chain")
        rc, res = smoke_audit(a)
    elif a.preflight:
        if not a.chain:
            ap.error("--preflight needs --chain")
        rc, res = preflight(a)
    else:
        ap.error("one of --iters / --check / --smoke-audit / --preflight / --self-test")
    print(json.dumps(res, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())

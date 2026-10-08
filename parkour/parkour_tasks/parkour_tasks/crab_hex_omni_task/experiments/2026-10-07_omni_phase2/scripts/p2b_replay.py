#!/usr/bin/env python3
"""Omni phase 2 -- the stage-p2b OFFLINE REPLAY GATE (CHARTER 10 / P2B_PLAN.md step 2). CPU ONLY, read-only on raws.

Re-scores the SAVED eval raws of the three S1 p2a heads (``P2__S1_s{16,17,18}`` model_19994) with the two p2b
levers' reward terms, exactly as the landed code (commit c5e1b104) computes them, and judges the pre-registered
replay rules. It launches nothing (no Isaac, no GPU, no training, no eval).

INPUTS (per head)
  field_hard__omni / field_light__omni, env_seeds 1 + 2: ``$P2_OUT/lr/<head>/<row>/seed00k/<ts>/`` (newest complete
  stamp, the p2_mech / p2_read convention), 100 x 80-s episodes each;
  the flat suite: the arm record's ``driver_stamp`` (``arms/<head>.record.json``), 1000 x 15.5-s episodes.
  Each ``raw/episode_*.npz`` carries per recorded step t (the state BEFORE ``env.step`` of step t, i.e. the post
  state of step t-1): foot_pos_w, terrain_z (a downward raycast of the terrain mesh under each foot), foot_force_norm
  (= ``|net_forces_w_history[:, 0, foot]|``, the reward's own contact read), root_pos_w / root_quat_w /
  root_lin_vel_b / root_ang_vel_b, cmd_applied (the body command step t acts on), segment_id, crab_failure.

STEP ALIGNMENT (as ``experiments/tools/replay_motion_terms.step_series``)
  reward step s (0 .. T-2) = the command ``cmd_applied[s]`` and the POST state at raw index s+1 -- the reward manager
  computes after the physics step, before the command resamples.

TERMS (kernels, unweighted; params from :func:`p2b_params`, sources in the report)
  CL  reward_foot_clearance (``omni_clearance.reward_foot_clearance_terrain``):
        h = foot_z - terrain_z;  in_swing = foot_force_norm <= 0.1  (``~(norm > contact_force_threshold)``)
        k = mean_feet( in_swing * clamp(relu(h - min_clearance_m), max=max_clearance_m) ) * gate
      penalty_swing_min_clearance (``penalty_swing_min_clearance_terrain``):
        k = mean_feet( in_swing * (h < min_clearance_m) ) * gate
      gate = ``rewards._cmd_gate`` omni branch = ``max(|cmd_xy|, r_turn |cmd_wz|) > min_forward_speed_cmd``
      (strict >, r_turn = omni_clock_turn_radius); no parkour_flat mask (KRABBY_FOOT_CLEAR_FLAT 1 -> parkour_name None).
  TG  reward_orientation -> ``reward_orientation_upright``: k = sum(projected_gravity_b[:, :2]^2), every tile.
  TRACKING INCOME (named before any result was read): w_lin * track_lin_vel_xy_exp(std) at the S1 run's own dumped
      weight / std (1.25, 0.2): exp(-|cmd_xy - v_b_xy|^2 / std^2). Info only: + w_ang * track_ang_vel_z_exp (1.0, 0.5).

WINDOWS
  walking   reward steps whose command hold is not ``stand`` / ``settle`` (and has a segment), with the command gate ON
            (so turn holds count: under omni a turn command gates the CL terms on), and NOT in the 2 s before a fall.
  pre-fall  for an episode that ended in ``crab_failure`` at raw index i_fail: reward steps s >= i_fail - 100
            (post states i_fail-99 .. i_fail-1; the failing step's post state is never recorded) -- 2 s at dt 0.02.
  Shares    |w| * sum_walking(k dt) / sum_walking(tracking dt), ratio of sums over all of a head's walking steps
            (both field rows, both env_seeds, and the suite); per-row values are reported beside it.

CHECKS (CHARTER 10 / P2B_PLAN.md step 2)
  CL (a) Spearman(swing apex above terrain, the swing's own foot-clearance income) > 0 among pit swings AND among
         ordinary swings (``p2_mech.extract_swings``: its swing span, apex and pit-vs-ordinary rule), walking liftoffs,
         per head (where n >= MIN_RHO_N) and pooled.
     (b) counterfactual crouch: root z - 1 cm at fixed feet, re-scored -> |delta income| <= 10 % of the term's payoff
         for +1 cm terrain-relative lift of every swing foot (walking steps; per term, per head). The library
         root-relative form is scored beside it for contrast (info).
     (c) foot-clearance income share in [1 %, 10 %] of tracking;  (d) swing-min penalty share <= 5 %.
  TG (e) mean cost per step in the pre-fall windows >= 5x the mean per step over walking;  (f) share in [1 %, 5 %].
  RE-DOSE: start (FOOT_CLEAR_W 1.0, SWING_MIN_CLEAR_W -0.4, ORIENT_W -0.7); a share outside its band moves only that
  term's weight, to ONE weight across the three heads -- the weight closest to the start (2 s.f., rounded inward)
  that lands every head in band; none -> FAIL (the user decides).

EPISODE_REWARD (the +200 liveness magnitude, TensorBoard units: ``Episode_Reward/<term>`` = episodic weighted sum /
max_episode_length_s, 40 s in training)
  per episode: rate = sum_all_steps(w k dt) / duration; value = rate * L / 40, L = 40 s for an episode that ended
  on its schedule (an 80-s field episode is scaled DOWN to 40 s at its own rate; a 15.5-s suite episode stands for
  a full 40-s training episode at its own rate), L = min(duration, 40 s) for an episode that ended in a fall.
  Rows are mixed like the S1 training distribution: 0.5 flat (the suite) + 0.5 field, the field split 0.62 hard /
  0.38 light (P2B_PLAN context: 62 % of obstacle episodes at d >= 0.70). Calibrated against the S1 runs' own
  training tails of Episode_Reward/track_lin_vel_xy_exp and the p2a (flat-only) reward_orientation.

CAVEATS (also in REPLAY_GATE_P2B.md)
  * terrain_z is the eval's raycast of the terrain MESH; the live twin samples the height FIELD at the nearest pixel
    (omni_exposure.tiled_pixel_index). They agree on tops and pit floors and can differ within ~1 pixel (0.08 m) of a
    vertical edge. The fidelity check runs the real twins with this terrain sample substituted, so every other line
    of the code is exercised as written.
  * No composed p2b dry-run dump exists (building one needs Isaac); params are the S1 p2a dump with the CL / TG
    overlay applied by the forward cfg's ``__post_init__`` rules, read from source.

CLI:  p2b_replay.py [--write] [--workers 8] [--limit N] [--no-tb]
      --write updates replay_gate.json (a "p2b" key; E1/F1 untouched) and writes REPLAY_GATE_P2B.md.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import glob
import importlib.util
import json
import math
import os
import re
import sys
import textwrap
import types
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
CAMPAIGN = HERE.parents[1]
OMNI = HERE.parents[3]                                      # .../crab_hex_omni_task
TASKS = OMNI.parent                                         # .../parkour_tasks/parkour_tasks
PARKOUR = TASKS.parents[1]                                  # .../parkour
REPO = PARKOUR.parent
OUT = Path(os.environ.get("P2_OUT") or PARKOUR / "outputs/omni_plan/phase2")
LROUT = OUT / "lr"
ARMS = OUT / "arms"
RT_PY = OMNI / "experiments/tools/replay_motion_terms.py"
MECH_PY = HERE.parent / "p2_mech.py"
CLEARANCE_PY = OMNI / "mdp/omni_clearance.py"
LIB_REWARDS_PY = PARKOUR / "parkour_isaaclab/envs/mdp/rewards.py"
PARKOURS_DIR = PARKOUR / "parkour_isaaclab/envs/mdp/parkours"
FWD_MDP = TASKS / "crab_hex_forward_task/mdp"
FWD_MDP_CFG_PY = TASKS / "crab_hex_forward_task/config/crab_hex/agents/parkour_mdp_cfg.py"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
REPLAY_JSON = CAMPAIGN / "replay_gate.json"
REPORT_MD = CAMPAIGN / "REPLAY_GATE_P2B.md"

HEADS = ("P2__S1_s16", "P2__S1_s17", "P2__S1_s18")
FIELD_ROWS = ("field_hard__omni", "field_light__omni")
FIELD_SEEDS = (1, 2)
SUITE = "suite__omni"
ROWS = FIELD_ROWS + (SUITE,)
NON_WALK_HOLDS = ("stand", "settle")

PRE_FALL_S = 2.0
LIFT_M = 0.01            # counterfactual lift of every swing foot (b)
CROUCH_M = 0.01          # counterfactual root drop at fixed feet (b)
CROUCH_MAX_FRAC = 0.10   # (b)
CL_BAND = (0.01, 0.10)   # (c)
SM_BAND = (0.0, 0.05)    # (d)
TG_BAND = (0.01, 0.05)   # (f)
TG_PREFALL_RATIO = 5.0   # (e)
MIN_RHO_N = 8
#: float32 (live torch) vs float64 (replay) rounding; a flipped swing / micro-swing / floor decision moves a step's
#: value by >= 1/6 * 1e-3, so 1e-6 separates rounding from a real disagreement
FIDELITY_TOL = 1e-6
TRAIN_EP_S = 40.0
#: the S1 training row mix for the Episode_Reward estimate (CHARTER 3: FLAT_FRAC 0.5; P2B_PLAN: 62 % of obstacle
#: episodes at d >= 0.70)
ROW_MIX = {SUITE: 0.5, "field_hard__omni": 0.5 * 0.62, "field_light__omni": 0.5 * 0.38}
START_W = {"reward_foot_clearance": 1.0, "penalty_swing_min_clearance": -0.4, "reward_orientation": -0.7}
BANDS = {"reward_foot_clearance": CL_BAND, "penalty_swing_min_clearance": SM_BAND, "reward_orientation": TG_BAND}
TERM_KERNEL = {"reward_foot_clearance": "cl", "penalty_swing_min_clearance": "sm", "reward_orientation": "tg"}
KNOB_OF = {"reward_foot_clearance": "KRABBY_FOOT_CLEAR_W", "penalty_swing_min_clearance": "KRABBY_SWING_MIN_CLEAR_W",
           "reward_orientation": "KRABBY_ORIENT_W"}
ARM_TERMS = {"CL": ("reward_foot_clearance", "penalty_swing_min_clearance"), "TG": ("reward_orientation",)}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RT = _load(RT_PY, "replay_motion_terms__p2b")     # _yaml_load_tolerant only (read-only reuse)
MECH = _load(MECH_PY, "p2_mech__p2b")             # the swing / pit definitions (read-only reuse)


# ================================================================================================ params
def _const_from_source(path: Path, name: str) -> float:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return float(ast.literal_eval(node.value))
    raise SystemExit(f"{name} not found in {path}")


def _lineno(path: Path, pattern: str) -> int | None:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def p2b_params(env: dict, *, foot_clear_min: float = 0.03) -> dict:
    """Every number the p2b kernels use, with its source. ``env`` = the head's S1 p2a ``params/env.yaml`` dict; the
    CL / TG overlay (P2B_PLAN recipe, candidates v8) is applied by the forward cfg's own ``__post_init__`` rules:
    KRABBY_FOOT_CLEAR_MIN -> reward_foot_clearance.min_clearance_m, KRABBY_FOOT_CLEAR_FLAT=1 -> parkour_name None on
    both lift terms, the *_W knobs -> weights; the swing-min floor stays MIN_SWING_CLEARANCE_M (no knob moves it)."""
    rw = env["rewards"]
    fc, sm, ori = rw["reward_foot_clearance"], rw["penalty_swing_min_clearance"], rw["reward_orientation"]
    tl, ta = rw["track_lin_vel_xy_exp"], rw["track_ang_vel_z_exp"]
    cfg = _rel(FWD_MDP_CFG_PY)
    p = {
        "dt": float(env["decimation"]) * float(env["sim"]["dt"]),
        "episode_length_s": float(env["episode_length_s"]),
        "omni_command": bool(env.get("omni_command")),
        "turn_r": float(env["omni_clock_turn_radius"]),
        "fc_contact": float(fc["params"]["contact_force_threshold"]),
        "fc_min": float(foot_clear_min),
        "fc_max": float(fc["params"]["max_clearance_m"]),
        "fc_gate": float(fc["params"]["min_forward_speed_cmd"]),
        "fc_parkour_name": None,
        "sm_contact": float(sm["params"]["contact_force_threshold"]),
        "sm_min": float(sm["params"]["min_clearance_m"]),
        "sm_gate": float(sm["params"]["min_forward_speed_cmd"]),
        "sm_parkour_name": None,
        "tg_func": "reward_orientation_upright",
        "p2a_orient_w": float(ori["weight"]),
        "p2a_orient_func": str(ori["func"]),
        "track_lin_w": float(tl["weight"]), "track_lin_std": float(tl["params"]["std"]),
        "track_ang_w": float(ta["weight"]), "track_ang_std": float(ta["params"]["std"]),
        "p2a_fc_min": float(fc["params"]["min_clearance_m"]),
        "p2a_fc_parkour_name": fc["params"].get("parkour_name"),
        "p2a_fc_w": float(fc["weight"]), "p2a_sm_w": float(sm["weight"]),
        "ground_offset_from_root_m": float(fc["params"]["ground_offset_from_root_m"]),
    }
    src_max = _const_from_source(FWD_MDP_CFG_PY, "MAX_CLEARANCE_M")
    src_sm = _const_from_source(FWD_MDP_CFG_PY, "MIN_SWING_CLEARANCE_M")
    p["checks"] = {
        "max_clearance_m dump == MAX_CLEARANCE_M source": p["fc_max"] == src_max,
        "swing-min floor dump == MIN_SWING_CLEARANCE_M source": p["sm_min"] == src_sm,
        "omni_command (the _cmd_gate omni branch)": p["omni_command"],
        "contact thresholds 0.1": p["fc_contact"] == 0.1 and p["sm_contact"] == 0.1,
        "gate thresholds 0.12": p["fc_gate"] == 0.12 and p["sm_gate"] == 0.12,
        "dt 0.02": abs(p["dt"] - 0.02) < 1e-12,
        "episode_length_s 40": p["episode_length_s"] == 40.0,
    }
    p["sources"] = {
        "reward_foot_clearance.weight": f"KRABBY_FOOT_CLEAR_W start 1.0 (P2B_PLAN recipe; {cfg}:{_lineno(FWD_MDP_CFG_PY, 'KRABBY_FOOT_CLEAR_W.:')} overrides table); p2a dump {p['p2a_fc_w']}",
        "contact_force_threshold": f"{p['fc_contact']} ({cfg}:{_lineno(FWD_MDP_CFG_PY, 'contact_force_threshold.: 0.1')}..; dump, both terms)",
        "min_clearance_m (income floor)": f"KRABBY_FOOT_CLEAR_MIN 0.03 ({cfg}:{_lineno(FWD_MDP_CFG_PY, '_fcm = os.environ.get')}); p2a dump {p['p2a_fc_min']} (MIN_CLEARANCE_M)",
        "max_clearance_m (clamp)": f"{p['fc_max']} = MAX_CLEARANCE_M ({cfg}:{_lineno(FWD_MDP_CFG_PY, '^MAX_CLEARANCE_M')}); dump {p['fc_max']}",
        "min_forward_speed_cmd": f"{p['fc_gate']} (dump, both terms)",
        "gate": f"rewards._cmd_gate omni branch ({_rel(LIB_REWARDS_PY)}:{_lineno(LIB_REWARDS_PY, 'def _cmd_gate')}): max(|cmd_xy|, {p['turn_r']}*|wz|) > thr; omni_clock_turn_radius {p['turn_r']} (dump)",
        "parkour_name": f"None under KRABBY_FOOT_CLEAR_FLAT=1 ({cfg}:{_lineno(FWD_MDP_CFG_PY, 'KRABBY_FOOT_CLEAR_FLAT.. == .1.')}); p2a dump {p['p2a_fc_parkour_name']!r}",
        "swing-min weight": f"KRABBY_SWING_MIN_CLEAR_W start -0.4 (P2B_PLAN recipe); p2a dump {p['p2a_sm_w']}",
        "swing-min floor": f"{p['sm_min']} = MIN_SWING_CLEARANCE_M ({cfg}:{_lineno(FWD_MDP_CFG_PY, '^MIN_SWING_CLEARANCE_M')}); dump {p['sm_min']}",
        "TG weight": f"KRABBY_ORIENT_W start -0.7 = the p2a dump weight {p['p2a_orient_w']}",
        "TG func": f"reward_orientation_upright ({_rel(LIB_REWARDS_PY)}:{_lineno(LIB_REWARDS_PY, 'def reward_orientation_upright')}), via KRABBY_ORIENT_ALL_TILES ({_rel(MOTION_KNOBS_PY)} FUNC_SWAPS); p2a dump func {p['p2a_orient_func']}",
        "tracking": f"track_lin_vel_xy_exp w {p['track_lin_w']} std {p['track_lin_std']}; info track_ang_vel_z_exp w {p['track_ang_w']} std {p['track_ang_std']} (dump)",
        "dt / episode": f"decimation {env['decimation']} x sim.dt {env['sim']['dt']} = {p['dt']}; episode_length_s {p['episode_length_s']} (dump)",
    }
    return p


# ================================================================================================ pure kernels
def clock_speed(cmd: np.ndarray, turn_r: float) -> np.ndarray:
    """``crab_hex_clock_reward.clock_command_speed`` (planar): max(|cmd_xy|, r_turn |cmd_wz|)."""
    cmd = np.asarray(cmd, dtype=np.float64)
    return np.maximum(np.linalg.norm(cmd[:, :2], axis=1), turn_r * np.abs(cmd[:, 2]))


def cmd_gate(cmd: np.ndarray, thresh: float, turn_r: float, omni: bool = True) -> np.ndarray:
    """``rewards._cmd_gate(..., signed=True)``: the omni branch, else the forward literal ``cmd[:, 0] > thresh``."""
    if omni:
        return clock_speed(cmd, turn_r) > thresh
    return np.asarray(cmd)[:, 0] > thresh


def foot_height_terrain(foot_z: np.ndarray, terrain_z: np.ndarray, root_z=None) -> np.ndarray:
    """Terrain-relative foot height (``omni_clearance.foot_height_above_terrain``); ``root_z`` is accepted to make the
    crouch counterfactual explicit and is not used -- the twin never reads the root."""
    del root_z
    return np.asarray(foot_z, dtype=np.float32) - np.asarray(terrain_z, dtype=np.float32)


def foot_height_root(foot_z: np.ndarray, root_z: np.ndarray, offset: float) -> np.ndarray:
    """The library's root-relative height (``_foot_swing_height_above_ground``): foot_z - (root_z + offset)."""
    return (np.asarray(foot_z, dtype=np.float32)
            - (np.asarray(root_z, dtype=np.float32)[:, None] + np.float32(offset)))


def in_swing(force: np.ndarray, thr: float) -> np.ndarray:
    return ~(np.asarray(force) > thr)


def clearance_per_foot(h: np.ndarray, swing: np.ndarray, min_c: float, max_c: float) -> np.ndarray:
    """(S, F): in_swing * clamp(relu(h - min), max=max) -- before the mean over feet and the gate."""
    clr = np.minimum(np.maximum(np.asarray(h, dtype=np.float64) - min_c, 0.0), max_c)
    return swing.astype(np.float64) * clr


def k_foot_clearance(h, swing, gate, min_c, max_c) -> np.ndarray:
    return clearance_per_foot(h, swing, min_c, max_c).mean(axis=1) * gate


def k_swing_min(h, swing, gate, min_c) -> np.ndarray:
    return (swing & (np.asarray(h) < min_c)).astype(np.float64).mean(axis=1) * gate


def k_orientation(quat: np.ndarray) -> np.ndarray:
    g = MECH.projected_gravity_b(np.asarray(quat, dtype=np.float64))
    return np.sum(np.square(g[..., :2]), axis=-1)


def k_track_lin(cmd, v_b, std) -> np.ndarray:
    err = np.sum(np.square(np.asarray(cmd, dtype=np.float64)[:, :2] - np.asarray(v_b, dtype=np.float64)[:, :2]), axis=1)
    return np.exp(-err / std ** 2)


def k_track_ang(cmd, w_b, std) -> np.ndarray:
    return np.exp(-np.square(np.asarray(cmd, dtype=np.float64)[:, 2] - np.asarray(w_b, dtype=np.float64)[:, 2]) / std ** 2)


def rankdata(x) -> np.ndarray:
    """Average ranks (ties share the mean rank), vectorised."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    first = np.concatenate([[True], sx[1:] != sx[:-1]])
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    ends = np.concatenate([starts[1:], [x.size]])
    avg = 0.5 * (starts + ends - 1) + 1.0
    r = np.empty(x.size)
    r[order] = avg[grp]
    return r


def spearman(x, y, min_n: int = MIN_RHO_N):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < min_n:
        return None, n
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None, n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def redose(start: float, band: tuple[float, float], ratios: list[float]) -> dict:
    """``ratios[i]`` = head i's share per unit |weight|. The |weight| closest to |start| that lands every head in
    ``band`` (2 s.f., rounded inward), sign kept; ``feasible`` False when the heads' intervals do not overlap."""
    lo_b, hi_b = band
    pos = [r for r in ratios if r is not None and r > 0]
    if lo_b > 0 and len(pos) < len(ratios):          # a head whose term is 0 can never reach a non-zero floor
        return {"feasible": False, "weight": None, "interval_abs": [None, None], "moved": None}
    lo = max(lo_b / r for r in pos) if lo_b > 0 else 0.0
    hi = min(hi_b / r for r in pos) if pos else math.inf
    a0 = abs(start)
    if lo > hi:
        return {"feasible": False, "weight": None, "interval_abs": [lo, hi], "moved": None}
    if lo <= a0 <= hi:
        return {"feasible": True, "weight": start, "interval_abs": [lo, hi], "moved": False}
    target = lo if a0 < lo else hi

    def sf2(v, up):
        if v <= 0:
            return 0.0
        e = math.floor(math.log10(v)) - 1
        q = v / 10 ** e
        q = math.ceil(q - 1e-9) if up else math.floor(q + 1e-9)
        return float(f"{q * 10 ** e:.10g}")

    a = sf2(target, up=(a0 < lo))
    if not (lo - 1e-12 <= a <= hi + 1e-12):
        a = target
    return {"feasible": True, "weight": math.copysign(a, start), "interval_abs": [lo, hi], "moved": True}


# ================================================================================================ one episode
def replay_episode(z: dict, hold_labels: list[str], p: dict, *, want_swings: bool = True) -> dict:
    """One episode's sums (kernels x dt, unweighted) over the walking / pre-fall / all-step windows, the crouch and
    lift counterfactuals and the per-swing income; see the module docstring for every definition."""
    dt = p["dt"]
    force = np.asarray(z["foot_force_norm"])
    T = force.shape[0]
    S = T - 1
    out: dict = {"S": int(S)}
    if S < 1:
        out["empty"] = True
        return out
    cmd = np.asarray(z["cmd_applied"], dtype=np.float64)[:S]
    seg = np.asarray(z["segment_id"]).astype(int)
    post = slice(1, T)
    fz = np.asarray(z["foot_pos_w"])[post, :, 2]
    tz = np.asarray(z["terrain_z"])[post]
    rz = np.asarray(z["root_pos_w"])[post, 2]
    swing_fc = in_swing(force[post], p["fc_contact"])
    swing_sm = in_swing(force[post], p["sm_contact"])
    gate_fc = cmd_gate(cmd, p["fc_gate"], p["turn_r"], p["omni_command"]).astype(np.float64)
    gate_sm = cmd_gate(cmd, p["sm_gate"], p["turn_r"], p["omni_command"]).astype(np.float64)

    h = foot_height_terrain(fz, tz, rz)
    per_foot = clearance_per_foot(h, swing_fc, p["fc_min"], p["fc_max"]) * gate_fc[:, None]
    cl = per_foot.mean(axis=1)
    sm = k_swing_min(h, swing_sm, gate_sm, p["sm_min"])
    tg = k_orientation(np.asarray(z["root_quat_w"])[post])
    trk_lin = k_track_lin(cmd, np.asarray(z["root_lin_vel_b"])[post], p["track_lin_std"])
    trk_ang = k_track_ang(cmd, np.asarray(z["root_ang_vel_b"])[post], p["track_ang_std"])

    # counterfactuals (b): +1 cm on every swing foot (reward's own swing test, per term) / root -1 cm at fixed feet
    h_lift_fc = h + np.float32(LIFT_M) * swing_fc
    h_lift_sm = h + np.float32(LIFT_M) * swing_sm
    cl_lift = k_foot_clearance(h_lift_fc, swing_fc, gate_fc, p["fc_min"], p["fc_max"])
    sm_lift = k_swing_min(h_lift_sm, swing_sm, gate_sm, p["sm_min"])
    h_crouch = foot_height_terrain(fz, tz, rz - CROUCH_M)
    cl_crouch = k_foot_clearance(h_crouch, swing_fc, gate_fc, p["fc_min"], p["fc_max"])
    sm_crouch = k_swing_min(h_crouch, swing_sm, gate_sm, p["sm_min"])
    off = p["ground_offset_from_root_m"]
    h_root = foot_height_root(fz, rz, off)
    h_root_c = foot_height_root(fz, rz - CROUCH_M, off)
    cl_root = k_foot_clearance(h_root, swing_fc, gate_fc, p["fc_min"], p["fc_max"])
    sm_root = k_swing_min(h_root, swing_sm, gate_sm, p["sm_min"])
    cl_root_c = k_foot_clearance(h_root_c, swing_fc, gate_fc, p["fc_min"], p["fc_max"])
    sm_root_c = k_swing_min(h_root_c, swing_sm, gate_sm, p["sm_min"])
    cl_root_lift = k_foot_clearance(foot_height_root(fz, rz, off) + np.float32(LIFT_M) * swing_fc, swing_fc, gate_fc,
                                    p["fc_min"], p["fc_max"])
    sm_root_lift = k_swing_min(h_root + np.float32(LIFT_M) * swing_sm, swing_sm, gate_sm, p["sm_min"])

    # windows
    labels = np.array(list(hold_labels) + ["<none>"])
    seg_s = seg[:S]
    lab = labels[np.where(seg_s >= 0, seg_s, len(hold_labels))]
    cf = np.asarray(z["crab_failure"]).astype(bool)
    fell = bool(cf.any())
    i_fail = int(np.flatnonzero(cf)[-1]) if fell else None
    pre_n = int(round(PRE_FALL_S / dt))
    prefall = np.zeros(S, dtype=bool)
    if fell:
        prefall[max(0, i_fail - pre_n):] = True
    gate_any = gate_fc > 0
    walk = (~np.isin(lab, NON_WALK_HOLDS)) & (lab != "<none>") & gate_any & ~prefall

    def s(v, m=None):
        return float((v if m is None else v[m]).sum() * dt)

    series = {"cl": cl, "sm": sm, "tg": tg, "trk_lin": trk_lin, "trk_ang": trk_ang}
    out.update({
        "fell": fell, "i_fail": i_fail, "duration_s": S * dt,
        "n_walk": int(walk.sum()), "n_prefall": int(prefall.sum()),
        "walk": {k: s(v, walk) for k, v in series.items()},
        "all": {k: s(v) for k, v in series.items()},
        "prefall": {"tg": s(tg, prefall)},
        "cf": {
            "cl_lift": s(cl_lift - cl, walk), "sm_lift": s(sm_lift - sm, walk),
            "cl_crouch": s(cl_crouch - cl, walk), "sm_crouch": s(sm_crouch - sm, walk),
            "cl_root_crouch": s(cl_root_c - cl_root, walk), "sm_root_crouch": s(sm_root_c - sm_root, walk),
            "cl_root_lift": s(cl_root_lift - cl_root, walk), "sm_root_lift": s(sm_root_lift - sm_root, walk),
        },
        "walk_by_hold": {},
    })
    for hl in np.unique(lab[walk]):
        m = walk & (lab == hl)
        out["walk_by_hold"][str(hl)] = {"n": int(m.sum()), **{k: s(v, m) for k, v in series.items()}}
    if want_swings:
        sw = MECH.extract_swings({"foot_force_norm": force, "foot_pos_w": np.asarray(z["foot_pos_w"]),
                                  "terrain_z": np.asarray(z["terrain_z"]), "segment_id": seg}, list(hold_labels))
        rows = []
        for r in sw:
            a, b, f = int(r["start"]), int(r["end"]), int(r["foot"])
            s0, s1 = max(a - 1, 0), max(b - 1, 0)          # raw index t -> reward step t-1 (post state t)
            if s1 <= s0 or s0 >= S:
                continue
            inc = float(per_foot[s0:s1, f].sum() * dt / per_foot.shape[1])
            rows.append((r["apex_m"], inc, 1.0 if r["kind"] == "pit" else 0.0, 1.0 if walk[s0] else 0.0,
                         r["liftoff_rel_m"]))
        out["swings"] = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    return out


# ================================================================================================ inputs
def newest_stamp(head: str, row: str, seed: int) -> str | None:
    return MECH.newest_stamp(head, row, seed, LROUT)


def driver_stamp(head: str) -> str | None:
    rec = ARMS / f"{head}.record.json"
    try:
        return json.loads(rec.read_text()).get("driver_stamp")
    except (OSError, ValueError):
        return None


def episode_jobs(head: str, limit: int | None = None) -> tuple[list[tuple], list[dict]]:
    """``[(head, row, stamp, npz_path, reason, hold_labels)]`` and the stamps used."""
    jobs, used = [], []
    for row in FIELD_ROWS:
        for seed in FIELD_SEEDS:
            st = newest_stamp(head, row, seed)
            if st is None:
                raise SystemExit(f"{head} {row} seed{seed:03d}: no complete stamp under {LROUT}")
            meta = json.loads((Path(st) / "run_meta.json").read_text())
            reasons = meta["termination_reason"]
            n = len(reasons) if limit is None else min(limit, len(reasons))
            used.append({"head": head, "row": row, "seed": seed, "stamp": _rel(Path(st)),
                         "checkpoint_sha256": meta.get("checkpoint_sha256"), "n": n,
                         "terrain_source": meta.get("terrain_source"), "dt": meta.get("dt"),
                         "episode_length_s": (meta.get("env_overrides") or {}).get("episode_length_s")})
            for i in range(n):
                jobs.append((head, row, st, str(Path(st) / "raw" / f"episode_{i:02d}.npz"), reasons[i],
                             meta["hold_labels"]))
    st = driver_stamp(head)
    if not st or not (Path(st) / "complete.json").is_file():
        raise SystemExit(f"{head}: driver_stamp {st!r} missing or incomplete")
    meta = json.loads((Path(st) / "run_meta.json").read_text())
    emap = meta["env_index_map"]
    n = len(emap) if limit is None else min(limit, len(emap))
    used.append({"head": head, "row": SUITE, "seed": 1, "stamp": _rel(Path(st)),
                 "checkpoint_sha256": meta.get("checkpoint_sha256"), "n": n,
                 "terrain_source": meta.get("terrain_source"), "dt": meta.get("dt"),
                 "episode_length_s": (meta.get("env_overrides") or {}).get("episode_length_s")})
    for ent in emap[:n]:
        idx = int(ent["env_index"])
        jobs.append((head, SUITE, st, str(Path(st) / "raw" / f"episode_{idx:02d}.npz"),
                     meta["termination_reason"][idx], meta["hold_labels"]))
    return jobs, used


_P_BY_HEAD: dict = {}


def _job(args):
    head, row, st, path, reason, hold_labels = args
    with np.load(path) as npz:
        z = {k: npz[k] for k in ("foot_force_norm", "foot_pos_w", "terrain_z", "root_pos_w", "root_quat_w",
                                 "root_lin_vel_b", "root_ang_vel_b", "cmd_applied", "segment_id", "crab_failure")}
    r = replay_episode(z, hold_labels, _P_BY_HEAD[head])
    r.update(head=head, row=row, reason=reason, path=path)
    return r


def head_env(head: str) -> tuple[dict, str]:
    rec = json.loads((ARMS / f"{head}.record.json").read_text())
    env_yaml = Path(rec["run_dir"]) / "params" / "env.yaml"
    return RT._yaml_load_tolerant(env_yaml), str(env_yaml)


# ================================================================================================ fidelity check
def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Load a package module from its real source file without executing its parents' ``__init__`` (the
    ``tests/unit/test_omni_p2b_knobs.py`` loader)."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    parts = dotted.split(".")
    directory = path.parent
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(directory)]
            sys.modules[name] = pkg
        elif str(directory) not in getattr(pkg, "__path__", []):
            pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
        directory = directory.parent
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


def _source_of(path: Path, name: str) -> str:
    src = path.read_text()
    node = [n for n in ast.parse(src, str(path)).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


def real_terms():
    """The landed twins (``omni_clearance``) plus the library's ``_cmd_gate`` / ``_clock_cmd_speed`` /
    ``_parkour_flat_mask`` / ``reward_orientation_upright`` exec'd from rewards.py's source (it cannot import without
    Kit). Returns ``(omni_clearance module, library namespace)``."""
    import torch  # noqa: PLC0415

    _load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
    oc = _load_real_module("parkour_tasks.crab_hex_omni_task.mdp.omni_clearance", CLEARANCE_PY)
    if str(FWD_MDP) not in sys.path:
        sys.path.insert(0, str(FWD_MDP))
    import crab_hex_clock_reward as clock  # noqa: PLC0415

    ns = {"torch": torch, "SceneEntityCfg": lambda *a, **k: None, "Articulation": object, "ParkourEvent": object,
          "ParkourManagerBasedRLEnv": object, "clock_command_speed": clock.clock_command_speed}
    for name in ("_clock_cmd_speed", "_cmd_gate", "_parkour_flat_mask", "reward_orientation_upright"):
        exec(compile("from __future__ import annotations\n" + _source_of(LIB_REWARDS_PY, name),  # noqa: S102
                     str(LIB_REWARDS_PY), "exec"), ns)
    return oc, ns


def _gravity_b_matrix(q):
    """projected_gravity_b by an independent route (the rotation matrix of the wxyz quaternion, transposed, applied
    to (0, 0, -1)) -- not :func:`p2_mech.projected_gravity_b`, which this module's kernel uses."""
    import torch  # noqa: PLC0415

    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    # third row of R (world z axis expressed in body frame = R^T e_z = third row of R)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return -torch.stack([r20, r21, r22], dim=-1)


def fidelity_check(paths: list[str], p: dict, hold_labels: list[str]) -> dict:
    """Run the REAL landed twins on recorded steps (the terrain sample replaced by the raws' terrain_z, the only
    substitution) and the exec'd ``reward_orientation_upright``; compare with this module's kernels."""
    import torch  # noqa: PLC0415

    oc, lib = real_terms()
    feet = list(range(1, 7))
    asset = types.SimpleNamespace(name="robot", body_ids=feet)
    sensor = types.SimpleNamespace(name="contact_forces", body_ids=feet)
    worst = {"cl": 0.0, "sm": 0.0, "tg": 0.0}
    n = 0
    saved = (oc._lib_rewards, oc._foot_height_terrain)
    try:
        oc._lib_rewards = lambda: types.SimpleNamespace(**lib)
        for path in paths:
            with np.load(path) as npz:
                z = {k: npz[k] for k in npz.files}
            T = z["foot_force_norm"].shape[0]
            S = T - 1
            post = slice(1, T)
            body = torch.zeros(S, 7, 3)
            body[:, feet] = torch.as_tensor(z["foot_pos_w"][post])
            forces = torch.zeros(S, 2, 7, 3)
            forces[:, 0, feet, 2] = torch.as_tensor(z["foot_force_norm"][post])   # norm preserved on one axis
            tz = torch.as_tensor(z["terrain_z"][post])
            oc._foot_height_terrain = lambda env, a, ids, _tz=tz: env.scene["robot"].data.body_pos_w[:, ids, 2] - _tz
            pg = _gravity_b_matrix(torch.as_tensor(z["root_quat_w"][post], dtype=torch.float64))
            robot = types.SimpleNamespace(data=types.SimpleNamespace(body_pos_w=body, projected_gravity_b=pg))
            scene = {"robot": robot}
            scene_obj = type("Scene", (dict,), {})(scene)
            scene_obj.sensors = {"contact_forces": types.SimpleNamespace(data=types.SimpleNamespace(
                net_forces_w_history=forces))}
            command = torch.as_tensor(z["cmd_applied"][:S])
            env = types.SimpleNamespace(
                num_envs=S, device="cpu", scene=scene_obj,
                cfg=types.SimpleNamespace(omni_command=p["omni_command"], omni_clock_turn_radius=p["turn_r"]),
                command_manager=types.SimpleNamespace(get_command=lambda name, _c=command: _c))
            real_cl = oc.reward_foot_clearance_terrain(
                env, asset, sensor, contact_force_threshold=p["fc_contact"], min_clearance_m=p["fc_min"],
                max_clearance_m=p["fc_max"], min_forward_speed_cmd=p["fc_gate"],
                ground_offset_from_root_m=p["ground_offset_from_root_m"], parkour_name=None).double().numpy()
            real_sm = oc.penalty_swing_min_clearance_terrain(
                env, asset, sensor, contact_force_threshold=p["sm_contact"], min_clearance_m=p["sm_min"],
                min_forward_speed_cmd=p["sm_gate"], ground_offset_from_root_m=p["ground_offset_from_root_m"],
                parkour_name=None).double().numpy()
            real_tg = lib["reward_orientation_upright"](env, "base_parkour", asset).double().numpy()
            # this module's kernels on the same steps
            cmd = z["cmd_applied"][:S].astype(np.float64)
            h = foot_height_terrain(z["foot_pos_w"][post, :, 2], z["terrain_z"][post])
            g_fc = cmd_gate(cmd, p["fc_gate"], p["turn_r"], p["omni_command"]).astype(np.float64)
            my_cl = k_foot_clearance(h, in_swing(z["foot_force_norm"][post], p["fc_contact"]), g_fc, p["fc_min"], p["fc_max"])
            g_sm = cmd_gate(cmd, p["sm_gate"], p["turn_r"], p["omni_command"]).astype(np.float64)
            my_sm = k_swing_min(h, in_swing(z["foot_force_norm"][post], p["sm_contact"]), g_sm, p["sm_min"])
            my_tg = k_orientation(z["root_quat_w"][post])
            worst["cl"] = max(worst["cl"], float(np.abs(real_cl - my_cl).max()))
            worst["sm"] = max(worst["sm"], float(np.abs(real_sm - my_sm).max()))
            worst["tg"] = max(worst["tg"], float(np.abs(real_tg - my_tg).max()))
            n += S
    finally:
        oc._lib_rewards, oc._foot_height_terrain = saved
    return {"n_steps": n, "n_episodes": len(paths), "max_abs_diff": worst,
            "tolerance": FIDELITY_TOL, "ok": all(v < FIDELITY_TOL for v in worst.values())}


# ================================================================================================ aggregation
def _sum(eps, key, sub):
    return float(sum(e[key][sub] for e in eps if not e.get("empty")))


def shares(eps: list[dict], w: dict) -> dict:
    trk = _sum(eps, "walk", "trk_lin")
    out = {"n_walk_steps": int(sum(e.get("n_walk", 0) for e in eps)), "tracking_lin_k_s": trk}
    lin_w = eps[0]["_p"]["track_lin_w"] if eps else 1.25
    ang_w = eps[0]["_p"]["track_ang_w"] if eps else 1.0
    inc = lin_w * trk
    inc_ang = inc + ang_w * _sum(eps, "walk", "trk_ang")
    out["tracking_income_s"] = inc
    out["tracking_income_lin_ang_s"] = inc_ang
    for term, kk in TERM_KERNEL.items():
        ksum = _sum(eps, "walk", kk)
        out[term] = {
            "kernel_s": ksum,
            "ratio_per_unit_w": ksum / inc if inc else None,
            "share": abs(w[term]) * ksum / inc if inc else None,
            "share_lin_ang": abs(w[term]) * ksum / inc_ang if inc_ang else None,
        }
    return out


def tg_prefall(eps: list[dict]) -> dict:
    pre_k = sum(e["prefall"]["tg"] for e in eps if not e.get("empty"))
    pre_n = sum(e.get("n_prefall", 0) for e in eps)
    walk_k = _sum(eps, "walk", "tg")
    walk_n = sum(e.get("n_walk", 0) for e in eps)
    n_falls = sum(1 for e in eps if e.get("fell"))
    pre_mean = pre_k / pre_n if pre_n else None
    walk_mean = walk_k / walk_n if walk_n else None
    ratio = pre_mean / walk_mean if (pre_mean is not None and walk_mean) else None
    return {"n_falls": n_falls, "n_prefall_steps": int(pre_n), "n_walk_steps": int(walk_n),
            "prefall_mean_tilt2_per_step": pre_mean / 0.02 if pre_mean is not None else None,
            "walk_mean_tilt2_per_step": walk_mean / 0.02 if walk_mean is not None else None,
            "ratio": ratio, "pass": (ratio is not None and ratio >= TG_PREFALL_RATIO)}


def crouch(eps: list[dict]) -> dict:
    cf = {k: sum(e["cf"][k] for e in eps if not e.get("empty")) for k in eps[0]["cf"]}
    out = {}
    for term, key in (("reward_foot_clearance", "cl"), ("penalty_swing_min_clearance", "sm")):
        lift, cr = cf[f"{key}_lift"], cf[f"{key}_crouch"]
        frac = abs(cr) / abs(lift) if lift else None
        out[term] = {"lift_payoff_k_s": lift, "crouch_change_k_s": cr, "frac": frac,
                     "pass": frac is not None and frac <= CROUCH_MAX_FRAC,
                     "root_form_lift_k_s": cf[f"{key}_root_lift"], "root_form_crouch_k_s": cf[f"{key}_root_crouch"],
                     "root_form_frac": (abs(cf[f"{key}_root_crouch"]) / abs(cf[f"{key}_root_lift"])
                                        if cf[f"{key}_root_lift"] else None)}
    return out


def swing_rho(eps: list[dict]) -> dict:
    arrs = [e["swings"] for e in eps if not e.get("empty") and e["swings"].size]
    a = np.concatenate(arrs) if arrs else np.zeros((0, 5))
    a = a[a[:, 3] > 0]                                      # walking liftoffs
    out = {}
    for kind, flag in (("pit", 1.0), ("ordinary", 0.0)):
        sub = a[a[:, 2] == flag]
        rho, n = spearman(sub[:, 0], sub[:, 1])
        out[kind] = {"n": n, "rho": rho,
                     "apex_median_m": float(np.median(sub[:, 0])) if n else None,
                     "income_k_median": float(np.median(sub[:, 1])) if n else None,
                     "frac_paying": float((sub[:, 1] > 0).mean()) if n else None}
    return out


def episode_reward(eps_by_row: dict, w: float, kernel: str) -> dict:
    """Signed Episode_Reward (TB units) per row and mixed (ROW_MIX); see the module docstring."""
    per_row = {}
    for row, eps in eps_by_row.items():
        vals = []
        for e in eps:
            if e.get("empty") or e["duration_s"] <= 0:
                continue
            rate = w * e["all"][kernel] / e["duration_s"]
            L = min(e["duration_s"], TRAIN_EP_S) if e.get("fell") else TRAIN_EP_S
            vals.append(rate * L / TRAIN_EP_S)
        per_row[row] = float(np.mean(vals)) if vals else None
    mixed = sum(ROW_MIX[r] * v for r, v in per_row.items() if v is not None)
    wsum = sum(ROW_MIX[r] for r, v in per_row.items() if v is not None)
    return {"per_row": per_row, "mixed": mixed / wsum if wsum else None}


def tb_tails(run_dirs: dict, tags=("Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/reward_orientation",
                                   "Train/mean_episode_length"), n: int = 500) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415
    except ImportError:
        return {}
    out = {}
    for head, rd in run_dirs.items():
        ev = sorted(glob.glob(str(Path(rd) / "events.out.tfevents.*")))
        if not ev:
            continue
        ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags().get("scalars", []))
        out[head] = {t: float(np.mean([x.value for x in ea.Scalars(t)[-n:]])) for t in tags if t in have}
    return out


# ================================================================================================ the gate
def run(workers: int = 8, limit: int | None = None, tb: bool = True) -> dict:
    params, envsrc, jobs, used = {}, {}, [], []
    for head in HEADS:
        env, src = head_env(head)
        params[head] = p2b_params(env)
        envsrc[head] = src
        j, u = episode_jobs(head, limit)
        jobs += j
        used += u
    # unchanged params identical across the three seeds' dumps
    keys = [k for k in params[HEADS[0]] if k not in ("sources", "checks")]
    same = {k: all(params[h][k] == params[HEADS[0]][k] for h in HEADS) for k in keys}
    _P_BY_HEAD.update(params)
    with Pool(workers) as pool:
        eps = pool.map(_job, jobs, chunksize=4)
    for e in eps:
        e["_p"] = params[e["head"]]
    w0 = dict(START_W)
    by_head = {h: [e for e in eps if e["head"] == h] for h in HEADS}
    res: dict = {"heads": {}, "pooled": {}}
    for h in HEADS + ("pooled",):
        he = eps if h == "pooled" else by_head[h]
        rows = {r: [e for e in he if e["row"] == r] for r in ROWS}
        res_h = {
            "n_episodes": len(he), "n_falls": sum(1 for e in he if e.get("fell")),
            "shares": shares(he, w0),
            "shares_by_row": {r: shares(v, w0) for r, v in rows.items() if v},
            "crouch": crouch(he), "swing_rho": swing_rho(he), "tg_prefall": tg_prefall(he),
            "tg_prefall_by_row": {r: tg_prefall(v) for r, v in rows.items() if v},
            "walk_by_hold": {},
        }
        holds = sorted({k for e in he for k in e.get("walk_by_hold", {})})
        for hl in holds:
            xs = [e["walk_by_hold"][hl] for e in he if hl in e.get("walk_by_hold", {})]
            trk = sum(x["trk_lin"] for x in xs) * params[HEADS[0]]["track_lin_w"]
            res_h["walk_by_hold"][hl] = {
                "n": sum(x["n"] for x in xs),
                **{t: (abs(w0[t]) * sum(x[TERM_KERNEL[t]] for x in xs) / trk if trk else None) for t in TERM_KERNEL}}
        if h == "pooled":
            res["pooled"] = res_h
        else:
            res["heads"][h] = res_h
    # ---- re-dose (per term, one weight across the three heads)
    dose = {}
    for term in TERM_KERNEL:
        ratios = [res["heads"][h]["shares"][term]["ratio_per_unit_w"] for h in HEADS]
        d = redose(START_W[term], BANDS[term], ratios)
        d["start"] = START_W[term]
        d["band"] = list(BANDS[term])
        d["shares_at_start"] = {h: abs(START_W[term]) * r for h, r in zip(HEADS, ratios)}
        d["shares_at_chosen"] = ({h: abs(d["weight"]) * r for h, r in zip(HEADS, ratios)}
                                 if d["weight"] is not None else None)
        d["pooled_share_at_chosen"] = (abs(d["weight"]) * res["pooled"]["shares"][term]["ratio_per_unit_w"]
                                       if d["weight"] is not None else None)
        dose[term] = d
    res["dose"] = dose
    # ---- Episode_Reward at the chosen weights
    er = {}
    for term, kk in TERM_KERNEL.items():
        wch = dose[term]["weight"] if dose[term]["weight"] is not None else START_W[term]
        er[term] = {"weight": wch, "pooled": episode_reward({r: [e for e in eps if e["row"] == r] for r in ROWS}, wch, kk),
                    "per_head": {h: episode_reward({r: [e for e in by_head[h] if e["row"] == r] for r in ROWS}, wch, kk)
                                 for h in HEADS}}
    # calibration: the replayed tracking and the p2a (flat-tile-only) orientation in the same units
    cal = {"track_lin_vel_xy_exp": {}, "reward_orientation_p2a_flat_only": {}}
    for h in HEADS:
        rows = {r: [e for e in by_head[h] if e["row"] == r] for r in ROWS}
        cal["track_lin_vel_xy_exp"][h] = episode_reward(rows, params[h]["track_lin_w"], "trk_lin")["mixed"]
        flat_only = episode_reward({SUITE: rows[SUITE]}, params[h]["p2a_orient_w"], "tg")["per_row"][SUITE]
        cal["reward_orientation_p2a_flat_only"][h] = ROW_MIX[SUITE] * flat_only if flat_only is not None else None
    if tb:
        rds = {h: json.loads((ARMS / f"{h}.record.json").read_text())["run_dir"] for h in HEADS}
        cal["tb_tail_500"] = tb_tails(rds)
    res["episode_reward"] = er
    res["calibration"] = cal
    # ---- fidelity: the real twins on s18 recorded steps (2 field_hard episodes with a fall + 2 suite)
    fid_paths = []
    s18 = [e for e in by_head["P2__S1_s18"]]
    fid_paths += [e["path"] for e in s18 if e["row"] == "field_hard__omni" and e.get("fell")][:2]
    fid_paths += [e["path"] for e in s18 if e["row"] == "field_hard__omni" and not e.get("fell")][:1]
    hl_field = json.loads((Path(fid_paths[0]).parents[1] / "run_meta.json").read_text())["hold_labels"] if fid_paths else []
    try:
        res["fidelity"] = fidelity_check(fid_paths, params["P2__S1_s18"], hl_field)
    except Exception as exc:  # noqa: BLE001 -- recorded, never silently passed
        res["fidelity"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    # ---- verdicts
    res["verdicts"] = verdicts(res)
    res["meta"] = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"), "script": _rel(HERE),
        "code_sha_note": "kernels transcribe commit c5e1b104 (mdp/omni_clearance.py, rewards.py)",
        "env_yaml": {h: _rel(Path(s)) for h, s in envsrc.items()}, "inputs": used,
        "params": {h: {k: v for k, v in params[h].items()} for h in HEADS},
        "params_identical_across_seeds": same, "limit": limit,
    }
    return res


def verdicts(res: dict) -> dict:
    heads = res["heads"]
    pooled = res["pooled"]
    dose = res["dose"]
    v = {"CL": {}, "TG": {}}
    # (a)
    a_ok = True
    a_detail = {}
    for h, r in list(heads.items()) + [("pooled", pooled)]:
        d = {}
        for kind in ("pit", "ordinary"):
            rho, n = r["swing_rho"][kind]["rho"], r["swing_rho"][kind]["n"]
            ok = None if rho is None else rho > 0
            d[kind] = {"rho": rho, "n": n, "pass": ok}
            if h == "pooled" and not ok:
                a_ok = False
            if h != "pooled" and ok is False:
                a_ok = False
        a_detail[h] = d
    v["CL"]["a_income_rises_with_apex_incl_pit"] = {"pass": a_ok, "detail": a_detail}
    # (b)
    b_ok = all(r["crouch"][t]["pass"] for r in list(heads.values()) + [pooled]
               for t in ("reward_foot_clearance", "penalty_swing_min_clearance"))
    v["CL"]["b_no_crouch_payoff"] = {"pass": b_ok, "detail": {
        h: {t: r["crouch"][t]["frac"] for t in ("reward_foot_clearance", "penalty_swing_min_clearance")}
        for h, r in list(heads.items()) + [("pooled", pooled)]}}
    # (c)(d)(f) at the chosen weights
    for key, term, arm in (("c_foot_clearance_share_1_10pct", "reward_foot_clearance", "CL"),
                           ("d_swing_min_share_le_5pct", "penalty_swing_min_clearance", "CL"),
                           ("f_tilt_share_1_5pct", "reward_orientation", "TG")):
        d = dose[term]
        v[arm][key] = {"pass": bool(d["feasible"]), "weight": d["weight"], "start": d["start"], "moved": d["moved"],
                       "shares_at_start": d["shares_at_start"], "shares_at_chosen": d["shares_at_chosen"],
                       "band": d["band"], "feasible_abs_interval": d["interval_abs"]}
    # (e)
    e_ok = all(r["tg_prefall"]["pass"] for r in list(heads.values()) + [pooled])
    v["TG"]["e_prefall_ge_5x_walking"] = {"pass": e_ok, "detail": {
        h: {"ratio": r["tg_prefall"]["ratio"], "n_falls": r["tg_prefall"]["n_falls"]}
        for h, r in list(heads.items()) + [("pooled", pooled)]}}
    fid_ok = bool(res.get("fidelity", {}).get("ok"))
    for arm in ("CL", "TG"):
        checks = v[arm]
        v[arm] = {"verdict": "PASS" if all(c["pass"] for c in checks.values()) and fid_ok else "FAIL",
                  "checks": checks, "fidelity_ok": fid_ok}
    return v


# ================================================================================================ outputs
def gate_json(res: dict) -> dict:
    """The ``replay_gate.json`` "p2b" section (p2_live reads ``p2b.<arm>.terms.<term>.episode_reward`` and
    ``p2b.<arm>.verdict``)."""
    out = {}
    for arm, terms in ARM_TERMS.items():
        v = res["verdicts"][arm]
        ent = {"verdict": v["verdict"], "checks": v["checks"], "fidelity_ok": v["fidelity_ok"], "terms": {}}
        for t in terms:
            er = res["episode_reward"][t]
            ent["terms"][t] = {
                "weight": er["weight"], "knob": KNOB_OF[t],
                "episode_reward": er["pooled"]["mixed"],
                "episode_reward_per_row": er["pooled"]["per_row"],
                "episode_reward_per_head": {h: x["mixed"] for h, x in er["per_head"].items()},
                "start_weight": START_W[t],
                "share_pooled_at_weight": res["dose"][t]["pooled_share_at_chosen"],
                "shares_at_weight": res["dose"][t]["shares_at_chosen"],
            }
        ent["knobs"] = ({"KRABBY_FOOT_CLEAR_W": ent["terms"]["reward_foot_clearance"]["weight"],
                         "KRABBY_FOOT_CLEAR_MIN": 0.03, "KRABBY_FOOT_CLEAR_FLAT": 1,
                         "KRABBY_SWING_MIN_CLEAR_W": ent["terms"]["penalty_swing_min_clearance"]["weight"],
                         "KRABBY_FOOT_CLEAR_TERRAIN": 1} if arm == "CL" else
                        {"KRABBY_ORIENT_ALL_TILES": 1, "KRABBY_ORIENT_W": ent["terms"]["reward_orientation"]["weight"]})
        ent["rule"] = ("CHARTER 10 / P2B_PLAN step 2: (a) income rises with terrain-relative apex incl. pit swings; "
                       "(b) crouch change <= 10 % of the +1 cm lift payoff; (c) income 1-10 % of tracking; (d) swing-min "
                       "<= 5 %" if arm == "CL" else
                       "CHARTER 10 / P2B_PLAN step 2: (e) pre-fall cost >= 5x walking; (f) cost 1-5 % of tracking")
        ent["episode_reward_units"] = ("TensorBoard Episode_Reward: per-episode weighted sum / 40 s; eval episodes at "
                                       "their own rate scaled to a 40-s episode (a fall shortens it); rows mixed 0.5 "
                                       "suite + 0.31 field_hard + 0.19 field_light")
        ent["generated"] = res["meta"]["generated"]
        ent["report"] = _rel(REPORT_MD)
        out[arm] = ent
    out["tracking_income"] = "1.25 * track_lin_vel_xy_exp(std 0.2), the S1 runs' dumped weight/std (named before reading)"
    out["meta"] = {k: res["meta"][k] for k in ("generated", "script", "code_sha_note", "env_yaml", "inputs",
                                               "params_identical_across_seeds")}
    out["calibration"] = res["calibration"]
    out["fidelity"] = res.get("fidelity")
    return out


def _pct(x, d=2):
    return "--" if x is None else f"{100 * x:.{d}f} %"


def _f(x, fmt="{:.4g}"):
    return "--" if x is None else fmt.format(x)


def render_md(res: dict) -> str:
    H = HEADS
    heads, pooled, dose, v = res["heads"], res["pooled"], res["dose"], res["verdicts"]
    p = res["meta"]["params"]["P2__S1_s18"]
    L = []
    L.append("# Replay gate p2b: terrain-relative lift (CL) and tilt guard (TG)\n")
    L.append(f"CHARTER §10 / P2B_PLAN.md step 2 offline replay gate. CPU only: saved eval raws, no Isaac, training or GPU. "
             f"Generated by `scripts/p2b_replay.py` on {res['meta']['generated']}. The numbers are also in "
             "`replay_gate.json` under `p2b` (the E1/F1 entries are untouched).\n")
    L.append("## Verdicts\n")
    L.append("| arm | verdict | weights (start → chosen) | Episode_Reward (TB units, at chosen) | failed checks |")
    L.append("|---|---|---|---|---|")
    for arm, terms in ARM_TERMS.items():
        ws = "; ".join(f"`{KNOB_OF[t]}` {START_W[t]:g} → {_f(dose[t]['weight'], '{:g}')}" for t in terms)
        ers = "; ".join(f"{t} {_f(res['episode_reward'][t]['pooled']['mixed'], '{:+.4g}')}" for t in terms)
        failed = [k for k, c in v[arm]["checks"].items() if not c["pass"]] + ([] if v[arm]["fidelity_ok"] else ["fidelity"])
        L.append(f"| **{arm}** | **{v[arm]['verdict']}** | {ws} | {ers} | {', '.join(failed) or 'none'} |")
    L.append("")
    L.append("Per CHARTER §10 a failed check stops and asks the user. No rule was tuned to pass.\n")
    # ---------------- method
    L.append("## Method\n")
    L.append("**Inputs.** The three S1 p2a heads (`P2__S1_s{16,17,18}` model_19994): field_hard and field_light at "
             "env_seeds 1 and 2 (100 × 80-s episodes each) and the flat suite from each arm record's `driver_stamp` "
             "(1000 × 15.5-s episodes).\n")
    L.append("| head | row | seed | stamp | n | terrain |")
    L.append("|---|---|---|---|---|---|")
    for u in res["meta"]["inputs"]:
        L.append(f"| {u['head']} | {u['row']} | {u['seed']} | `{u['stamp']}` | {u['n']} | {u['terrain_source']} |")
    L.append("")
    L.append("**Step alignment.** Reward step s pairs the command `cmd_applied[s]` with the post-physics state at raw "
             "index s+1, as `replay_motion_terms.step_series` does. The raws record the state before each `env.step`.\n")
    L.append("**Kernels** (from `mdp/omni_clearance.py` and `rewards.py` at c5e1b104):")
    L.append("- `reward_foot_clearance` (`reward_foot_clearance_terrain`): w · mean over the six feet of "
             "in_swing · clamp(relu(h − 0.03), max 0.26) · gate.")
    L.append("- `penalty_swing_min_clearance` (`penalty_swing_min_clearance_terrain`): w · mean over feet of "
             "in_swing · [h < 0.04] · gate.")
    L.append("- Here h = foot_z − terrain_z under the foot, in_swing = ‖contact force‖ ≤ 0.1 N, and gate = "
             f"max(‖cmd_xy‖, {p['turn_r']}·|cmd_wz|) > 0.12 (the `_cmd_gate` omni branch). No parkour_flat mask "
             "(FOOT_CLEAR_FLAT 1).")
    L.append("- `reward_orientation` → `reward_orientation_upright`: w · (g_x² + g_y²) of projected_gravity_b, every tile.")
    L.append("- **Tracking income** (named before any result was read): 1.25 · track_lin_vel_xy_exp with std 0.2, "
             "the S1 runs' dumped weight and std. Adding 1.0 · track_ang_vel_z_exp (std 0.5) is reported as info only.\n")
    L.append("**Windows.**")
    L.append("- **Walking:** reward steps whose command hold is not `stand`/`settle`, with the command gate on (turn "
             "holds count, because a turn command gates the CL terms on under omni), and outside the 2 s before a fall.")
    L.append("- **Pre-fall:** for a `crab_failure` episode, the 100 reward steps (2 s) before the failing step.")
    L.append("- **Shares:** |w| · Σ kernel / Σ tracking over a head's walking steps, a ratio of sums over both field "
             "rows, both env_seeds and the suite.\n")
    L.append("**Fidelity.** The real landed twins (`omni_clearance.reward_foot_clearance_terrain` / "
             "`penalty_swing_min_clearance_terrain`, with the library `_cmd_gate` exec'd from source) and the exec'd "
             "`reward_orientation_upright` were run on recorded s18 field_hard steps. Only the terrain sample was "
             "substituted (the raws' terrain_z). Result: "
             f"{_fidelity_line(res.get('fidelity'))}.\n")
    # ---------------- params
    L.append("## Parameters and their sources\n")
    L.append("No composed p2b dry-run dump exists, because building one needs Isaac. Each value below is the S1 p2a "
             "dump with the CL/TG overlay applied by the forward cfg's `__post_init__` rules, read from source.\n")
    L.append("| param | value and source |")
    L.append("|---|---|")
    for k, s in p["sources"].items():
        L.append(f"| {k} | {s} |")
    L.append("")
    chk = p["checks"]
    L.append("Source checks: " + "; ".join(f"{k}: {'ok' if ok else 'MISMATCH'}" for k, ok in chk.items()) + ".")
    same = res["meta"]["params_identical_across_seeds"]
    diff = [k for k, ok in same.items() if not ok]
    L.append(f"Every param is identical across the s16/s17/s18 dumps: {'yes' if not diff else 'NO: ' + ', '.join(diff)}.\n")
    L.append("**The clamp is 0.26, not 0.20.** The task text said `clamp(..., max=0.20)`, which is the library "
             "signature default. The forward cfg passes `MAX_CLEARANCE_M = 0.26` and all three dumps carry 0.26. The "
             "replay uses 0.26. Only swings above 0.29 m would differ.\n")
    # ---------------- CL
    L.append("## CL: terrain-relative lift\n")
    L.append("### (a) Income rises with terrain-relative swing apex, pit swings included\n")
    L.append("Swings are `p2_mech.extract_swings` (a no-contact run at ≤ 5 N bounded by contact). Apex is max(foot_z − "
             "terrain_z). A swing is a pit swing when it lifts off more than 0.03 m below the other feet's terrain. "
             "Each swing's income is the foot's own share of the term over the swing span, 1/6 per foot, with dt "
             "and the gate. Only walking liftoffs are counted. The rule is Spearman ρ > 0 in both groups.\n")
    L.append("| head | pit n | pit ρ | pit apex median m | pit paying | ordinary n | ordinary ρ | ordinary apex median m | ordinary paying |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for h, r in list(heads.items()) + [("pooled", pooled)]:
        sp, so = r["swing_rho"]["pit"], r["swing_rho"]["ordinary"]
        L.append(f"| {h} | {sp['n']} | {_f(sp['rho'], '{:+.3f}')} | {_f(sp['apex_median_m'], '{:.3f}')} | "
                 f"{_pct(sp['frac_paying'], 0)} | {so['n']} | {_f(so['rho'], '{:+.3f}')} | "
                 f"{_f(so['apex_median_m'], '{:.3f}')} | {_pct(so['frac_paying'], 0)} |")
    L.append(f"\nResult: **{'PASS' if v['CL']['checks']['a_income_rises_with_apex_incl_pit']['pass'] else 'FAIL'}**.\n")
    L.append("### (b) No crouch payoff (counterfactual)\n")
    L.append("The root is lowered 1 cm at fixed foot positions and every walking step is re-scored. That change is "
             "compared with the payoff of lifting every swing foot 1 cm (terrain-relative). The rule is |Δcrouch| ≤ "
             "10 % of |Δlift|. The library's root-relative form (the p2a/2b form) is shown beside it for contrast.\n")
    L.append("| head | term | Δlift (k·s) | Δcrouch (k·s) | crouch/lift | root-form crouch/lift |")
    L.append("|---|---|---|---|---|---|")
    for h, r in list(heads.items()) + [("pooled", pooled)]:
        for t in ("reward_foot_clearance", "penalty_swing_min_clearance"):
            c = r["crouch"][t]
            L.append(f"| {h} | {t} | {_f(c['lift_payoff_k_s'])} | {_f(c['crouch_change_k_s'])} | {_pct(c['frac'])} | "
                     f"{_pct(c['root_form_frac'], 0)} |")
    L.append(f"\nResult: **{'PASS' if v['CL']['checks']['b_no_crouch_payoff']['pass'] else 'FAIL'}**. The "
             "terrain-relative kernels never read the root, so the crouch change is exactly 0. The root-relative "
             "form pays a 1-cm crouch as much as a 1-cm lift, which is the finding behind the switch.\n")
    L.append("### (c) and (d) Shares of tracking income, non-fall walking\n")
    L.append(_share_table(res, ("reward_foot_clearance", "penalty_swing_min_clearance")))
    L.append("")
    # ---------------- TG
    L.append("## TG: tilt guard on all tiles\n")
    L.append("### (e) Pre-fall cost vs walking\n")
    L.append("| head | falls | pre-fall tilt² /step | walking tilt² /step | ratio | ≥ 5× |")
    L.append("|---|---|---|---|---|---|")
    for h, r in list(heads.items()) + [("pooled", pooled)]:
        t = r["tg_prefall"]
        L.append(f"| {h} | {t['n_falls']} | {_f(t['prefall_mean_tilt2_per_step'] and t['prefall_mean_tilt2_per_step'] * 0.02, '{:.4g}')} | "
                 f"{_f(t['walk_mean_tilt2_per_step'] and t['walk_mean_tilt2_per_step'] * 0.02, '{:.4g}')} | "
                 f"{_f(t['ratio'], '{:.1f}×')} | {'yes' if t['pass'] else 'NO'} |")
    L.append("\nBy row (pooled over heads): " + "; ".join(
        f"{r} {_f(x['ratio'], '{:.1f}×')} ({x['n_falls']} falls)" for r, x in pooled["tg_prefall_by_row"].items()) + ".")
    L.append(f"\nResult: **{'PASS' if v['TG']['checks']['e_prefall_ge_5x_walking']['pass'] else 'FAIL'}**.\n")
    L.append("### (f) Share of tracking income, non-fall walking\n")
    L.append(_share_table(res, ("reward_orientation",)))
    L.append("")
    # ---------------- re-dose
    L.append("## Re-dose\n")
    L.append("The start weights are FOOT_CLEAR_W 1.0, SWING_MIN_CLEAR_W −0.4 and ORIENT_W −0.7. A term whose share "
             "is outside its band in any head moves only its own weight. The new weight is one value across the "
             "three heads: the one closest to the start that lands every head in band, at 2 s.f. rounded inward. If "
             "no such weight exists, the check FAILs and goes to the user.\n")
    L.append("| term | knob | band | feasible |w| interval | start | chosen | moved | shares at chosen (s16 / s17 / s18) | pooled |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t, d in dose.items():
        iv = d["interval_abs"]
        sh = " / ".join(_pct(d["shares_at_chosen"][h]) for h in H) if d["shares_at_chosen"] else "--"
        L.append(f"| {t} | `{KNOB_OF[t]}` | {_pct(d['band'][0], 0)}–{_pct(d['band'][1], 0)} | "
                 f"[{_f(iv[0], '{:.3g}')}, {_f(iv[1], '{:.3g}')}] | {d['start']:g} | **{_f(d['weight'], '{:g}')}** | "
                 f"{d['moved']} | {sh} | {_pct(d['pooled_share_at_chosen'])} |")
    L.append("")
    # ---------------- episode reward
    L.append("## Episode_Reward magnitudes (for the +200 liveness read)\n")
    L.append("These are signed TensorBoard units (`Episode_Reward/<term>` = the episodic weighted sum / 40 s), at "
             "the chosen weights. Each eval episode is scored at its own per-second rate and scaled to a 40-s "
             "training episode:")
    L.append("- An 80-s field episode that completes its schedule counts as 40 s at its rate.")
    L.append("- A 15.5-s suite episode stands for a full 40-s episode at its rate.")
    L.append("- A fall shortens the episode to min(its duration, 40 s).\n")
    L.append("Rows are mixed like S1 training: 0.5 flat (the suite) plus 0.5 field, the field split 0.62 hard / 0.38 "
             "light. p2_live accepts a live value within 3× of the stored magnitude.\n")
    L.append("| term | weight | suite | field_hard | field_light | **mixed (stored)** | s16 | s17 | s18 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t, er in res["episode_reward"].items():
        pr = er["pooled"]["per_row"]
        L.append(f"| {t} | {er['weight']:g} | {_f(pr.get(SUITE), '{:+.4g}')} | {_f(pr.get('field_hard__omni'), '{:+.4g}')} | "
                 f"{_f(pr.get('field_light__omni'), '{:+.4g}')} | **{_f(er['pooled']['mixed'], '{:+.4g}')}** | "
                 + " | ".join(_f(er["per_head"][h]["mixed"], "{:+.4g}") for h in H) + " |")
    cal = res["calibration"]
    L.append("\n**Calibration of the units against the S1 runs' own training logs** (mean of the last 500 its):\n")
    L.append("| head | replay track_lin (mixed) | TB track_lin | ratio | replay p2a orientation (flat only, −0.7) | TB reward_orientation | ratio |")
    L.append("|---|---|---|---|---|---|---|")
    tbt = cal.get("tb_tail_500") or {}
    for h in H:
        a, b = cal["track_lin_vel_xy_exp"].get(h), (tbt.get(h) or {}).get("Episode_Reward/track_lin_vel_xy_exp")
        c, d = cal["reward_orientation_p2a_flat_only"].get(h), (tbt.get(h) or {}).get("Episode_Reward/reward_orientation")
        L.append(f"| {h} | {_f(a, '{:.4g}')} | {_f(b, '{:.4g}')} | {_f(a / b if a and b else None, '{:.2f}')} | "
                 f"{_f(c, '{:+.4g}')} | {_f(d, '{:+.4g}')} | {_f(c / d if c and d else None, '{:.2f}')} |")
    L.append("")
    # ---------------- per hold
    L.append("## Shares by hold (pooled heads, start weights; descriptive)\n")
    L.append("| hold | steps | foot-clearance | swing-min | tilt |")
    L.append("|---|---|---|---|---|")
    for hl, x in pooled["walk_by_hold"].items():
        L.append(f"| {hl} | {x['n']} | {_pct(x['reward_foot_clearance'])} | {_pct(x['penalty_swing_min_clearance'])} | "
                 f"{_pct(x['reward_orientation'])} |")
    L.append("")
    # ---------------- caveats
    L.append("## Caveats\n")
    L.append("- **Terrain sample.** The raws' terrain_z is a downward raycast of the terrain mesh. The live twin "
             "samples the height field at the nearest pixel (0.08 m) of the tile the foot is on. They agree on block "
             "tops and pit floors and can differ within about one pixel of a vertical wall or pit edge. The fidelity "
             "check substitutes only this sample.")
    L.append("- **Contact.** The reward's swing test is ‖force‖ ≤ 0.1 N on the recorded `foot_force_norm` (the same "
             "sensor read). p2_mech's swing span uses 5 N; it is used only to delimit swings for check (a).")
    L.append("- **Distribution.** Eval schedules (8 holds per field episode; one category per suite episode) are not "
             "the training command distribution. The shares are eval-walking shares, and the Episode_Reward mix is "
             "an estimate. The calibration table bounds the error.")
    tbt = res["calibration"].get("tb_tail_500") or {}
    ori = [(h, res["calibration"]["reward_orientation_p2a_flat_only"].get(h),
            (tbt.get(h) or {}).get("Episode_Reward/reward_orientation")) for h in H]
    ori = [(h, a / b) for h, a, b in ori if a and b]
    if ori:
        hmin, rmin = min(ori, key=lambda x: x[1])
        er_tg = res["episode_reward"]["reward_orientation"]["per_head"][hmin]["mixed"]
        er_pool = res["episode_reward"]["reward_orientation"]["pooled"]["mixed"]
        L.append(f"- **TG liveness margin.** The flat-tile tilt cost replays at {min(r for _, r in ori):.2f}–"
                 f"{max(r for _, r in ori):.2f}× the S1 runs' own training tail. Training has push and mass "
                 "randomisation and stochastic actions; the evals have neither. If the live TG term runs high by the "
                 f"same factor, {hmin} would log about {er_tg / rmin:+.4f} against the stored {er_pool:+.4f}, which is "
                 f"{abs(er_tg / rmin / er_pool):.1f}× (p2_live's limit is 3×). A false liveness FAIL at +200 is "
                 "possible there and should be read against this table. The same factor puts TG's training share "
                 f"near {100 * res['dose']['reward_orientation']['pooled_share_at_chosen'] / rmin:.1f} % at the "
                 "chosen weight, inside the 5 % ceiling.")
    mt = {hl: x["reward_orientation"] for hl, x in pooled["walk_by_hold"].items() if hl.startswith("move_turn")}
    wtg = dose["reward_orientation"]["weight"]
    if mt and wtg is not None:
        L.append("- **Tilt concentrates in move_turn.** At the chosen weight the move_turn holds' TG share is "
                 + ", ".join(f"{hl} {100 * s * abs(wtg) / abs(START_W['reward_orientation']):.1f} %" for hl, s in mt.items())
                 + " of their tracking, against about 0.2–1.1 % on the other holds. This is descriptive: the bands "
                 "are read on all walking steps.")
    L.append("- **Heads.** These are the p2a heads, which never trained with the p2b terms. A policy trained under CL "
             "will move these shares, which is what the +200 liveness read and the s18 stop exist for.")
    L.append("- **field_hard env_seeds.** Only env_seeds 1 and 2 exist for the p2a heads. Seeds 3 and 4 arrive with "
             "the step-3 REF evals.\n")
    return "\n".join(L) + "\n"


def _fidelity_line(f: dict | None) -> str:
    if not f:
        return "not run"
    if f.get("error"):
        return f"ERROR {f['error']}"
    d = f["max_abs_diff"]
    return (f"{'match' if f['ok'] else 'MISMATCH'} on {f['n_steps']} steps / {f['n_episodes']} episodes "
            f"(max |Δ| clearance {d['cl']:.1e}, swing-min {d['sm']:.1e}, tilt {d['tg']:.1e})")


def _share_table(res: dict, terms) -> str:
    H = HEADS
    L = ["| term | band | weight | s16 | s17 | s18 | pooled | in band at start | chosen weight | s16 / s17 / s18 at chosen |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t in terms:
        d = res["dose"][t]
        sh = [res["heads"][h]["shares"][t]["share"] for h in H]
        lo, hi = d["band"]
        inb = all(lo <= s <= hi for s in sh)
        L.append(f"| {t} | {_pct(lo, 0)}–{_pct(hi, 0)} | {START_W[t]:g} | " + " | ".join(_pct(s) for s in sh)
                 + f" | {_pct(res['pooled']['shares'][t]['share'])} | {'yes' if inb else 'no'} | **{_f(d['weight'], '{:g}')}** | "
                 + (" / ".join(_pct(d["shares_at_chosen"][h]) for h in H) if d["shares_at_chosen"] else "--") + " |")
    L.append("")
    L.append("By row at the start weight (pooled heads), with the lin+ang tracking share in brackets:")
    L.append("")
    L.append("| term | " + " | ".join(ROWS) + " |")
    L.append("|---|" + "---|" * len(ROWS))
    for t in terms:
        cells = []
        for r in ROWS:
            x = res["pooled"]["shares_by_row"].get(r, {}).get(t, {})
            cells.append(f"{_pct(x.get('share'))} ({_pct(x.get('share_lin_ang'))})")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Per head (all rows) with lin+ang tracking: " + "; ".join(
        f"{h} " + ", ".join(f"{t} {_pct(res['heads'][h]['shares'][t]['share_lin_ang'])}" for t in terms) for h in H) + ".")
    return "\n".join(L)


def write_outputs(res: dict) -> None:
    gate = json.loads(REPLAY_JSON.read_text()) if REPLAY_JSON.is_file() else {}
    gate["p2b"] = json.loads(json.dumps(gate_json(res), default=_jsonable))
    REPLAY_JSON.write_text(json.dumps(gate, indent=1, default=_jsonable) + "\n")
    REPORT_MD.write_text(render_md(res))


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="episodes per stamp (smoke runs; never with --write)")
    ap.add_argument("--no-tb", action="store_true", help="skip the TensorBoard calibration read")
    ap.add_argument("--write", action="store_true", help="update replay_gate.json (p2b) and write REPLAY_GATE_P2B.md")
    ap.add_argument("--json", default=None, help="also dump the full result (minus per-episode arrays) here")
    args = ap.parse_args(argv)
    if args.write and args.limit is not None:
        ap.error("--write refuses a --limit run")
    res = run(workers=args.workers, limit=args.limit, tb=not args.no_tb)
    if args.json:
        Path(args.json).write_text(json.dumps({k: v for k, v in res.items()}, indent=1, default=_jsonable))
    if args.write:
        write_outputs(res)
    for arm in ("CL", "TG"):
        v = res["verdicts"][arm]
        print(f"{arm}: {v['verdict']}  " + ", ".join(f"{k}={'PASS' if c['pass'] else 'FAIL'}" for k, c in v["checks"].items()))
    for t, d in res["dose"].items():
        print(f"  {t}: start {d['start']:g} -> {d['weight']}  shares@start "
              + " ".join(f"{100 * s:.2f}%" for s in d["shares_at_start"].values())
              + f"  episode_reward {res['episode_reward'][t]['pooled']['mixed']:+.5g}")
    print("fidelity:", _fidelity_line(res.get("fidelity")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

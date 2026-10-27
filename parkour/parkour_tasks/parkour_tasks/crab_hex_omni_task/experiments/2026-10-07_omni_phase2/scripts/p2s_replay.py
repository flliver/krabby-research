#!/usr/bin/env python3
"""Omni phase 2, stage p2s -- the OFFLINE REPLAY GATE (CHARTER 11 / P2S_PLAN.md step 4). CPU ONLY, read-only on raws.

Re-scores SAVED eval raws with the two p2s stand-slot terms EXACTLY as the landed code computes them (commit
e1a08baf: ``reward_stand_support`` / ``penalty_stand_slide`` in ``mdp/crab_hex_omni_motion_rewards.py``, imported and
called on a stub env built from the npz), solves the two doses from the M rule, and judges the pre-registered gates.
It launches nothing (no Isaac, no GPU, no training, no eval).

INPUTS (every stamp is a saved eval; the newest complete stamp per name, the ``p2s_stand`` / ``p2_read`` convention)
  suite stands (deterministic): REF es1 / es2 (``lr/REF_S1_s18/suite__omni/seed00{1,2}``), C0 / CL / TG / A1 / F1
      (``p2s_stand._suite_stamp_for``: the driver's own suite stamp of X12 / X13 / X14 / X06 / X09), the FRESH
      zero-action statue (``lr/DIAG_STATUE_s18/suite__omni/seed001``; the 2026-09-23 E6 statue is reported beside it);
  suite stands (sampled, ``--sample-actions`` at the training sigma): REF (``lr/DIAG_REFSAMP_s18``) and the noisy
      statue (``lr/DIAG_NOISYSTATUE_s18``, model_19994 with the actor output layer zeroed);
  wall stands: the ``field_turn__omni`` stand (hold 1, after the walk-out; env_seed 1) and the ``field_hard__omni``
      opening stand (hold 0; env_seeds 1 and 2) of REF / C0 / TG;
  moving holds (G4): every non-stand row of the suite's other nine waves (REF es1, TG, sampled REF), plus the stand
      wave's own settle hold, plus every non-stand hold of the REF field_turn / field_hard rows.

STEP ALIGNMENT (``replay_motion_terms.step_series``): reward step s pairs the command ``cmd_applied[s]`` with the POST
  state at raw index s+1 (the raws record the state BEFORE each ``env.step``).

THE GATE g (emulating ``stand_mask()``; the raws carry no slot id -- ``p2_replay``'s convention, refined)
  stand_mask[s] = the episode's "stand" hold (``hold_labels.index("stand")``; on the suite, the stand-category wave only)
                  AND cmd_applied[s, :2] == 0 (vx, vy)
  g[s]          = stand_mask[s] AND -projected_gravity_b z (post state) > 0.9  (inside the landed kernels).
  ``p2_replay`` also required wz == 0. On a stand hold the command term's facing P-control writes a nonzero wz once
  the facing error leaves its dead band (0 rows on most deterministic stands, 7.5 % of the noisy statue's rows); those
  rows ARE stand-slot rows in training (``stand_mask()`` is the slot, not "slot and zero turn" -- P2S_PLAN "The
  lever"), so the primary gate keeps them. The literal cmd (vx, vy, wz) == 0 gate is re-run as a SENSITIVITY (B,
  doses, sampled margins, anti-termination); the report shows both.
  WINDOW: the stand hold's STEADY rows (``p2s_stand.hold_masks``: fall-trimmed, the eval's ``steady_mask``, the first
  1.0 s of the hold dropped), i.e. reward steps s whose post row s+1 is steady -- the same rows ``p2s_stand``'s own
  numpy T1/T2 read, so the fidelity check compares like with like. Every per-second number below is the step-weighted
  mean over these gate rows (one step = dt).

TERMS (landed, unweighted, per step; ``s_f = clamp(|F_f| / 50 N, 0, 1)``, F = the recorded ``foot_force_norm``)
  T1 reward_stand_support = g * (sum_f s_f - 3)          T2 penalty_stand_slide = g * sum_f s_f * |v_f| (3-D, m/s)

EXISTING TERMS (B, the anti-termination totals): every nonzero-weight term of the S1 s18 run's own
  ``params/env.yaml`` (``parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB``), at its
  dumped weight, per step at the step alignment above:
  * via ``replay_motion_terms.step_series`` (the shared kernels, ``train_params_from_dumps`` of that dump):
    track_lin_vel_xy_exp, track_ang_vel_z_exp, reward_tracking_yaw (facing), reward_yaw_progress, reward_action_rate
    (runner clip, one-step delay), reward_feet_air_time_positive (float32 sensor clocks), penalty_foot_idle_when_forward;
  * transcribed here from ``rewards.py`` / ``crab_hex_omni_rewards.py`` (source lines in the report):
    penalty_tracking_error_l1 (|cmd_xy - v_xy|; at cmd 0 the travel-frame rotation leaves the norm unchanged),
    penalty_lin_vel_y (cross-command, gated ||cmd_xy|| > 0.12), reward_forward_progress_along_command (gated),
    reward_orientation (g_x^2 + g_y^2, the parkour_flat-tile form -- the suite is flat), reward_lin_vel_z (v_z^2, flat
    form), reward_delta_torques (sum (tau_s - tau_{s-1})^2, 24 joints), penalty_tibia_deviation_in_stance
    (sum_f (q_tibia_f - q0_f)^2 [F_f > 0.1 N], FL..RR in both lists), penalty_excess_feet_contact_forward (gated),
    reward_collision (Hip / Femur links from ``leg_contact_force`` > 0.1 N; the trunk's own contact is not recorded),
    reward_clock_schedule (0.001; the omni-formation tool's ``_clock_income`` with the recorded phase -- at a stopped
    clock it is phase-free stance quality) and reward_clock_swing_apex (0.001; 0 at a stopped clock);
  * NOT replayed (stated): reward_stride_length (gated off at ||cmd|| <= 0.12), reward_feet_edge (no edge on the flat
    suite), reward_feet_stumble (needs the horizontal force; the raws keep only its norm).

DOSES (P2S_PLAN "The lever" + "Left for implementation"; B is computed and frozen BEFORE the solve)
  B   = existing-term income/s, statue minus REF (deterministic, REF es1 + es2 pooled), suite stand gate rows.
  dT1 = T1_statue - T1_REF, dT2 = T2_REF - T2_statue (same rows; the statue's MEASURED T1, not six-down +3).
  M in {1.0 (ST1), 2.5 (ST2)}/s:  w_slide = -0.5 M;  w_support = (M - B - |w_slide| dT2) / dT1;  2 s.f.
  One weight pair per dose; the realised margin B + w_support dT1 + |w_slide| dT2 is reported per REF seed.

GATES (a failed check => verdict FAIL with its reason; the orchestrator stops and asks)
  G1  AUC(statue vs each deterministic head, per-episode T1/s) >= 0.95 (statue higher) and the same for T2/s (statue
      lower). Heads: REF es1, REF es2, C0, CL, TG, A1, F1.
  G2  within-head Spearman rho(T1/s, ``p2s_stand`` off-ground share) <= -0.5 on TG and CL (suite stands, steady window);
      every head reported.
  G3  at each dose, on the statue's own gate rows through the landed kernels: a still tripod hover (3 feet unloaded,
      their weight moved to the other 3, still) costs >= 0.5 M more than the statue (the worse of the two tripods); a
      six-down shuffle (the statue's weight spread over six feet, every foot +0.1 m/s along world x) costs >= 0.25 M
      more than the same still six-down stance; a 1-cm root crouch at fixed feet changes nothing (|delta| <= 1e-12).
  G4  exactly 0 (and finite) on every non-stand row, through the landed kernels with the emulated mask; plus an AST
      check that both functions multiply by ``_stand_gate`` and that ``_stand_gate`` reads ``stand_mask()``; plus an
      emulation check (mask off -> 0; mask on but tipped -> 0).
  G5  finite on every wall stand row; the share of feet over pits (terrain_z more than 5 cm below the pad level the
      loaded feet stand on, :func:`pit_mask`) is reported, with the flat suite stand as a zero control.
  ANTI-TERMINATION  the marching head's net stand-slot reward (existing terms + w_support T1 + w_slide T2) >= +0.5/s
      at both doses on the deterministic REF stands (es1, es2) and on the SAMPLED REF stands; reported on the
      deterministic field_turn wall stands.
  FIDELITY  the landed kernels (float64 stub) equal ``p2s_stand``'s numpy T1/T2 per episode to 1e-6 on every suite
      stand episode read here (and per step against this module's numpy mirror).

REPORTED, NOT GATED
  The sampled pair's per-term margins (noisy statue minus sampled REF) at each dose, and f = sampled / deterministic
  (T1 + T2) margin -- step 2 found T2 REVERSES under sampling.
  LIVENESS (for p2_live; TensorBoard units: Episode_Reward/<term> = weight x mean over all env-steps): with the stand
  slot share pinned at 0.15 (the S1 dump's ``commands.base_velocity.cmd_pool``), ER = w x 0.15 x the SAMPLED REF
  per-stand-row mean; the implied mean sum s_f = 3 + ER_support / (0.15 w_support). The episode-length factor
  (S1 s18 mean_episode_length / 2000, from ``replay_gate.json`` p2b calibration) is shown beside it.

CLI:  p2s_replay.py [--write] [--workers 8] [--limit N]
      --write merges a "p2s" key into replay_gate.json (E1 / F1 / p2b untouched) and writes REPLAY_GATE_P2S.md.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import importlib.util
import json
import math
import os
import re
import sys
import types
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
CAMPAIGN = HERE.parents[1]
OMNI = HERE.parents[3]                                      # .../crab_hex_omni_task
TASKS = OMNI.parent                                         # .../parkour_tasks/parkour_tasks
PARKOUR = TASKS.parents[1]                                  # .../parkour
REPO = PARKOUR.parent
OUT = Path(os.environ.get("P2_OUT") or PARKOUR / "outputs/omni_plan/phase2")
LROUT = OUT / "lr"
STAND_PY = HERE.parent / "p2s_stand.py"
RT_PY = OMNI / "experiments/tools/replay_motion_terms.py"
OF_PY = OMNI / "experiments/2026-09-16_1805_omni_formation/replay_reward_economy.py"
MOTION_REWARDS_PY = OMNI / "mdp/crab_hex_omni_motion_rewards.py"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
LIB_REWARDS_PY = PARKOUR / "parkour_isaaclab/envs/mdp/rewards.py"
OMNI_REWARDS_PY = OMNI / "mdp/crab_hex_omni_rewards.py"
S1_RUN = PARKOUR / "logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
REPLAY_JSON = CAMPAIGN / "replay_gate.json"
REPORT_MD = CAMPAIGN / "REPLAY_GATE_P2S.md"

SUITE, FH, FT = "suite__omni", "field_hard__omni", "field_turn__omni"
FEET = ("FL", "FR", "ML", "MR", "RL", "RR")
TRIPODS = {"A (FL, MR, RL)": (0, 3, 4), "B (FR, ML, RR)": (1, 2, 5)}

DOSES = {"ST1": 1.0, "ST2": 2.5}            #: target still-minus-march margin M per dose (/s)
SLIDE_PER_M = -0.5                          #: w_slide = -0.5 M per (m/s)
G1_AUC_MIN = 0.95
G2_RHO_MAX = -0.5
G2_DECISIVE = ("TG", "CL")
G3_HOVER_FRAC = 0.5                         #: hover must cost >= 0.5 M more than the statue
G3_SHUFFLE_FRAC = 0.25                      #: shuffle must cost >= 0.25 M more than the still six-down stance
SHUFFLE_V = 0.1                             #: m/s per foot
CROUCH_M = 0.01
CROUCH_TOL = 1e-12
ANTI_TERM_MIN = 0.5                         #: net stand-slot reward floor (/s)
FIDELITY_TOL = 1e-6
PIT_TOL_M = 0.05                            #: terrain this far below the pad level = a pit under the foot
TRAIN_EP_STEPS = 2000                       #: 40 s / 0.02
F_STEP2_LINE = 0.33                         #: step 2's f stop line (reported here, not a step-4 gate)
STAND_SHARE_PLAN = 0.15                     #: P2S_PLAN "Left for implementation": pin the stand-slot share


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ST = _load(STAND_PY, "p2s_stand__p2s_replay")              # the ONE stand-estimator set (read-only reuse)
RT = _load(RT_PY, "replay_motion_terms__p2s")              # step_series / train_params_from_dumps / spearman
OF = _load(OF_PY, "omni_formation_economy__p2s")           # _clock_income / _apex_income / _gravity_b
MR = _load(MOTION_REWARDS_PY, "crab_hex_omni_motion_rewards__p2s")   # THE LANDED KERNELS (pure torch)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _lineno(path: Path, pattern: str) -> int | None:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def _const_from_source(path: Path, name: str) -> float:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return float(ast.literal_eval(node.value))
    raise SystemExit(f"{name} not found in {path}")


def sig2(x: float) -> float:
    """2 significant figures (P2S_PLAN "round to 2 s.f.")."""
    return float(f"{x:.2g}") if x else 0.0


# ================================================================================================ params
def s1_params() -> dict:
    """Every number the replay uses, read off the S1 s18 run's own dumps and the landed knob module."""
    env = RT._yaml_load_tolerant(S1_RUN / "params/env.yaml")
    agent = RT._yaml_load_tolerant(S1_RUN / "params/agent.yaml")
    tp = RT.train_params_from_dumps(env, agent, f"S1 s18 dump: {_rel(S1_RUN / 'params')}")
    rw = env["rewards"]
    weights = {k: float(v.get("weight") or 0.0) for k, v in rw.items() if isinstance(v, dict) and v.get("weight")}
    pool = str(env["commands"]["base_velocity"]["cmd_pool"])
    share = {tok.split(":")[0].strip(): float(tok.split(":")[1]) for tok in pool.split(",")}
    tib = rw["penalty_tibia_deviation_in_stance"]["params"]
    p = {
        "tp": tp,
        "weights": weights,
        "dt": tp.dt,
        "action_clip": tp.action_clip,
        "cmd_pool": pool,
        "stand_share": share.get("stand"),
        "tibia_joints": list(tib["asset_cfg"]["joint_names"]),
        "tibia_feet": list(tib["sensor_cfg"]["body_names"]),
        "tibia_contact": float(tib["contact_force_threshold"]) if "contact_force_threshold" in tib else 0.1,
        "collision_bodies": list(rw["reward_collision"]["params"]["sensor_cfg"]["body_names"]),
        "cross_min_cmd": float(rw["penalty_lin_vel_y"]["params"].get("min_cmd_norm", 0.12)),
        "progress_min_cmd": float(rw["reward_forward_progress_along_command"]["params"]["min_cmd_norm"]),
        "progress_cap": float(rw["reward_forward_progress_along_command"]["params"]["max_speed_scale"]),
        "excess_max_feet": int(rw["penalty_excess_feet_contact_forward"]["params"]["max_feet_on_ground"]),
        "excess_contact": float(rw["penalty_excess_feet_contact_forward"]["params"]["contact_force_threshold"]),
        "excess_min_cmd": float(rw["penalty_excess_feet_contact_forward"]["params"]["min_forward_speed_cmd"]),
        "clock_combine": str(rw["reward_clock_schedule"]["params"].get("combine", "product")),
        "load_ref_n": _const_from_source(MOTION_KNOBS_PY, "STAND_LOAD_REF_N"),
        "upright_min": _const_from_source(MOTION_KNOBS_PY, "STAND_UPRIGHT_MIN"),
        "tripod_offset": _const_from_source(MOTION_REWARDS_PY, "STAND_SUPPORT_TRIPOD"),
        "env_yaml": _rel(S1_RUN / "params/env.yaml"),
    }
    p["checks"] = {
        "stand slot share 0.15 (S1 dump cmd_pool)": p["stand_share"] == STAND_SHARE_PLAN,
        "kernel load_ref_n 50 N = gait_eval MOTION_TOUCH_REF_N": p["load_ref_n"] == ST.TOUCH_REF_N == 50.0,
        "kernel upright_min 0.9": p["upright_min"] == 0.9,
        "tripod offset 3": p["tripod_offset"] == 3.0,
        "tibia joints FL..RR in foot order": [j.split("_")[0] for j in p["tibia_joints"]] == list(FEET)
                                             and [b.split("_")[0] for b in p["tibia_feet"]] == list(FEET),
        "dt 0.02": abs(p["dt"] - 0.02) < 1e-12,
        "clock combine product": p["clock_combine"] == "product",
        "replay_motion_terms notes empty": not tp.notes,
    }
    return p


# ================================================================================================ landed kernels
class _Scene(dict):
    def __init__(self, robot, sensors):
        super().__init__(robot=robot)
        self.sensors = sensors


_BODY_IDS = list(range(6))
_SENSOR = types.SimpleNamespace(name="contact_forces", body_ids=_BODY_IDS)
_ASSET = types.SimpleNamespace(name="robot", body_ids=_BODY_IDS)


def stub_env(force, vel, gz, stand, *, root_z=None, foot_z=None, dtype=None):
    """A duck-typed env for the landed kernels from per-step arrays: ``force`` (S, 6) N (placed on z, so its norm is
    the recorded norm), ``vel`` (S, 6, 3) foot ``body_lin_vel_w``, ``gz`` (S,) = projected_gravity_b z, ``stand`` (S,)
    the emulated ``stand_mask()``. ``root_z`` / ``foot_z`` ride along (the crouch counterfactual moves them; the
    kernels must not read them)."""
    dtype = dtype or torch.float64
    force = torch.as_tensor(np.asarray(force), dtype=dtype)
    n = force.shape[0]
    net = torch.zeros(n, 6, 3, dtype=dtype)
    net[..., 2] = force
    grav = torch.zeros(n, 3, dtype=dtype)
    grav[:, 2] = torch.as_tensor(np.asarray(gz), dtype=dtype)
    data = types.SimpleNamespace(body_lin_vel_w=torch.as_tensor(np.asarray(vel), dtype=dtype),
                                 projected_gravity_b=grav)
    if root_z is not None:
        data.root_pos_w = torch.zeros(n, 3, dtype=dtype)
        data.root_pos_w[:, 2] = torch.as_tensor(np.asarray(root_z), dtype=dtype)
    if foot_z is not None:
        data.body_pos_w = torch.zeros(n, 6, 3, dtype=dtype)
        data.body_pos_w[..., 2] = torch.as_tensor(np.asarray(foot_z), dtype=dtype)
    stand_t = torch.as_tensor(np.asarray(stand, dtype=bool))
    term = types.SimpleNamespace(stand_mask=lambda: stand_t)
    return types.SimpleNamespace(scene=_Scene(types.SimpleNamespace(data=data),
                                              {"contact_forces": types.SimpleNamespace(
                                                  data=types.SimpleNamespace(net_forces_w=net))}),
                                 command_manager=types.SimpleNamespace(get_term=lambda name: term))


def landed_terms(force, vel, gz, stand, *, load_ref_n=50.0, upright_min=0.9, dtype=None, **kw):
    """``(T1, T2)`` per step from the LANDED ``reward_stand_support`` / ``penalty_stand_slide`` (unweighted)."""
    env = stub_env(force, vel, gz, stand, dtype=dtype, **kw)
    t1 = MR.reward_stand_support(env, _SENSOR, _ASSET, load_ref_n=load_ref_n, upright_min=upright_min)
    t2 = MR.penalty_stand_slide(env, _SENSOR, _ASSET, load_ref_n=load_ref_n, upright_min=upright_min)
    return t1.double().numpy(), t2.double().numpy()


def np_terms(force, vel, gz, stand, *, load_ref_n=50.0, upright_min=0.9, tripod=3.0):
    """This module's numpy mirror of the two kernels (the per-step fidelity reference)."""
    s = np.clip(np.asarray(force, dtype=np.float64) / load_ref_n, 0.0, 1.0)
    g = (np.asarray(stand, dtype=bool) & (-np.asarray(gz, dtype=np.float64) > upright_min)).astype(np.float64)
    speed = np.linalg.norm(np.asarray(vel, dtype=np.float64), axis=-1)
    return (s.sum(axis=1) - tripod) * g, (s * speed).sum(axis=1) * g


def gravity_z(quat) -> np.ndarray:
    """projected_gravity_b z of a wxyz quaternion series (the omni-formation tool's ``_gravity_b``)."""
    return np.asarray(OF._gravity_b(np.asarray(quat, dtype=np.float64))[2])


# ================================================================================================ existing S1 terms
_CTX: dict = {}


def _ctx() -> dict:
    if not _CTX:
        p = s1_params()
        _CTX["p"] = p
    return _CTX


def existing_terms(z: dict, p: dict) -> dict[str, np.ndarray]:
    """Every replayed nonzero-weight S1 term, WEIGHTED, per reward step (length S = T - 1; the module docstring)."""
    tp, w = p["tp"], p["weights"]
    T = int(np.asarray(z["actions"]).shape[0])
    S = T - 1
    jn = z["_joint_names"]
    rod_idx = [i for i, n in enumerate(jn) if "Hip_Femur" in n or "Femur_Tibia" in n]
    q0 = RT.resolve_default_pos(tp.default_joint_pos, jn)
    ser = RT.step_series(z, tp, rod_idx=rod_idx, q0=q0)
    cmd = np.asarray(z["cmd_applied"], dtype=np.float64)[:S]

    def post(key):
        return np.asarray(z[key], dtype=np.float64)[1:T]

    v_b = post("root_lin_vel_b")
    force = post("foot_force_norm")
    speed_cmd = np.linalg.norm(cmd[:, :2], axis=1)
    dir_b = cmd[:, :2] / np.maximum(speed_cmd, 1e-6)[:, None]
    gx, gy, gz = OF._gravity_b(post("root_quat_w"))
    upright = (-gz > OF.MIN_UPRIGHT_GZ).astype(np.float64)
    # tibia deviation in stance: joints and feet both FL..RR (checked in s1_params)
    tib_j = [jn.index(n) for n in p["tibia_joints"]]
    q = post("joint_pos")
    tib = np.sum(np.square(q[:, tib_j] - q0[tib_j]) * (force > p["tibia_contact"]), axis=1)
    # delta torques: the term's previous buffer is zeroed at reset (tau_{-1} = 0)
    tau = post("applied_torque")
    dtau = np.sum(np.square(np.diff(np.vstack([np.zeros((1, tau.shape[1])), tau]), axis=0)), axis=1)
    # collision: Hip / Femur leg links (leg_link_names order); the trunk "body" contact is not recorded
    links = z["_leg_link_names"]
    coll_ids = [i for i, n in enumerate(links) if re.fullmatch(r".*_Hip|.*_Femur", n)]
    coll = (np.sum(post("leg_contact_force")[:, coll_ids] > 0.1, axis=1).astype(np.float64)
            if "leg_contact_force" in z else np.zeros(S))
    # clock income (0.001): the omni-formation tool's kernels with the recorded phase
    v_clock = np.maximum(speed_cmd, OF.OMNI_CLOCK_TURN_R * np.abs(cmd[:, 2]))
    running = v_clock > OF.CLOCK_CMD_STOP_M_S
    phase = np.asarray(z["clock_phase"], dtype=np.float64)[:S] if "clock_phase" in z else np.zeros(S)
    phi = phase[:, None] + OF.FOOT_OFFSETS[None, :]
    fv = post("foot_lin_vel_w")
    _csum, cprod = OF._clock_income(phi, force, np.linalg.norm(fv[..., :2], axis=-1), running)
    height = post("foot_pos_w")[..., 2] - (post("root_pos_w")[:, 2] + OF.GROUND_OFFSET_FROM_ROOT_M)[:, None]
    apex = OF._apex_income(phi, height, running)
    cross = v_b[:, 0] * (-dir_b[:, 1]) + v_b[:, 1] * dir_b[:, 0]
    progress = np.clip(np.sum(v_b[:, :2] * (cmd[:, :2] / (speed_cmd[:, None] + 1e-8)), axis=1), 0.0,
                       p["progress_cap"]) * (speed_cmd > p["progress_min_cmd"])
    excess = (np.maximum((force > p["excess_contact"]).sum(axis=1) - p["excess_max_feet"], 0.0)
              * (v_clock > p["excess_min_cmd"]))
    return {
        "track_lin_vel_xy_exp": w.get("track_lin_vel_xy_exp", 0.0) * ser["inc_lin"],
        "track_ang_vel_z_exp": w.get("track_ang_vel_z_exp", 0.0) * ser["inc_ang"],
        "reward_tracking_yaw": w.get("reward_tracking_yaw", 0.0) * ser["inc_facing"],
        "reward_yaw_progress": w.get("reward_yaw_progress", 0.0) * ser["inc_yaw_progress"],
        "reward_action_rate": w.get("reward_action_rate", 0.0) * ser["action_rate"],
        "reward_feet_air_time_positive": w.get("reward_feet_air_time_positive", 0.0) * ser["airtime"],
        "penalty_foot_idle_when_forward": w.get("penalty_foot_idle_when_forward", 0.0) * ser["foot_idle"],
        "penalty_tracking_error_l1": w.get("penalty_tracking_error_l1", 0.0)
                                     * np.linalg.norm(cmd[:, :2] - v_b[:, :2], axis=1),
        "penalty_lin_vel_y": w.get("penalty_lin_vel_y", 0.0) * cross ** 2 * (speed_cmd > p["cross_min_cmd"]),
        "reward_forward_progress_along_command": w.get("reward_forward_progress_along_command", 0.0) * progress,
        "reward_orientation": w.get("reward_orientation", 0.0) * (gx ** 2 + gy ** 2),
        "reward_lin_vel_z": w.get("reward_lin_vel_z", 0.0) * v_b[:, 2] ** 2,
        "reward_delta_torques": w.get("reward_delta_torques", 0.0) * dtau,
        "penalty_tibia_deviation_in_stance": w.get("penalty_tibia_deviation_in_stance", 0.0) * tib,
        "penalty_excess_feet_contact_forward": w.get("penalty_excess_feet_contact_forward", 0.0) * excess,
        "reward_collision": w.get("reward_collision", 0.0) * coll,
        "reward_clock_schedule": w.get("reward_clock_schedule", 0.0) * cprod * upright,
        "reward_clock_swing_apex": w.get("reward_clock_swing_apex", 0.0) * apex * upright,
    }


EXISTING_NOT_REPLAYED = {
    "reward_stride_length": "gated off at ||cmd|| <= min_cmd_norm 0.12 (0 on every stand row)",
    "reward_feet_edge": "x-edge mask x contact on terrain_levels > 3; the suite stand is the flat ring (no edges)",
    "reward_feet_stumble": "needs |F_xy| > 4 |F_z|; the raws keep only the force norm (a still stand on flat ground "
                           "carries ~vertical loads)",
}


# ================================================================================================ one episode
_KEYS = ("foot_force_norm", "foot_pos_w", "foot_lin_vel_w", "terrain_z", "root_pos_w", "root_quat_w",
         "root_lin_vel_b", "root_ang_vel_b", "cmd_applied", "actions", "joint_vel", "joint_pos", "applied_torque",
         "heading_err", "segment_id", "steady_mask", "crab_failure", "leg_contact_force", "clock_phase")


def _read(path: str) -> dict:
    with np.load(path) as npz:
        return {k: npz[k] for k in _KEYS if k in npz.files}


def pit_mask(foot_z, terrain_z, loaded, offset: float, tol: float = PIT_TOL_M) -> np.ndarray:
    """(S, 6) bool: the terrain under the foot lies more than ``tol`` below the PAD LEVEL the robot stands on. Pad
    level per row = the median over the LOADED feet of (foot_z - the ``p2s_stand`` pooled stance offset), i.e. the
    ground its supporting feet are on (a row with no loaded foot falls back to the median terrain_z under the six
    feet). The foot's own lift never enters: a raised foot over flat ground is not over a pit."""
    fz, tz = np.asarray(foot_z, dtype=np.float64), np.asarray(terrain_z, dtype=np.float64)
    ld = np.asarray(loaded, dtype=bool)
    pad_src = np.where(ld, fz - offset, np.nan)
    any_ld = ld.any(axis=1)
    pad = np.where(any_ld, np.nanmedian(np.where(any_ld[:, None], pad_src, 0.0), axis=1), np.median(tz, axis=1))
    return tz < pad[:, None] - tol


def stand_episode(z: dict, hold_idx: int, p: dict, *, with_existing: bool = True, keep_g3: bool = False) -> dict:
    """One stand hold: the gate mask, the landed T1/T2 (float64 and float32), the numpy mirror, ``p2s_stand``'s own
    estimators, the existing-term sums, the over-pit shares and the G3 inputs (gate rows' force / velocity / gz)."""
    dt = p["dt"]
    T = int(np.asarray(z["foot_force_norm"]).shape[0])
    S = T - 1
    out: dict = {"S": S}
    full_raw, steady_raw = ST.hold_masks(z, hold_idx, dt)
    cmd = np.asarray(z["cmd_applied"], dtype=np.float64)[:S]
    xy_zero = np.all(cmd[:, :2] == 0.0, axis=1)
    wz_zero = cmd[:, 2] == 0.0
    in_hold = steady_raw[1:T]                                    # step s <-> post row s + 1
    stand = in_hold & xy_zero                                    # the slot: facing-control wz allowed
    strict = stand & wz_zero                                     # sensitivity: cmd (vx, vy, wz) == 0
    force = np.asarray(z["foot_force_norm"], dtype=np.float64)[1:T]
    vel = np.asarray(z["foot_lin_vel_w"], dtype=np.float64)[1:T]
    gz = gravity_z(np.asarray(z["root_quat_w"])[1:T])
    t1, t2 = landed_terms(force, vel, gz, stand, load_ref_n=p["load_ref_n"], upright_min=p["upright_min"])
    t1f, t2f = landed_terms(force, vel, gz, stand, load_ref_n=p["load_ref_n"], upright_min=p["upright_min"],
                            dtype=torch.float32)
    n1, n2 = np_terms(force, vel, gz, stand, load_ref_n=p["load_ref_n"], upright_min=p["upright_min"],
                      tripod=p["tripod_offset"])
    n = int(stand.sum())
    upright = -gz > p["upright_min"]
    out.update({
        "n": n, "n_hold_steps": int(in_hold.sum()), "n_hold_xy_nonzero": int((in_hold & ~xy_zero).sum()),
        "n_hold_wz_nonzero": int((in_hold & ~wz_zero).sum()),
        "n_not_upright": int((stand & ~upright).sum()),
        "t1_sum": float(t1[stand].sum()), "t2_sum": float(t2[stand].sum()),
        "n_strict": int(strict.sum()),
        "t1_sum_strict": float(t1[strict].sum()), "t2_sum_strict": float(t2[strict].sum()),
        "finite": bool(np.isfinite(t1).all() and np.isfinite(t2).all()),
        "off_gate_nonzero": int(np.count_nonzero(t1[~stand]) + np.count_nonzero(t2[~stand])),
        "step_fid": float(max(np.abs(t1 - n1).max(initial=0.0), np.abs(t2 - n2).max(initial=0.0))),
        "f32_dev": float(max(np.abs(t1f - n1).max(initial=0.0), np.abs(t2f - n2).max(initial=0.0))),
    })
    out["t1"] = out["t1_sum"] / n if n else None
    out["t2"] = out["t2_sum"] / n if n else None
    m = ST.episode_stand_metrics(z, hold_idx, dt)
    out["st"] = {k: m.get(k) for k in ("t1_per_s", "t2_per_s", "off_ground_share", "steps_per_s", "clearance_m",
                                       "loaded_count", "hover", "fall", "stance_offset_m", "ks_m_s",
                                       "drift_m_s")}
    # over-pit share (:func:`pit_mask`)
    off = m.get("stance_offset_m")
    if off is not None and n:
        rows = np.flatnonzero(stand) + 1
        fz = np.asarray(z["foot_pos_w"], dtype=np.float64)[rows, :, 2]
        tz = np.asarray(z["terrain_z"], dtype=np.float64)[rows]
        loaded = np.asarray(z["foot_force_norm"])[rows] > ST.TOUCH_REF_N
        pit = pit_mask(fz, tz, loaded, off)
        out["pit_feet"] = int(pit.sum())
        out["pit_loaded"] = int((pit & loaded).sum())
        out["n_loaded"] = int(loaded.sum())
        out["pit_any_row"] = int(pit.any(axis=1).sum())
    if with_existing and n:
        ex = existing_terms(z, p)
        out["existing"] = {k: float(v[stand].sum()) for k, v in ex.items()}
        out["existing_strict"] = {k: float(v[strict].sum()) for k, v in ex.items()}
    if keep_g3:
        out["g3"] = {"force": force[stand], "vel": vel[stand], "gz": gz[stand],
                     "root_z": np.asarray(z["root_pos_w"], dtype=np.float64)[1:T, 2][stand],
                     "foot_z": np.asarray(z["foot_pos_w"], dtype=np.float64)[1:T, :, 2][stand]}
    return out


def nonstand_rows(z: dict, stand_idx: int | None, stand_wave: bool, p: dict) -> dict:
    """G4: every reward step that is NOT the emulated stand slot, through the landed kernels."""
    T = int(np.asarray(z["foot_force_norm"]).shape[0])
    S = T - 1
    cmd = np.asarray(z["cmd_applied"], dtype=np.float64)[:S]
    seg = np.asarray(z["segment_id"]).astype(int)[1:T]
    stand = (seg == stand_idx) & np.all(cmd[:, :2] == 0.0, axis=1) if (stand_wave and stand_idx is not None) \
        else np.zeros(S, dtype=bool)
    force = np.asarray(z["foot_force_norm"], dtype=np.float64)[1:T]
    vel = np.asarray(z["foot_lin_vel_w"], dtype=np.float64)[1:T]
    gz = gravity_z(np.asarray(z["root_quat_w"])[1:T])
    t1, t2 = landed_terms(force, vel, gz, stand, load_ref_n=p["load_ref_n"], upright_min=p["upright_min"])
    o1, o2 = landed_terms(force, vel, gz, np.ones(S, dtype=bool), load_ref_n=p["load_ref_n"],
                          upright_min=p["upright_min"])
    off = ~stand
    return {"n": int(off.sum()), "nonzero": int(np.count_nonzero(t1[off]) + np.count_nonzero(t2[off])),
            "finite": bool(np.isfinite(t1[off]).all() and np.isfinite(t2[off]).all()),
            "cmd_zero_rows": int((off & np.all(cmd == 0.0, axis=1)).sum()),
            "open_t1_abs_sum": float(np.abs(o1[off]).sum()), "open_t2_sum": float(o2[off].sum())}


# ================================================================================================ inputs
def stamps() -> dict:
    """``{name: (stamp, row, kind)}``; kind = det / samp / statue / wall."""
    s: dict = {}
    s["REF_es1"] = (ST.newest_stamp("REF_S1_s18", SUITE, 1), SUITE, "det")
    s["REF_es2"] = (ST.newest_stamp("REF_S1_s18", SUITE, 2), SUITE, "det")
    for short in ("C0", "CL", "TG", "A1", "F1"):
        s[short] = (ST._suite_stamp_for(short), SUITE, "det")
    s["statue"] = (ST.newest_stamp("DIAG_STATUE_s18", SUITE, 1), SUITE, "statue")
    s["E6_statue"] = (ST._E6_STATUE_STAMP, SUITE, "statue_old")
    s["REF_samp"] = (ST.newest_stamp("DIAG_REFSAMP_s18", SUITE, 1), SUITE, "samp")
    s["noisy_statue_samp"] = (ST.newest_stamp("DIAG_NOISYSTATUE_s18", SUITE, 1), SUITE, "samp")
    for label, name in (("REF", "REF_S1_s18"), ("C0", "P2__C0_s18"), ("TG", "P2__TG_s18")):
        s[f"{label}_field_turn"] = (ST.newest_stamp(name, FT, 1), FT, "wall")
        for seed in (1, 2):
            s[f"{label}_field_hard_es{seed}"] = (ST.newest_stamp(name, FH, seed), FH, "wall")
    missing = [k for k, v in s.items() if not v[0]]
    if missing:
        raise SystemExit(f"no complete stamp for {missing}")
    return s


G4_SOURCES = ("REF_es1", "TG", "REF_samp", "REF_field_turn", "REF_field_hard_es1")


def jobs_for(stamps_: dict, limit: int | None) -> list[tuple]:
    out = []
    for name, (st, row, kind) in stamps_.items():
        meta = json.loads((Path(st) / "run_meta.json").read_text())
        labels = meta["hold_labels"]
        hold = labels.index("stand")
        emap = meta.get("env_index_map") or [{"env_index": i, "label": None}
                                              for i in range(len(meta["termination_reason"]))]
        n_stand = 0
        for e in emap:
            is_stand = e.get("label") in (None, "stand")
            path = str(Path(st) / "raw" / f"episode_{int(e['env_index']):02d}.npz")
            if is_stand and (limit is None or n_stand < limit):
                n_stand += 1
                out.append(("stand", name, path, hold, True, meta["joint_names"], meta.get("leg_link_names") or []))
                if name in G4_SOURCES:     # the suite stand wave's settle hold / a schedule row's non-stand holds
                    out.append(("g4", name, path, hold, True, meta["joint_names"], []))
            elif not is_stand and name in G4_SOURCES:
                out.append(("g4", name, path, hold, False, meta["joint_names"], []))
    return out


def _init_worker() -> None:
    torch.set_num_threads(1)
    _ctx()


def _job(args):
    kind, name, path, hold, stand_wave, joints, links = args
    p = _ctx()["p"]
    z = _read(path)
    z["_joint_names"], z["_leg_link_names"] = list(joints), list(links)
    if kind == "g4":
        r = nonstand_rows(z, hold, stand_wave, p)
    else:
        r = stand_episode(z, hold, p, with_existing=True, keep_g3=(name == "statue"))
    r.update(kind=kind, name=name, path=path)
    return r


# ================================================================================================ statistics
def auc(a, b) -> float | None:
    """P(a > b) + 0.5 P(a == b) over all pairs."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if not a or not b:
        return None
    A, B = np.asarray(a, dtype=np.float64)[:, None], np.asarray(b, dtype=np.float64)[None, :]
    return float(((A > B).sum() + 0.5 * (A == B).sum()) / (A.size * B.size))


def pooled(eps: list[dict], key: str, var: str = "") -> float | None:
    """Step-weighted mean over the gate rows (``var`` "_strict": the cmd (vx, vy, wz) == 0 sensitivity rows)."""
    n = sum(e[f"n{var}"] for e in eps if e.get(f"n{var}"))
    return sum(e[f"{key}_sum{var}"] for e in eps if e.get(f"n{var}")) / n if n else None


def pooled_existing(eps: list[dict], var: str = "") -> dict:
    n = sum(e[f"n{var}"] for e in eps if e.get(f"n{var}"))
    terms = eps[0]["existing"].keys() if eps else []
    per = {t: sum(e[f"existing{var}"][t] for e in eps if e.get(f"n{var}")) / n for t in terms} if n else {}
    return {"per_term": per, "total": float(sum(per.values())), "n_steps": n}


def median(eps: list[dict], key: str) -> float | None:
    v = [e[key] for e in eps if e.get(key) is not None]
    return float(np.median(v)) if v else None


# ================================================================================================ gates
def solve_doses(B: float, dT1: float, dT2: float) -> dict:
    """P2S_PLAN's M rule; one weight pair per dose."""
    out = {}
    for dose, M in DOSES.items():
        # w_slide is SET by the rule (-0.5 M: -0.5, -1.25), not solved, so it is kept exact; 2 s.f. applies to the
        # solved w_support (rounding -1.25 would be a tie between -1.2 and -1.3 and move the rule's own dose)
        w_slide_exact = SLIDE_PER_M * M
        w_support_exact = (M - B - abs(w_slide_exact) * dT2) / dT1
        w_slide, w_support = w_slide_exact, sig2(w_support_exact)
        out[dose] = {"M": M, "w_support": w_support, "w_slide": w_slide,
                     "w_support_exact": w_support_exact, "w_slide_exact": w_slide_exact,
                     "realised_margin": B + w_support * dT1 + abs(w_slide) * dT2,
                     "feasible": w_support_exact > 0}
    return out


def g3_counterfactuals(statue_eps: list[dict], w_support: float, w_slide: float, M: float, p: dict) -> dict:
    """The three gaming counterfactuals on the statue's own gate rows, through the landed kernels."""
    f = np.concatenate([e["g3"]["force"] for e in statue_eps])
    v = np.concatenate([e["g3"]["vel"] for e in statue_eps])
    gz = np.concatenate([e["g3"]["gz"] for e in statue_eps])
    rz = np.concatenate([e["g3"]["root_z"] for e in statue_eps])
    fz = np.concatenate([e["g3"]["foot_z"] for e in statue_eps])
    on = np.ones(f.shape[0], dtype=bool)
    kw = {"load_ref_n": p["load_ref_n"], "upright_min": p["upright_min"]}

    def rate(force, vel, **extra):
        t1, t2 = landed_terms(force, vel, gz, on, **kw, **extra)
        return float(w_support * t1.mean() + w_slide * t2.mean()), float(t1.mean()), float(t2.mean())

    base, b1, b2 = rate(f, v)
    total = f.sum(axis=1, keepdims=True)
    hover = {}
    for name, ids in TRIPODS.items():
        stance = [i for i in range(6) if i not in ids]
        fh, vh = f.copy(), v.copy()
        fh[:, list(ids)] = 0.0
        fh[:, stance] = total / 3.0
        vh[:, list(ids)] = 0.0
        r, t1, t2 = rate(fh, vh)
        hover[name] = {"rate": r, "t1": t1, "t2": t2, "cost_vs_statue": base - r}
    worst = min(hover.values(), key=lambda h: h["cost_vs_statue"])
    f6 = np.repeat(total / 6.0, 6, axis=1)
    still6, s1, s2 = rate(f6, v)
    vs = v.copy()
    vs[..., 0] += SHUFFLE_V
    shuf, h1, h2 = rate(f6, vs)
    # the crouch: root (and the root-relative pad) 1 cm down at fixed feet; force / velocity / attitude unchanged
    c1, c2 = landed_terms(f, v, gz, on, **kw, root_z=rz - CROUCH_M, foot_z=fz)
    o1, o2 = landed_terms(f, v, gz, on, **kw, root_z=rz, foot_z=fz)
    crouch = float(max(np.abs(c1 - o1).max(initial=0.0), np.abs(c2 - o2).max(initial=0.0)))
    return {
        "n_rows": int(f.shape[0]), "statue": {"rate": base, "t1": b1, "t2": b2},
        "hover": {"per_tripod": hover, "cost": worst["cost_vs_statue"], "threshold": G3_HOVER_FRAC * M,
                  "pass": worst["cost_vs_statue"] >= G3_HOVER_FRAC * M},
        "shuffle": {"still_six_down": {"rate": still6, "t1": s1, "t2": s2},
                    "shuffle": {"rate": shuf, "t1": h1, "t2": h2}, "cost": still6 - shuf,
                    "ideal_cost": abs(w_slide) * 6 * SHUFFLE_V,
                    "threshold": G3_SHUFFLE_FRAC * M, "pass": still6 - shuf >= G3_SHUFFLE_FRAC * M},
        "crouch": {"max_abs_delta": crouch, "tol": CROUCH_TOL, "pass": crouch <= CROUCH_TOL},
        "pass": None,
    }


def ast_gate_check(src: str | None = None) -> dict:
    """Both terms return ``<expr> * gate`` with ``gate = _stand_gate(...)``, and ``_stand_gate`` reads
    ``...get_term(command_name).stand_mask()`` and returns ``(stand & upright)``."""
    src = src if src is not None else MOTION_REWARDS_PY.read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    res = {}
    for name in ("reward_stand_support", "penalty_stand_slide"):
        fn = fns.get(name)
        ok_assign = ok_ret = False
        if fn is not None:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "gate" for t in node.targets)
                        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "_stand_gate"):
                    ok_assign = True
            rets = [n for n in fn.body if isinstance(n, ast.Return)]
            if rets and isinstance(rets[-1].value, ast.BinOp) and isinstance(rets[-1].value.op, ast.Mult):
                r = rets[-1].value.right
                ok_ret = isinstance(r, ast.Name) and r.id == "gate"
        res[name] = {"gate_from__stand_gate": ok_assign, "returns_times_gate": ok_ret,
                     "pass": bool(fn is not None and ok_assign and ok_ret)}
    sg = fns.get("_stand_gate")
    reads_mask = combines = False
    if sg is not None:
        for node in ast.walk(sg):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "stand_mask"
                    and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute)
                    and node.func.value.func.attr == "get_term"):
                reads_mask = True
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
                names = {getattr(node.left, "id", None), getattr(node.right, "id", None)}
                combines = combines or names == {"stand", "upright"}
    res["_stand_gate"] = {"reads_stand_mask": reads_mask, "stand_and_upright": combines,
                          "pass": bool(reads_mask and combines)}
    res["pass"] = all(v["pass"] for v in res.values())
    return res


def emulation_check(p: dict) -> dict:
    """Mask off -> exactly 0; mask on but tipped (-g_z 0.5) -> exactly 0; mask on, upright -> nonzero."""
    rng = np.random.default_rng(11)
    f = rng.uniform(0.0, 800.0, (64, 6))
    v = rng.normal(0.0, 0.3, (64, 6, 3))
    kw = {"load_ref_n": p["load_ref_n"], "upright_min": p["upright_min"]}
    off = landed_terms(f, v, -np.ones(64), np.zeros(64, dtype=bool), **kw)
    tip = landed_terms(f, v, -0.5 * np.ones(64), np.ones(64, dtype=bool), **kw)
    on = landed_terms(f, v, -np.ones(64), np.ones(64, dtype=bool), **kw)
    r = {"mask_off_zero": bool(not np.any(off[0]) and not np.any(off[1])),
         "tipped_zero": bool(not np.any(tip[0]) and not np.any(tip[1])),
         "on_nonzero": bool(np.any(on[0]) and np.any(on[1]))}
    r["pass"] = all(r.values())
    return r


# ================================================================================================ analysis
DET_HEADS = ("REF_es1", "REF_es2", "C0", "CL", "TG", "A1", "F1")
WALLS = tuple(f"{h}_{r}" for h in ("REF", "C0", "TG") for r in ("field_turn", "field_hard_es1", "field_hard_es2"))


def analyse(stamps_: dict, eps: list[dict], p: dict) -> dict:
    by = {}
    for e in eps:
        if e["kind"] == "stand":
            by.setdefault(e["name"], []).append(e)
    g4_rows = [e for e in eps if e["kind"] == "g4"]
    res: dict = {"heads": {}}
    for name, lst in by.items():
        ex = pooled_existing(lst)
        res["heads"][name] = {
            "stamp": _rel(stamps_[name][0]), "kind": stamps_[name][2], "n_episodes": len(lst),
            "n_gate_steps": int(sum(e["n"] for e in lst)),
            "t1": pooled(lst, "t1"), "t2": pooled(lst, "t2"),
            "t1_median_ep": median(lst, "t1"), "t2_median_ep": median(lst, "t2"),
            "existing_total": ex["total"], "existing_per_term": ex["per_term"],
            "off_ground_median": float(np.median([e["st"]["off_ground_share"] for e in lst
                                                  if e["st"]["off_ground_share"] is not None])),
            "hover_episodes": int(sum(1 for e in lst if e["st"]["hover"])),
            "drift_median": float(np.median([e["st"]["drift_m_s"] for e in lst if e["st"]["drift_m_s"] is not None])),
            "falls": int(sum(1 for e in lst if e["st"]["fall"])),
            "n_not_upright": int(sum(e["n_not_upright"] for e in lst)),
            "n_hold_xy_nonzero": int(sum(e["n_hold_xy_nonzero"] for e in lst)),
            "n_hold_wz_nonzero": int(sum(e["n_hold_wz_nonzero"] for e in lst)),
            "finite": all(e["finite"] for e in lst),
            "off_gate_nonzero": int(sum(e["off_gate_nonzero"] for e in lst)),
        }
        if any("pit_feet" in e for e in lst):
            nf = sum(6 * e["n"] for e in lst if "pit_feet" in e)
            nl = sum(e["n_loaded"] for e in lst if "pit_feet" in e)
            res["heads"][name]["pits"] = {
                "share_feet_over_pit": sum(e["pit_feet"] for e in lst if "pit_feet" in e) / nf if nf else None,
                "share_loaded_feet_over_pit": sum(e["pit_loaded"] for e in lst if "pit_feet" in e) / nl if nl else None,
                "share_rows_any_foot_over_pit": (sum(e["pit_any_row"] for e in lst if "pit_feet" in e)
                                                 / sum(e["n"] for e in lst if "pit_feet" in e)),
            }
    H = res["heads"]

    # ---- fidelity: landed kernels (float64) vs p2s_stand numpy, per episode; per step vs the numpy mirror
    suite_eps = [e for e in eps if e["kind"] == "stand" and stamps_[e["name"]][1] == SUITE and e.get("n")]
    ep_diff, n_cmp, n_skip = 0.0, 0, 0
    for e in suite_eps:
        if e["n_not_upright"] or e["st"]["t1_per_s"] is None:
            n_skip += 1
            continue
        ep_diff = max(ep_diff, abs(e["t1"] - e["st"]["t1_per_s"]), abs(e["t2"] - e["st"]["t2_per_s"]))
        n_cmp += 1
    stand_eps = [e for e in eps if e["kind"] == "stand"]
    res["fidelity"] = {
        "episodes_compared": n_cmp, "episodes_skipped_not_upright": n_skip,
        "max_abs_diff_vs_p2s_stand": ep_diff,
        "max_abs_step_diff_vs_numpy_mirror": max(e["step_fid"] for e in stand_eps),
        "float32_max_abs_step_dev": max(e["f32_dev"] for e in stand_eps),
        "n_steps": int(sum(e["S"] for e in stand_eps)), "tolerance": FIDELITY_TOL,
    }
    res["fidelity"]["ok"] = bool(n_cmp > 0 and ep_diff < FIDELITY_TOL
                                 and res["fidelity"]["max_abs_step_diff_vs_numpy_mirror"] < FIDELITY_TOL)

    # ---- B (frozen before the solve), dT1, dT2
    ref_det = by["REF_es1"] + by["REF_es2"]
    statue = by["statue"]
    ex_ref, ex_st = pooled_existing(ref_det), pooled_existing(statue)
    B = ex_st["total"] - ex_ref["total"]
    t1_st, t2_st = pooled(statue, "t1"), pooled(statue, "t2")
    t1_ref, t2_ref = pooled(ref_det, "t1"), pooled(ref_det, "t2")
    dT1, dT2 = t1_st - t1_ref, t2_ref - t2_st
    res["B"] = {"value": B, "statue_income": ex_st["total"], "ref_income": ex_ref["total"],
                "per_term": {t: ex_st["per_term"][t] - ex_ref["per_term"][t] for t in ex_ref["per_term"]},
                "statue_per_term": ex_st["per_term"], "ref_per_term": ex_ref["per_term"],
                "per_ref_seed": {s: ex_st["total"] - H[s]["existing_total"] for s in ("REF_es1", "REF_es2")},
                "not_replayed": EXISTING_NOT_REPLAYED,
                "definition": "existing S1-weight term income/s on suite stand gate rows (steady window), fresh "
                              "statue minus REF (deterministic, es1 + es2 pooled); computed before the dose solve"}
    res["T"] = {"t1_statue": t1_st, "t2_statue": t2_st, "t1_ref": t1_ref, "t2_ref": t2_ref, "dT1": dT1, "dT2": dT2,
                "t1_statue_median": median(statue, "t1"), "t2_statue_median": median(statue, "t2"),
                "t1_ref_median": median(ref_det, "t1"), "t2_ref_median": median(ref_det, "t2")}
    doses = solve_doses(B, dT1, dT2)
    # sensitivity: the literal cmd (vx, vy, wz) == 0 gate (p2_replay's), same solve
    sens: dict = {}
    ex_ref_s, ex_st_s = pooled_existing(ref_det, "_strict"), pooled_existing(statue, "_strict")
    B_s = ex_st_s["total"] - ex_ref_s["total"]
    dT1_s = pooled(statue, "t1", "_strict") - pooled(ref_det, "t1", "_strict")
    dT2_s = pooled(ref_det, "t2", "_strict") - pooled(statue, "t2", "_strict")
    sens["B"], sens["dT1"], sens["dT2"] = B_s, dT1_s, dT2_s
    sens["doses"] = {k: {kk: v[kk] for kk in ("w_support", "w_slide", "w_support_exact")}
                     for k, v in solve_doses(B_s, dT1_s, dT2_s).items()}
    sens["rows_dropped"] = {k: int(sum(e["n"] - e["n_strict"] for e in by[k])) for k in by}
    nsn, rsn = by["noisy_statue_samp"], by["REF_samp"]
    sens["sampled"] = {}
    for dose, d in doses.items():
        m1 = d["w_support"] * (pooled(nsn, "t1", "_strict") - pooled(rsn, "t1", "_strict"))
        m2 = d["w_slide"] * (pooled(nsn, "t2", "_strict") - pooled(rsn, "t2", "_strict"))
        sens["sampled"][dose] = {"t1_margin": m1, "t2_margin": m2}
    sens["anti_termination_min"] = {
        dose: min(pooled_existing(by[h], "_strict")["total"] + d["w_support"] * pooled(by[h], "t1", "_strict")
                  + d["w_slide"] * pooled(by[h], "t2", "_strict") for h in ("REF_es1", "REF_es2", "REF_samp"))
        for dose, d in doses.items()}
    res["sensitivity_strict_gate"] = sens
    # info: the same solve on per-episode MEDIANS (step 2's p2s_noise_probe convention)
    dT1_m = res["T"]["t1_statue_median"] - res["T"]["t1_ref_median"]
    dT2_m = res["T"]["t2_ref_median"] - res["T"]["t2_statue_median"]
    res["median_solve"] = {"dT1": dT1_m, "dT2": dT2_m,
                           "doses": {k: {kk: v[kk] for kk in ("w_support", "w_slide", "w_support_exact")}
                                     for k, v in solve_doses(B, dT1_m, dT2_m).items()}}
    for dose, d in doses.items():
        d["realised_margin_per_ref_seed"] = {
            s: (res["heads"]["statue"]["existing_total"] - H[s]["existing_total"]
                + d["w_support"] * (t1_st - H[s]["t1"]) + abs(d["w_slide"]) * (H[s]["t2"] - t2_st))
            for s in ("REF_es1", "REF_es2")}
    res["doses"] = doses

    # ---- G1
    st_t1 = [e["t1"] for e in statue]
    st_t2 = [e["t2"] for e in statue]
    g1 = {}
    for h in DET_HEADS:
        a1 = auc(st_t1, [e["t1"] for e in by[h]])
        a2 = auc([-x for x in st_t2], [-e["t2"] for e in by[h]])
        g1[h] = {"auc_t1": a1, "auc_t2": a2,
                 "pass": bool(a1 is not None and a2 is not None and a1 >= G1_AUC_MIN and a2 >= G1_AUC_MIN)}
    g1_old = {h: {"auc_t1": auc([e["t1"] for e in by["E6_statue"]], [e["t1"] for e in by[h]]),
                  "auc_t2": auc([-e["t2"] for e in by["E6_statue"]], [-e["t2"] for e in by[h]])} for h in DET_HEADS}
    res["G1"] = {"per_head": g1, "threshold": G1_AUC_MIN, "pass": all(v["pass"] for v in g1.values()),
                 "info_E6_statue": g1_old}

    # ---- G2
    g2 = {}
    for h in DET_HEADS:
        rho, n = RT.spearman([e["t1"] for e in by[h]], [e["st"]["off_ground_share"] for e in by[h]])
        g2[h] = {"rho": rho, "n": n}
    for h in G2_DECISIVE:
        g2[h]["decisive"] = True
        g2[h]["pass"] = bool(g2[h]["rho"] is not None and g2[h]["rho"] <= G2_RHO_MAX)
    res["G2"] = {"per_head": g2, "threshold": G2_RHO_MAX, "decisive": list(G2_DECISIVE),
                 "pass": all(g2[h]["pass"] for h in G2_DECISIVE),
                 "population": "suite stands, steady window, per-episode T1/s vs p2s_stand off-ground share"}

    # ---- G3 (per dose)
    g3 = {}
    for dose, d in doses.items():
        r = g3_counterfactuals(statue, d["w_support"], d["w_slide"], d["M"], p)
        r["pass"] = bool(r["hover"]["pass"] and r["shuffle"]["pass"] and r["crouch"]["pass"])
        g3[dose] = r
    res["G3"] = {"per_dose": g3, "pass": all(v["pass"] for v in g3.values())}

    # ---- G4
    ast_r = ast_gate_check()
    emu = emulation_check(p)
    g4_data = {"rows": int(sum(r["n"] for r in g4_rows)), "nonzero": int(sum(r["nonzero"] for r in g4_rows)),
               "finite": all(r["finite"] for r in g4_rows),
               "cmd_zero_rows": int(sum(r["cmd_zero_rows"] for r in g4_rows)),
               "open_gate_t1_abs_sum": float(sum(r["open_t1_abs_sum"] for r in g4_rows)),
               "open_gate_t2_sum": float(sum(r["open_t2_sum"] for r in g4_rows)),
               "stand_episode_off_gate_nonzero": int(sum(e["off_gate_nonzero"] for e in stand_eps)),
               "sources": list(G4_SOURCES)}
    g4_data["pass"] = bool(g4_data["nonzero"] == 0 and g4_data["finite"] and g4_data["rows"] > 0
                           and g4_data["stand_episode_off_gate_nonzero"] == 0)
    res["G4"] = {"data": g4_data, "ast": ast_r, "emulation": emu,
                 "pass": bool(g4_data["pass"] and ast_r["pass"] and emu["pass"])}

    # ---- G5
    g5 = {w: {"finite": H[w]["finite"], "n_gate_steps": H[w]["n_gate_steps"], "t1": H[w]["t1"], "t2": H[w]["t2"],
              "pits": H[w].get("pits"), "falls": H[w]["falls"], "n_episodes": H[w]["n_episodes"]} for w in WALLS}
    res["G5"] = {"per_stand": g5, "pass": all(v["finite"] and v["n_gate_steps"] > 0 for v in g5.values()),
                 "pit_definition": f"terrain_z under the foot more than {PIT_TOL_M} m below the row's pad level (median "
                                   "over loaded feet of foot_z minus the p2s_stand pooled stance offset), over gate rows",
                 "flat_control": {h: H[h].get("pits") for h in ("REF_es1", "statue")}}

    # ---- anti-termination
    def net(h, d):
        return H[h]["existing_total"] + d["w_support"] * H[h]["t1"] + d["w_slide"] * H[h]["t2"]

    anti = {}
    for dose, d in doses.items():
        gated = {h: net(h, d) for h in ("REF_es1", "REF_es2", "REF_samp")}
        walls = {w: net(w, d) for w in WALLS}
        info = {h: net(h, d) for h in ("C0", "CL", "TG", "A1", "F1", "statue", "noisy_statue_samp")}
        anti[dose] = {"gated": gated, "walls_reported": walls, "info": info, "threshold": ANTI_TERM_MIN,
                      "pass": all(v >= ANTI_TERM_MIN for v in gated.values())}
    res["anti_termination"] = {"per_dose": anti, "pass": all(v["pass"] for v in anti.values()),
                               "walls_note": "field rows priced with the flat-tile forms of reward_orientation / "
                                             "reward_lin_vel_z (the off-flat forms zero / halve them, so this is "
                                             "the conservative side for a floor)"}

    # ---- sampled margins (reported)
    ns, rs = H["noisy_statue_samp"], H["REF_samp"]
    samp = {"B_sampled": ns["existing_total"] - rs["existing_total"],
            "t1_noisy_statue": ns["t1"], "t1_ref_samp": rs["t1"], "t2_noisy_statue": ns["t2"], "t2_ref_samp": rs["t2"],
            "per_dose": {}}
    for dose, d in doses.items():
        m1 = d["w_support"] * (ns["t1"] - rs["t1"])
        m2 = d["w_slide"] * (ns["t2"] - rs["t2"])
        det = d["w_support"] * dT1 + abs(d["w_slide"]) * dT2
        m1_med = d["w_support"] * (median(by["noisy_statue_samp"], "t1") - median(by["REF_samp"], "t1"))
        m2_med = d["w_slide"] * (median(by["noisy_statue_samp"], "t2") - median(by["REF_samp"], "t2"))
        det_med = (d["w_support"] * (median(statue, "t1") - median(by["REF_es1"], "t1"))
                   + abs(d["w_slide"]) * (median(by["REF_es1"], "t2") - median(statue, "t2")))
        samp["per_dose"][dose] = {"t1_margin": m1, "t2_margin": m2, "t1_plus_t2": m1 + m2,
                                  "f_median_step2_convention": (m1_med + m2_med) / det_med if det_med else None,
                                  "det_t1_plus_t2": det, "f": (m1 + m2) / det if det else None,
                                  "f_t1_only": m1 / (d["w_support"] * dT1) if d["w_support"] else None,
                                  "total_with_existing": samp["B_sampled"] + m1 + m2}
    samp["t2_reverses"] = bool(ns["t2"] > rs["t2"])
    res["sampled"] = samp

    # ---- liveness (TensorBoard units)
    share = p["stand_share"]
    lt = _len_factor()
    live = {}
    for dose, d in doses.items():
        er1 = d["w_support"] * share * rs["t1"]
        er2 = d["w_slide"] * share * rs["t2"]
        live[dose] = {
            "reward_stand_support": {"episode_reward": er1, "per_stand_row_mean": rs["t1"],
                                     "episode_reward_x_len": er1 * lt[0] if lt[0] else None,
                                     "episode_reward_det_ref": d["w_support"] * share * t1_ref},
            "penalty_stand_slide": {"episode_reward": er2, "per_stand_row_mean": rs["t2"],
                                    "episode_reward_x_len": er2 * lt[0] if lt[0] else None,
                                    "episode_reward_det_ref": d["w_slide"] * share * t2_ref},
            "implied_sum_s_f": 3.0 + er1 / (share * d["w_support"]) if d["w_support"] else None,
        }
    res["liveness"] = {"per_dose": live, "stand_share": share, "len_factor": lt[0], "len_factor_source": lt[1],
                       "relation": "implied mean sum_f s_f on stand rows = 3 + ER(reward_stand_support) / "
                                   "(stand_share x w_support); a dead T1 reads ER = 0 -> 3 (keep the non-zero "
                                   "requirement beside the [2.5, 6] band)",
                       "statistic": "SAMPLED REF (DIAG_REFSAMP_s18) suite stand gate rows, steady window"}

    # ---- verdict
    reasons = []
    for k, ok in (("fidelity", res["fidelity"]["ok"]), ("G1", res["G1"]["pass"]), ("G2", res["G2"]["pass"]),
                  ("G3", res["G3"]["pass"]), ("G4", res["G4"]["pass"]), ("G5", res["G5"]["pass"]),
                  ("anti_termination", res["anti_termination"]["pass"]),
                  ("params checks", all(p["checks"].values())),
                  ("doses feasible", all(d["feasible"] for d in doses.values()))):
        if not ok:
            reasons.append(k)
    if not res["G1"]["pass"]:
        reasons.append("G1 heads failing: " + ", ".join(h for h, v in g1.items() if not v["pass"]))
    if not res["G2"]["pass"]:
        reasons.append("G2: " + ", ".join(f"{h} rho {g2[h]['rho']}" for h in G2_DECISIVE if not g2[h]["pass"]))
    res["verdict"] = "PASS" if not reasons else "FAIL"
    res["reasons"] = reasons
    return res


def _len_factor() -> tuple[float | None, str]:
    try:
        cal = json.loads(REPLAY_JSON.read_text())["p2b"]["calibration"]["tb_tail_500"]["P2__S1_s18"]
        L = float(cal["Train/mean_episode_length"])
        return L / TRAIN_EP_STEPS, f"replay_gate.json p2b.calibration.tb_tail_500.P2__S1_s18 mean_episode_length {L:.1f} / {TRAIN_EP_STEPS}"
    except (OSError, ValueError, KeyError, TypeError):
        return None, "unavailable"


# ================================================================================================ report
def _f(x, nd=3, sign=False):
    if x is None:
        return "."
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _g(x):
    return "." if x is None else f"{x:.3g}"


def render(res: dict, p: dict, generated: str) -> str:
    L = []
    A = L.append
    D, H = res["doses"], res["heads"]
    A("# Replay gate p2s: stand-slot support income (T1) and slide penalty (T2)")
    A("")
    A(f"CHARTER §11 / P2S_PLAN.md step 4 offline replay gate. CPU only: saved eval raws, no Isaac, training or GPU. "
      f"Generated by `scripts/p2s_replay.py` on {generated}. The numbers are also in `replay_gate.json` under `p2s` "
      "(the E1 / F1 / p2b entries are untouched).")
    A("")
    A("## Verdict")
    A("")
    A(f"**{res['verdict']}**" + ("" if res["verdict"] == "PASS" else f": {'; '.join(res['reasons'])}") + ".")
    A("")
    A("| dose | M (/s) | `KRABBY_STAND_SUPPORT_W` | `KRABBY_STAND_SLIDE_W` | realised det margin (/s) | "
      "Episode_Reward T1 / T2 (TB units) |")
    A("|---|---|---|---|---|---|")
    for dose, d in D.items():
        lv = res["liveness"]["per_dose"][dose]
        A(f"| **{dose}** | {d['M']} | **{d['w_support']:g}** | **{d['w_slide']:g}** | {_f(d['realised_margin'], 3, True)} "
          f"| {_f(lv['reward_stand_support']['episode_reward'], 4, True)} / "
          f"{_f(lv['penalty_stand_slide']['episode_reward'], 4, True)} |")
    A("")
    A("| check | rule | result |")
    A("|---|---|---|")
    fid = res["fidelity"]
    A(f"| fidelity | landed kernels = p2s_stand numpy to {FIDELITY_TOL:g} | {'PASS' if fid['ok'] else 'FAIL'} "
      f"(max {_g(fid['max_abs_diff_vs_p2s_stand'])} per episode, {_g(fid['max_abs_step_diff_vs_numpy_mirror'])} per step) |")
    worst1 = min(v["auc_t1"] for v in res["G1"]["per_head"].values())
    worst2 = min(v["auc_t2"] for v in res["G1"]["per_head"].values())
    A(f"| G1 statue separation | AUC ≥ {G1_AUC_MIN} vs every head, T1 and T2 | {'PASS' if res['G1']['pass'] else 'FAIL'} "
      f"(worst T1 {worst1:.3f}, worst T2 {worst2:.3f}) |")
    g2 = res["G2"]["per_head"]
    A(f"| G2 T1 tracks lifting | ρ(T1/s, off-ground) ≤ {G2_RHO_MAX} on TG and CL | {'PASS' if res['G2']['pass'] else 'FAIL'} "
      f"(TG {_f(g2['TG']['rho'], 2, True)}, CL {_f(g2['CL']['rho'], 2, True)}) |")
    g3 = res["G3"]["per_dose"]
    A(f"| G3 gaming counterfactuals | hover ≥ 0.5·M, shuffle ≥ 0.25·M, crouch 0 | {'PASS' if res['G3']['pass'] else 'FAIL'} "
      + "(" + "; ".join(f"{k}: hover {v['hover']['cost']:.2f} ≥ {v['hover']['threshold']:.2f}, shuffle "
                        f"{v['shuffle']['cost']:.2f} ≥ {v['shuffle']['threshold']:.3f}" for k, v in g3.items()) + ") |")
    g4 = res["G4"]
    A(f"| G4 zero off the stand slot | exactly 0 on non-stand rows; AST + emulation | {'PASS' if g4['pass'] else 'FAIL'} "
      f"({g4['data']['nonzero']} nonzero of {g4['data']['rows']:,} rows) |")
    A(f"| G5 wall stands | finite; pit share reported | {'PASS' if res['G5']['pass'] else 'FAIL'} |")
    at = res["anti_termination"]["per_dose"]
    A(f"| anti-termination | marching head's net stand-slot reward ≥ +{ANTI_TERM_MIN}/s (det es1, es2; sampled) | "
      f"{'PASS' if res['anti_termination']['pass'] else 'FAIL'} ("
      + "; ".join(f"{k}: min {min(v['gated'].values()):+.2f}" for k, v in at.items()) + ") |")
    A("")
    A("Per P2S_PLAN step 4, a failed check stops and asks. No rule was tuned to pass.")
    A("")
    # ---- method
    A("## Method")
    A("")
    A("**Inputs** (newest complete stamp per name; every stand is the 12-s suite stand hold unless noted):")
    A("")
    A("| name | kind | stamp | episodes | gate steps |")
    A("|---|---|---|---|---|")
    for name, h in H.items():
        A(f"| {name} | {h['kind']} | `{h['stamp']}` | {h['n_episodes']} | {h['n_gate_steps']:,} |")
    A("")
    A("- **Step alignment.** Reward step s pairs `cmd_applied[s]` with the post-physics state at raw index s+1, as "
      "`replay_motion_terms.step_series` does.")
    A("- **Gate.** `stand_mask()` is emulated as (the stand hold, and on the suite the stand-category wave only) AND "
      "cmd (vx, vy) = 0 (`p2_replay` also required wz = 0; see the next bullet); the landed kernels then apply the "
      "upright gate (−g_z > 0.9) "
      "themselves. Window: the stand hold's steady rows (`p2s_stand.hold_masks`: fall-trimmed, first 1.0 s of the hold "
      "dropped), the rows `p2s_stand`'s own numpy T1/T2 read. Per-second values are step-weighted means over these rows.")
    nu = sum(h["n_not_upright"] for h in H.values())
    nxy = sum(h["n_hold_xy_nonzero"] for h in H.values())
    nwz = {k: h["n_hold_wz_nonzero"] for k, h in H.items() if h["n_hold_wz_nonzero"]}
    A(f"- Stand-hold steady rows with a nonzero translation command: {nxy}. Gate rows failing the upright test: {nu}. "
      "Rows carrying a facing-control yaw-rate command (wz ≠ 0, the command term's P-control past its dead band) are "
      "stand-slot rows in training (`stand_mask()` is the slot, not \"slot and zero turn\", P2S_PLAN \"The lever\"), so "
      "the primary gate keeps them: " + ", ".join(f"{k} {v}" for k, v in nwz.items()) + ". The literal "
      "cmd (vx, vy, wz) = 0 gate is run as a sensitivity (below).")
    A(f"- **Kernels.** `reward_stand_support` / `penalty_stand_slide` imported from "
      f"`{_rel(MOTION_REWARDS_PY)}` (e1a08baf) and called on a stub env built from the raws: the recorded "
      "`foot_force_norm` on z (norm preserved), `foot_lin_vel_w` as `body_lin_vel_w`, projected_gravity_b from "
      f"`root_quat_w`. Params from `motion_knobs.py`: load_ref_n {p['load_ref_n']:g} N, upright_min {p['upright_min']:g}.")
    A(f"- **Fidelity.** {fid['episodes_compared']} suite stand episodes: the landed kernels (float64) equal "
      f"`p2s_stand`'s numpy T1/T2 to {_g(fid['max_abs_diff_vs_p2s_stand'])} (episode means; tolerance {FIDELITY_TOL:g}); "
      f"per step against this module's numpy mirror {_g(fid['max_abs_step_diff_vs_numpy_mirror'])} over "
      f"{fid['n_steps']:,} steps. In float32 (training's dtype) the per-step deviation is "
      f"{_g(fid['float32_max_abs_step_dev'])}.")
    A("")
    # ---- B and doses
    B, T = res["B"], res["T"]
    A("## B, named before the solve")
    A("")
    A(f"B = existing S1-weight income/s on the suite stand gate rows, fresh statue minus REF (deterministic, es1 + es2 "
      f"pooled) = **{B['value']:+.4f}/s** (statue {B['statue_income']:.4f}/s, REF {B['ref_income']:.4f}/s; per REF seed "
      + ", ".join(f"{k} {v:+.4f}" for k, v in B["per_ref_seed"].items()) + "). Weights: the S1 s18 run's own "
      f"`{p['env_yaml']}`.")
    A("")
    A("| term (weight) | statue /s | REF /s | still − march |")
    A("|---|---|---|---|")
    for t, v in sorted(B["per_term"].items(), key=lambda kv: -abs(kv[1])):
        A(f"| {t} ({p['weights'].get(t, 0):g}) | {B['statue_per_term'][t]:+.4f} | {B['ref_per_term'][t]:+.4f} | {v:+.4f} |")
    A(f"| **total** | **{B['statue_income']:+.4f}** | **{B['ref_income']:+.4f}** | **{B['value']:+.4f}** |")
    A("")
    A("Not replayed: " + "; ".join(f"`{k}` ({v})" for k, v in B["not_replayed"].items()) + ".")
    A("")
    A(f"**T1 / T2 on the same rows (pooled means):** statue T1 {T['t1_statue']:.4f}, T2 {T['t2_statue']:.4f} m/s; "
      f"REF T1 {T['t1_ref']:.4f}, T2 {T['t2_ref']:.4f} m/s. So **dT1 = {T['dT1']:.4f}** and **dT2 = {T['dT2']:.4f}** "
      f"(episode medians: statue {T['t1_statue_median']:.3f} / {T['t2_statue_median']:.3f}, REF {T['t1_ref_median']:.3f} / "
      f"{T['t2_ref_median']:.3f}). M is measured against the statue's MEASURED T1 (+{T['t1_statue']:.2f} pooled, "
      f"+{T['t1_statue_median']:.1f} median: static indeterminacy leaves one grounded foot at 0 N in most episodes), "
      "not six-down +3. The doses use pooled means (the reward's expectation); the median solve is shown below.")
    A("")
    A("**Solve** (w_slide = −0.5·M; w_support = (M − B − |w_slide|·dT2)/dT1; 2 s.f.):")
    A("")
    A("| dose | M | w_slide | w_support exact → 2 s.f. | realised margin (pooled) | per REF seed |")
    A("|---|---|---|---|---|---|")
    for dose, d in D.items():
        A(f"| {dose} | {d['M']} | {d['w_slide']:g} | {d['w_support_exact']:.4f} → **{d['w_support']:g}** | "
          f"{d['realised_margin']:+.3f} | " + ", ".join(f"{k} {v:+.3f}" for k, v in d["realised_margin_per_ref_seed"].items())
          + " |")
    A("")
    A("Planning estimates were ST1 (+0.38, −0.50) and ST2 (+1.12, −1.25). w_slide = −0.5·M is set by the rule, "
      "not solved, so it stays exact (−0.5, −1.25); 2 s.f. applies to the solved w_support.")
    A("")
    ms, sg = res["median_solve"], res["sensitivity_strict_gate"]
    A("**Sensitivities (info, not the doses of record):**")
    A("")
    A("| variant | B | dT1 | dT2 | ST1 (w_support, w_slide) | ST2 (w_support, w_slide) |")
    A("|---|---|---|---|---|---|")
    A(f"| of record: slot gate, pooled means | {B['value']:+.4f} | {T['dT1']:.4f} | {T['dT2']:.4f} | "
      + " | ".join(f"({D[k]['w_support']:g}, {D[k]['w_slide']:g})" for k in D) + " |")
    A(f"| per-episode medians (step 2's convention) | {B['value']:+.4f} | {ms['dT1']:.4f} | {ms['dT2']:.4f} | "
      + " | ".join(f"({ms['doses'][k]['w_support']:g}, {ms['doses'][k]['w_slide']:g})" for k in D) + " |")
    A(f"| literal cmd (vx, vy, wz) = 0 gate | {sg['B']:+.4f} | {sg['dT1']:.4f} | {sg['dT2']:.4f} | "
      + " | ".join(f"({sg['doses'][k]['w_support']:g}, {sg['doses'][k]['w_slide']:g})" for k in D) + " |")
    A("")
    A("The literal gate drops " + ", ".join(f"{k} {v}" for k, v in sg["rows_dropped"].items() if v)
      + " rows; under it the sampled margins are " + "; ".join(
          f"{k} T1 {v['t1_margin']:+.3f}, T2 {v['t2_margin']:+.3f}" for k, v in sg["sampled"].items())
      + " and the anti-termination minimum is " + ", ".join(f"{k} {v:+.3f}" for k, v in sg["anti_termination_min"].items())
      + ".")
    A("")
    # ---- G1
    A("## G1: statue separation (per-episode T1/s, T2/s)")
    A("")
    A("| head | T1 /s (pooled) | T2 m/s (pooled) | AUC T1 (statue higher) | AUC T2 (statue lower) | pass |")
    A("|---|---|---|---|---|---|")
    A(f"| statue | {H['statue']['t1']:.3f} | {H['statue']['t2']:.3f} | . | . | . |")
    for h, v in res["G1"]["per_head"].items():
        A(f"| {h} | {H[h]['t1']:.3f} | {H[h]['t2']:.3f} | {v['auc_t1']:.3f} | {v['auc_t2']:.3f} | "
          f"{'yes' if v['pass'] else '**no**'} |")
    A("")
    A("The 2026-09-23 E6 statue (a deterministic zero-action rollout of the same row and seed, so it replays the "
      "fresh statue) gives the same picture (info): worst AUC T1 "
      f"{min(v['auc_t1'] for v in res['G1']['info_E6_statue'].values()):.3f}, worst T2 "
      f"{min(v['auc_t2'] for v in res['G1']['info_E6_statue'].values()):.3f}.")
    A("")
    # ---- G2
    A("## G2: T1 tracks lifting (within head)")
    A("")
    A("Spearman ρ(per-episode T1/s, `p2s_stand` off-ground share), suite stands, steady window. Decisive: TG and CL. "
      "Not blind: the audit measured TG −0.90 / −0.91 and CL −0.53 to −0.55.")
    A("")
    A("| head | ρ | n | off-ground (median) | hover episodes | decisive | pass |")
    A("|---|---|---|---|---|---|---|")
    for h, v in g2.items():
        A(f"| {h} | {_f(v['rho'], 3, True)} | {v['n']} | {H[h]['off_ground_median']:.3f} | {H[h]['hover_episodes']} | "
          f"{'yes' if v.get('decisive') else ''} | {('yes' if v['pass'] else '**no**') if 'pass' in v else ''} |")
    A("")
    # ---- G3
    A("## G3: gaming counterfactuals (landed kernels, the statue's own gate rows)")
    A("")
    A("- **Hover:** one tripod's force set to 0 (still; raised height does not enter either kernel), its weight moved "
      "equally onto the other three feet; the worse of the two tripods counts.")
    A(f"- **Shuffle:** the statue's total load spread over six feet (all six loaded), every foot +{SHUFFLE_V} m/s along "
      "world x, against the same six-down stance at the statue's recorded foot speeds.")
    A(f"- **Crouch:** root z −{CROUCH_M * 100:.0f} cm at fixed feet (root and foot heights carried on the stub env).")
    A("")
    A("| dose | statue rate | hover rate (worst) | hover cost ≥ 0.5·M | still six-down | shuffle | shuffle cost ≥ 0.25·M | "
      "crouch max |Δ| | pass |")
    A("|---|---|---|---|---|---|---|---|---|")
    for dose, v in g3.items():
        worst = min(v["hover"]["per_tripod"].values(), key=lambda h: h["cost_vs_statue"])
        A(f"| {dose} | {v['statue']['rate']:+.3f} | {worst['rate']:+.3f} | {v['hover']['cost']:+.3f} ≥ "
          f"{v['hover']['threshold']:.3f} | {v['shuffle']['still_six_down']['rate']:+.3f} | "
          f"{v['shuffle']['shuffle']['rate']:+.3f} | {v['shuffle']['cost']:+.3f} ≥ {v['shuffle']['threshold']:.3f} "
          f"(ideal {v['shuffle']['ideal_cost']:.3f}) | {v['crouch']['max_abs_delta']:.1e} | {'yes' if v['pass'] else '**no**'} |")
    A("")
    # ---- G4
    A("## G4: zero off the stand slot")
    A("")
    d4 = g4["data"]
    A(f"- **Data:** {d4['rows']:,} non-stand reward steps (the other nine suite waves and the stand wave's settle hold of "
      f"{', '.join(d4['sources'][:3])}; every non-stand hold of {', '.join(d4['sources'][3:])}) through the landed "
      f"kernels: {d4['nonzero']} nonzero, all finite: {d4['finite']}. {d4['cmd_zero_rows']:,} of them carry cmd = 0 "
      "(settle, turn holds, stops) and are still zero, because the gate is the slot, not the command value. With the "
      f"gate forced open the same rows would pay Σ|T1| {d4['open_gate_t1_abs_sum']:.0f} and ΣT2 "
      f"{d4['open_gate_t2_sum']:.0f} (unweighted step sums), so the zeros come from the gate. Off-gate rows inside the "
      f"stand episodes: {d4['stand_episode_off_gate_nonzero']} nonzero.")
    A(f"- **AST:** both functions return `… * gate` with `gate = _stand_gate(…)`, and `_stand_gate` reads "
      f"`get_term(command_name).stand_mask()` and returns `stand & upright`: {g4['ast']['pass']}.")
    A(f"- **Emulation:** mask off → 0: {g4['emulation']['mask_off_zero']}; mask on, tipped (−g_z 0.5) → 0: "
      f"{g4['emulation']['tipped_zero']}; mask on, upright → nonzero: {g4['emulation']['on_nonzero']}.")
    A("")
    # ---- G5
    A("## G5: wall stands")
    A("")
    A(f"Pit: {res['G5']['pit_definition']}.")
    A("")
    A("| stand | episodes | gate steps | finite | T1 /s | T2 m/s | feet over a pit | loaded feet over a pit | rows with any foot over a pit | falls |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for w, v in res["G5"]["per_stand"].items():
        pt = v["pits"] or {}
        A(f"| {w} | {v['n_episodes']} | {v['n_gate_steps']:,} | {v['finite']} | {_f(v['t1'])} | {_f(v['t2'])} | "
          f"{_f(pt.get('share_feet_over_pit'))} | {_f(pt.get('share_loaded_feet_over_pit'))} | "
          f"{_f(pt.get('share_rows_any_foot_over_pit'))} | {v['falls']} |")
    A("")
    A("Neither kernel reads terrain or foot height, so a foot over a pit changes T1 / T2 only through its measured "
      "force and speed.")
    A("")
    # ---- anti-termination
    A("## Anti-termination: net stand-slot reward of the marching head")
    A("")
    A("Existing-term income/s + w_support·T1 + w_slide·T2 on the same gate rows; floor +0.5/s on REF es1, es2 "
      "(deterministic) and the sampled REF. The sampled rows' existing-term income is recomputed on those rows "
      "(sampled actions included, so action_rate carries the exploration noise).")
    A("")
    A("| stand | existing /s | T1 | T2 | " + " | ".join(f"net {k}" for k in D) + " |")
    A("|---|---|---|---|" + "---|" * len(D))
    rows = ["REF_es1", "REF_es2", "REF_samp"] + list(WALLS) + ["C0", "CL", "TG", "A1", "F1", "statue", "noisy_statue_samp"]
    for h in rows:
        nets = []
        for dose in D:
            a = at[dose]
            v = a["gated"].get(h, a["walls_reported"].get(h, a["info"].get(h)))
            nets.append(f"**{v:+.3f}**" if h in a["gated"] else f"{v:+.3f}")
        tag = " (gated)" if h in ("REF_es1", "REF_es2", "REF_samp") else (" (walls, reported)" if h in WALLS else "")
        A(f"| {h}{tag} | {H[h]['existing_total']:+.3f} | {H[h]['t1']:+.3f} | {H[h]['t2']:.3f} | " + " | ".join(nets) + " |")
    A("")
    A(f"Walls: {res['anti_termination']['walls_note']}.")
    A("")
    # ---- sampled
    s = res["sampled"]
    A("## Under sampling: the noisy statue vs sampled REF (reported, not a gate)")
    A("")
    A(f"Sampled T1: noisy statue {s['t1_noisy_statue']:.3f}, REF {s['t1_ref_samp']:.3f}. Sampled T2: noisy statue "
      f"{s['t2_noisy_statue']:.3f} m/s, REF {s['t2_ref_samp']:.3f} m/s. Existing-term margin under sampling "
      f"{s['B_sampled']:+.3f}/s (deterministic B {B['value']:+.3f}/s).")
    A("")
    A("| dose | T1 margin | T2 margin | T1 + T2 sampled | T1 + T2 deterministic | f | f (T1 only) |")
    A("|---|---|---|---|---|---|---|")
    for dose, v in s["per_dose"].items():
        A(f"| {dose} | {v['t1_margin']:+.3f} | {v['t2_margin']:+.3f} | {v['t1_plus_t2']:+.3f} | "
          f"{v['det_t1_plus_t2']:+.3f} | {v['f']:.3f} | {v['f_t1_only']:.3f} |")
    A("")
    pd = s["per_dose"]
    below = [f"{k} {v['f']:.3f}" for k, v in pd.items() if v["f"] < F_STEP2_LINE]
    A(f"**f at the final weights against step 2's {F_STEP2_LINE} stop line:** {pd['ST2']['f']:.3f} at ST2 and "
      f"{pd['ST1']['f']:.3f} at ST1 on pooled means; {pd['ST2']['f_median_step2_convention']:.3f} and "
      f"{pd['ST1']['f_median_step2_convention']:.3f} on step 2's per-episode-median convention (step 2 read 0.332 at "
      "the planning dose, ST1's 0.316). "
      + (f"On pooled means it is BELOW the line ({', '.join(below)}). " if below else "")
      + "Step 2's rule is not a step-4 gate; it is flagged because the doses moved. f is a lower bound (open-loop "
      "noisy statue).")
    A("")
    ns_t, rs_t = H["noisy_statue_samp"]["existing_per_term"], H["REF_samp"]["existing_per_term"]
    top = sorted(ns_t, key=lambda t: -abs(ns_t[t] - rs_t[t]))[:4]
    A(f"The existing-term margin under sampling is {s['B_sampled']:+.3f}/s because the open-loop noisy statue drifts "
      f"and turns (median drift {H['noisy_statue_samp']['drift_median']:.3f} m/s vs REF "
      f"{H['REF_samp']['drift_median']:.3f}; "
      f"{H['noisy_statue_samp']['n_hold_wz_nonzero']} rows of facing-control wz): largest per-term gaps "
      + ", ".join(f"{t} {ns_t[t] - rs_t[t]:+.3f}" for t in top) + ". A closed-loop still policy would not pay this.")
    A("")
    if s["t2_reverses"]:
        st2 = s["per_dose"].get("ST2", {})
        A(f"**T2 reverses under sampling.** The noisy still stance moves its loaded feet more than the noisy march "
          f"({s['t2_noisy_statue']:.3f} vs {s['t2_ref_samp']:.3f} m/s), so under sampling the slide term pays the march "
          f"{-st2.get('t2_margin', 0):.3f}/s more than the still stance at ST2 "
          f"({-s['per_dose']['ST1']['t2_margin']:.3f}/s at ST1). T1 carries the whole sampled margin. The noisy statue is "
          "open-loop (it cannot correct its own noise), so this is a bound on the sampled margin, not its value.")
    else:
        A("T2 does not reverse under sampling on these rows.")
    A("")
    # ---- liveness
    lv = res["liveness"]
    A("## Liveness magnitudes for p2_live")
    A("")
    A(f"TensorBoard units: `Episode_Reward/<term>` = weight × mean over all env-steps, so stand-slot share "
      f"{lv['stand_share']:g} (S1 dump `cmd_pool`) × the per-stand-row mean. Statistic: {lv['statistic']}.")
    A("")
    A("| dose | term | weight | per-stand-row mean | Episode_Reward | × episode-length factor | det REF (info) |")
    A("|---|---|---|---|---|---|---|")
    for dose, v in lv["per_dose"].items():
        for t, key in (("reward_stand_support", "w_support"), ("penalty_stand_slide", "w_slide")):
            e = v[t]
            A(f"| {dose} | `{t}` | {D[dose][key]:g} | {e['per_stand_row_mean']:.4f} | **{e['episode_reward']:+.5f}** | "
              f"{_f(e['episode_reward_x_len'], 5, True)} | {e['episode_reward_det_ref']:+.5f} |")
    A("")
    A(f"Implied-Σs_f relation: {lv['relation']}. At the sampled REF rows: " + ", ".join(
        f"{k} {v['implied_sum_s_f']:.3f}" for k, v in lv["per_dose"].items()) + f". Episode-length factor "
      f"{_f(lv['len_factor'], 4)} ({lv['len_factor_source']}).")
    A("")
    # ---- params
    A("## Parameters and their sources")
    A("")
    A("| item | value and source |")
    A("|---|---|")
    A(f"| T1 / T2 kernels | `{_rel(MOTION_REWARDS_PY)}`:{_lineno(MOTION_REWARDS_PY, 'def reward_stand_support')} / "
      f":{_lineno(MOTION_REWARDS_PY, 'def penalty_stand_slide')} (gate :{_lineno(MOTION_REWARDS_PY, 'def _stand_gate')}) |")
    A(f"| load_ref_n / upright_min | {p['load_ref_n']:g} / {p['upright_min']:g} (`{_rel(MOTION_KNOBS_PY)}`:"
      f"{_lineno(MOTION_KNOBS_PY, '^STAND_LOAD_REF_N')}-{_lineno(MOTION_KNOBS_PY, '^STAND_UPRIGHT_MIN')}) |")
    A(f"| existing weights | `{p['env_yaml']}` (every nonzero-weight term; table above) |")
    A(f"| stand-slot share | {p['stand_share']:g} (`commands.base_velocity.cmd_pool` = `{p['cmd_pool']}`) |")
    A(f"| tracking-error L1 | `{_rel(LIB_REWARDS_PY)}`:{_lineno(LIB_REWARDS_PY, 'def penalty_tracking_error_l1')} |")
    A(f"| orientation / lin_vel_z | `{_rel(LIB_REWARDS_PY)}`:{_lineno(LIB_REWARDS_PY, 'def reward_orientation')} / "
      f":{_lineno(LIB_REWARDS_PY, 'def reward_lin_vel_z')} (flat-tile forms) |")
    A(f"| delta torques / collision / tibia | `{_rel(LIB_REWARDS_PY)}`:{_lineno(LIB_REWARDS_PY, 'class reward_delta_torques')} / "
      f":{_lineno(LIB_REWARDS_PY, 'def reward_collision')} / :{_lineno(LIB_REWARDS_PY, 'def penalty_joint_deviation_when_in_contact')} |")
    A(f"| cross-command | `{_rel(OMNI_REWARDS_PY)}`:{_lineno(OMNI_REWARDS_PY, 'def penalty_cross_command_vel_l2')} |")
    A(f"| clock income | `{_rel(OF_PY)}` `_clock_income` / `_apex_income` (recorded phase) |")
    A("")
    A("Source checks: " + "; ".join(f"{k}: {'ok' if v else '**FAIL**'}" for k, v in p["checks"].items()) + ".")
    A("")
    A("## Caveats")
    A("")
    A("- The eval rolls out the deterministic mean; the sampled pair (REF, noisy statue) is the training-time statistic "
      "and is gated only for anti-termination.")
    A("- `reward_collision` reads only the Hip / Femur links (`leg_contact_force`); the trunk contact is not recorded. "
      "`reward_feet_stumble` is not replayable from force norms. Neither fires on a still upright stand.")
    A("- The window is the steady stand hold of a 12-s eval hold after a 2-s settle; training stand slots also begin "
      "mid-stride, which this window does not contain.")
    return "\n".join(L) + "\n"


# ================================================================================================ main
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def to_json(res: dict, p: dict, generated: str) -> dict:
    """The ``replay_gate.json`` "p2s" entry: per-dose arms in the p2b shape (``p2s[dose]["terms"][term]
    ["episode_reward"]``) plus every gate."""
    out = {"verdict": res["verdict"], "reasons": res["reasons"]}
    for dose, d in res["doses"].items():
        lv = res["liveness"]["per_dose"][dose]
        out[dose] = {
            "M": d["M"],
            "knobs": {"KRABBY_STAND_SUPPORT_W": d["w_support"], "KRABBY_STAND_SLIDE_W": d["w_slide"]},
            "w_support_exact": d["w_support_exact"], "w_slide_exact": d["w_slide_exact"],
            "realised_margin_det": d["realised_margin"], "realised_margin_per_ref_seed": d["realised_margin_per_ref_seed"],
            "terms": {
                "reward_stand_support": {"weight": d["w_support"], "knob": "KRABBY_STAND_SUPPORT_W", **lv["reward_stand_support"]},
                "penalty_stand_slide": {"weight": d["w_slide"], "knob": "KRABBY_STAND_SLIDE_W", **lv["penalty_stand_slide"]},
            },
            "implied_sum_s_f": lv["implied_sum_s_f"],
            "anti_termination": res["anti_termination"]["per_dose"][dose],
            "G3": {k: v for k, v in res["G3"]["per_dose"][dose].items()},
            "sampled": res["sampled"]["per_dose"][dose],
        }
    out.update({
        "B": res["B"], "dT1": res["T"]["dT1"], "dT2": res["T"]["dT2"], "T": res["T"],
        "sensitivity": {"median_solve": res["median_solve"], "strict_cmd_zero_gate": res["sensitivity_strict_gate"],
                        "note": "info only; the doses of record are the slot gate on pooled means"},
        "gates": {
            "fidelity": res["fidelity"],
            "G1": res["G1"], "G2": res["G2"], "G3": {"pass": res["G3"]["pass"]}, "G4": res["G4"], "G5": res["G5"],
            "anti_termination": {"pass": res["anti_termination"]["pass"], "threshold": ANTI_TERM_MIN,
                                 "walls_note": res["anti_termination"]["walls_note"]},
        },
        "sampled": {k: v for k, v in res["sampled"].items() if k != "per_dose"},
        "liveness": {k: v for k, v in res["liveness"].items() if k != "per_dose"},
        "heads": {k: {kk: vv for kk, vv in v.items() if kk != "existing_per_term"} for k, v in res["heads"].items()},
        "rule": "P2S_PLAN step 4: fidelity; G1 AUC >= 0.95 both terms vs every head; G2 rho(T1, off-ground) <= -0.5 on "
                "TG and CL; G3 hover >= 0.5 M, shuffle >= 0.25 M, crouch 0; G4 exactly 0 off the stand slot; G5 finite "
                "on wall stands; anti-termination net >= +0.5/s (det REF es1/es2 and sampled REF) at both doses",
        "episode_reward_units": "TensorBoard Episode_Reward/<term> = weight x mean over all env-steps = weight x "
                                f"stand_share {res['liveness']['stand_share']:g} x the SAMPLED REF per-stand-row mean",
        "generated": generated,
        "report": _rel(REPORT_MD),
        "meta": {"script": _rel(HERE), "kernels": f"{_rel(MOTION_REWARDS_PY)} (landed e1a08baf)",
                 "env_yaml": p["env_yaml"], "param_checks": p["checks"], "cmd_pool": p["cmd_pool"]},
    })
    return _jsonable(out)


def merge_replay_json(path: Path, entry: dict) -> dict:
    """Write ``entry`` under the "p2s" key, every other key untouched."""
    cur = json.loads(path.read_text()) if path.is_file() else {}
    cur["p2s"] = entry
    path.write_text(json.dumps(cur, indent=1) + "\n")
    return cur


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="merge 'p2s' into replay_gate.json and write the report")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="stand episodes per stamp (smoke runs)")
    args = ap.parse_args(argv)

    p = _ctx()["p"]
    st = stamps()
    jobs = jobs_for(st, args.limit)
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker) as pool:
            eps = pool.map(_job, jobs, chunksize=8)
    else:
        eps = [_job(j) for j in jobs]
    res = analyse(st, eps, p)
    generated = _dt.datetime.now().isoformat(timespec="seconds")
    entry = to_json(res, p, generated)
    if args.write:
        merge_replay_json(REPLAY_JSON, entry)
        REPORT_MD.write_text(render(res, p, generated))
        print(f"wrote {REPLAY_JSON} ['p2s'] and {REPORT_MD}")
    summary = {"verdict": res["verdict"], "reasons": res["reasons"], "B": res["B"]["value"], "dT1": res["T"]["dT1"],
               "dT2": res["T"]["dT2"],
               "doses": {k: {kk: v[kk] for kk in ("w_support", "w_slide", "w_support_exact", "realised_margin")}
                         for k, v in res["doses"].items()}}
    print(json.dumps(_jsonable(summary), indent=1))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

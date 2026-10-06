#!/usr/bin/env python3
"""Omni phase 2 -- the CHARTER 4 offline replay gate for E1 (actuated power) and F1 (stand stance income). CPU ONLY.

Reads four saved ``suite__omni`` eval stamps (raw npz + metrics json; never Isaac, a GPU, training or git) and writes
``replay_gate.json`` (the CHARTER 9 shape, ``{"E1": {"episode_reward": x, ...}, "F1": {"episode_reward": y, ...}}``,
on rsl_rl's ``Episode_Reward`` basis = per max-episode-second) and ``REPLAY_GATE.md`` into the campaign directory.

    <isaac_venv>/bin/python p2_replay.py [--out DIR] [--workers N] [--limit N]

STAMPS (CHARTER 4): the B s18 c6x head (the gate head; E1 Gate 1 and both weights are read on it), W, the C0
side-first s16 c6x head (``SF__C0_rest_s16.record.json`` driver_stamp) and the E6 scripted statue (zero actions).

TERM MATH -- the implemented rewards, not a re-derivation:
  * ``penalty_mechanical_power`` and ``reward_stand_stance_quality`` are lifted out of
    ``parkour_isaaclab/envs/mdp/rewards.py`` with ``ast`` and exec'd against duck-typed stubs (the method
    ``tests/unit/test_crab_hex_omni_stand_power_knobs.py`` uses; ``rewards.py`` itself cannot import without Kit / pxr).
    ``reward_stand_stance_quality`` calls the real ``crab_hex_clock_reward.stand_stance_income``.
  * E1 = ``penalty_mechanical_power`` with ``asset_cfg.joint_ids`` = the articulation indices the forward cfg's
    ``_CRAB_ACTUATED_JOINT_NAMES`` regexes (read from ``parkour_mdp_cfg.py`` by ``ast``) fullmatch in the stamp's
    ``run_meta.joint_names`` -- 18 joints, asserted = the eval's ``motion_joint_groups`` "actuated" set, excluding
    exactly the six ``*_Body_Hip_RevoluteJoint`` (linkage emulator).
  * F1 = ``reward_stand_stance_quality`` with the params ``motion_knobs`` builds (``vel_ref`` = CLOCK_VEL_REF_M_S,
    ``min_upright_gz`` 0.9; checked against the source). Inputs exactly as training reads them: ``foot_force_norm``
    is the recorded ``net_forces_w_history[:, 0]`` norm (= ``net_forces_w``), ``foot_lin_vel_w`` is
    ``body_lin_vel_w`` of the six footpads, ``projected_gravity_b`` z from ``root_quat_w`` (w, x, y, z). The stand
    slot is emulated as ``stand_mask`` = the stand category's own hold AND cmd (vx, vy, wz) == 0 on that row (raws
    carry no slot id; CHARTER 4 "stand = cmd 0 and wz 0").
  * Timing as ``replay_motion_terms.py``: row t is recorded BEFORE env.step(actions[t]); env step s reads command row
    s and post-physics state row s + 1. The window is the own hold (``segment_id``), fall-trimmed as the eval does.
  * Task (locomotion) income = ``replay_motion_terms.task_income``: track_lin_vel_xy_exp + track_ang_vel_z_exp +
    reward_tracking_yaw (facing) + reward_yaw_progress at the TRAINING weights/params of each head's params/env.yaml
    (``train_params_from_dumps``), via the tool's own kernel functions.

E1 RULE (CHARTER 4 / the precedent 2026-09-22_gait_motion_quality REPLAY_GATE.md Gate 1): on the B head, per category,
Spearman rho of the per-episode cost (Q-normalised as the precedent: per m on fwd/side/back/diag, per rad on
turn/move_turn, per s on stand) vs the per-episode ``action_jitter`` and ``contact_switch_hz`` (the eval's
``motion_quality_metrics``, recomputed and cross-checked against the stamp); PASS = the larger rho >= +0.5 (a cost must
rise with the defect) on stand AND on >= 4/6 moving categories. Weight: per moving category the median over the
best-tracking quartile (by income/s) of raw-cost / income (the precedent's Gate-2 ratio); the feasible band is
[max_c 0.01/r_c, min_c 0.05/r_c] (1-5 % on every moving category); the proposal is the band's geometric midpoint at 2
significant figures (negative: a penalty).

F1 RULE (CHARTER 4): stand rows only. (a) the statue beats every stepping head on the term (per-second income,
median over episodes; AUC reported); (b) DECISIVE: within each stepping head, rho(income/s, stand steps/s) <= -0.5,
steps/s = the eval's per-episode ``rigidity.steps_per_s`` (a step = an unloaded run >= 5 env steps of
``foot_force_norm`` at 50 N -- STAND_STEPPING.md's "steps/s [S]"), with the scratch measure's lift-offs/s (contact
>= 1 N -> air for >= 2 steps) and swings/s (lift-offs clearing >= 1 cm) as robustness columns; (c) rho with stand
lift clearance and tripod is reported only; (d) the term is asserted zero on every moving-category row; (e) weight =
the smallest w in [0.5, 1.0] (0.05 grid) at which a still stance beats the B head's stepping by >= 0.5/s on stand rows:
margin(w) = base + w * (I_statue - I_head), base = STAND_STEPPING.md section 4's c6x still-minus-head total (+0.168/s,
parsed from that file), I = the F1 income per second at weight 1. None in range -> FAIL.

EPISODE_REWARD BASIS (CHARTER 8 term liveness, 3x tolerance): rsl_rl logs ``Episode_Reward/<term>`` =
mean(episode_sum(w * value * dt)) / episode_length_s. Replayed as w * sum_c mix_c * rate_c * (L / T_max): rate_c =
the category's pooled raw rate per second, mix_c = the c6x command pool (KRABBY_OMNI_CMD_POOL in candidates.yaml) with
the stand share replaced by B's logged ``Metrics/base_velocity/stand_frac_actual``, L / T_max = B's logged
``Train/mean_episode_length`` / (episode_length_s / dt) -- all from the B c6x training log tail (last 200 its). The
same conversion is applied to the four income terms and compared with their logged Episode_Reward (cross-check).
F1 = w * stand_frac * I_head * (L / T_max) (the stand slot only).
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import json
import math
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
TOOL = OMNI / "experiments" / "tools" / "replay_motion_terms.py"
LIB_REWARDS_PY = PARKOUR / "parkour_isaaclab" / "envs" / "mdp" / "rewards.py"
FWD_MDP = TASKS / "crab_hex_forward_task" / "mdp"
FWD_MDP_CFG_PY = TASKS / "crab_hex_forward_task" / "config" / "crab_hex" / "agents" / "parkour_mdp_cfg.py"
MOTION_KNOBS_PY = OMNI / "config" / "crab_hex" / "motion_knobs.py"
STAND_STEPPING_MD = OMNI / "experiments" / "2026-10-01_ppo_construction" / "sf" / "STAND_STEPPING.md"
CANDIDATES_YAML = CAMPAIGN / "candidates.yaml"

GE = PARKOUR / "logs" / "rsl_rl" / "gait_eval" / "omni_phases"
C0_RECORD = PARKOUR / "outputs" / "omni_plan" / "ppo_construction" / "arms_p1" / "SF__C0_rest_s16.record.json"
B_TRAIN_DIR = PARKOUR / "logs" / "rsl_rl" / "crab_hex_omni_flat_walk" / "2026-10-06_02-24-21_seed18_X60_c6x_A15pB"

MOVING = ("fwd", "side", "back", "diag", "turn", "move_turn")
RHO_MIN = 0.5
E1_SHARE_BAND = (0.01, 0.05)
F1_MARGIN = 0.5
F1_W_RANGE = (0.5, 1.0)
F1_W_GRID = 0.05
F1_MIN_UPRIGHT_GZ = 0.9
WIN_STEPS = 50                     # 1-s windows for the stillest / jerkiest read
SCRATCH_CONTACT_N = 1.0            # STAND_STEPPING scratch measure: contact = foot_force_norm >= 1 N
SCRATCH_MIN_AIR = 2                # lift-off = airborne run >= 2 steps after contact
SCRATCH_LIFT_M = 0.01              # swing = lift-off clearing the foot's own median stance height by >= 1 cm
TB_TAIL = 200
#: the c6x command pool's categories -> suite categories (omni = an oblique translate = diag)
POOL_TO_CAT = {"stand": "stand", "fwd": "fwd", "side_l": "side", "side_r": "side", "back": "back", "omni": "diag",
               "turn": "turn", "fwd_turn": "move_turn", "side_l_turn": "move_turn", "side_r_turn": "move_turn",
               "omni_turn": "move_turn"}
INCOME_TB = {"inc_lin": "track_lin_vel_xy_exp", "inc_ang": "track_ang_vel_z_exp", "inc_facing": "reward_tracking_yaw",
             "inc_yaw_progress": "reward_yaw_progress"}


# ============================================================================ imports of the shared code
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RT = _load_module(TOOL, "replay_motion_terms__p2")          # the shared replay tool (read-only reuse)
if str(FWD_MDP) not in sys.path:
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_clock_reward as CLOCK  # noqa: E402  (torch only; the real F1 math)
import torch  # noqa: E402


def _source_of(path: Path, name: str) -> str:
    src = path.read_text()
    node = [n for n in ast.parse(src, str(path)).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def lift_reward_functions() -> dict:
    """``penalty_mechanical_power`` / ``reward_stand_stance_quality`` exec'd from rewards.py's source against stubs."""
    ns = {"torch": torch, "SceneEntityCfg": lambda name="robot", **kw: _ns(name=name, joint_ids=slice(None)),
          "Articulation": object, "ParkourManagerBasedRLEnv": object,
          "stand_stance_income": CLOCK.stand_stance_income, "CLOCK_VEL_REF_M_S": CLOCK.VEL_REF_M_S}
    for name in ("penalty_mechanical_power", "reward_stand_stance_quality"):
        exec(compile("from __future__ import annotations\n" + _source_of(LIB_REWARDS_PY, name),  # noqa: S102
                     str(LIB_REWARDS_PY), "exec"), ns)
    return {"power": ns["penalty_mechanical_power"], "stand": ns["reward_stand_stance_quality"]}


def actuated_regexes() -> list[str]:
    tree = ast.parse(FWD_MDP_CFG_PY.read_text())
    for n in tree.body:
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_CRAB_ACTUATED_JOINT_NAMES"
                                              for t in n.targets)):
            return [ast.literal_eval(e) for e in n.value.elts]
    raise SystemExit(f"_CRAB_ACTUATED_JOINT_NAMES not found in {FWD_MDP_CFG_PY}")


def f1_params_from_source() -> dict:
    """The F1 builder's params in motion_knobs.py (vel_ref / min_upright_gz) -- checked, not assumed."""
    src = MOTION_KNOBS_PY.read_text()
    i = src.index("STAND_STANCE_TERM: (lib_rew.reward_stand_stance_quality")
    block = src[i:i + 900]
    ok_vel = '"vel_ref": lib_rew.CLOCK_VEL_REF_M_S' in block
    m = re.search(r'"min_upright_gz":\s*([0-9.]+)', block)
    if not ok_vel or not m:
        raise SystemExit("motion_knobs F1 builder params changed: re-read before replaying")
    return {"vel_ref": float(CLOCK.VEL_REF_M_S), "min_upright_gz": float(m.group(1))}


def stand_stepping_base() -> tuple[float, str]:
    """STAND_STEPPING.md section 4's total still-minus-head (c6x weights, B head stand)."""
    txt = STAND_STEPPING_MD.read_text()
    m = re.search(r"\|\s*\*\*total\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*\+([0-9.]+)\*\*", txt)
    if not m:
        raise SystemExit(f"cannot parse the section-4 total row of {STAND_STEPPING_MD}")
    return float(m.group(3)), f"{STAND_STEPPING_MD.relative_to(REPO)} section 4 total row (head {m.group(1)}/s, ideal still {m.group(2)}/s)"


# ============================================================================ stamps
def stamps() -> dict:
    c0 = json.loads(C0_RECORD.read_text())
    drv = None

    def find(d):
        nonlocal drv
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "driver_stamp" and isinstance(v, str):
                    drv = drv or v
                find(v)
        elif isinstance(d, list):
            for v in d:
                find(v)
    find(c0)
    if not drv:
        raise SystemExit(f"no driver_stamp in {C0_RECORD}")
    return {
        "B": GE / "2026-10-01_ppo_construction/seed18_X60_c6x_A15pB/suite__omni/seed001/2026-10-06_07-56-23",
        "W": GE / "2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29",
        "C0": Path(drv),
        "statue": GE / "2026-09-23_clock_deprecation/step0_E6_statue/suite__omni/seed001/2026-09-23_21-59-41",
    }


HEAD_LABEL = {"B": "B s18 c6x (gate head)", "W": "W (old head)", "C0": "C0 side-first s16 c6x", "statue": "E6 statue"}
STEPPING = ("B", "W", "C0")


# ============================================================================ per-episode replay
_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    torch.set_num_threads(1)
    _CTX.clear()
    _CTX.update(ctx)
    _CTX["fns"] = lift_reward_functions()
    _CTX["M"] = RT.gait_metrics()


def _upright_gz(q: np.ndarray) -> np.ndarray:
    """-projected_gravity_b z = R[2,2] of the root quaternion (w, x, y, z)."""
    x, y = q[:, 1], q[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def _scratch_steps(force: np.ndarray, hz: np.ndarray, dur: float) -> dict:
    """STAND_STEPPING scratch measure (measure_stand.py): lift-offs/s, swings/s >= 1 cm, median clearance."""
    air = force < SCRATCH_CONTACT_N
    lifts, peaks = 0, []
    for f in range(force.shape[1]):
        a = air[:, f]
        st = hz[~a, f]
        base = np.median(st) if st.size else np.median(hz[:, f])
        t, T = 0, a.shape[0]
        while t < T:
            if a[t] and (t == 0 or not a[t - 1]):
                u = t
                while u < T and a[u]:
                    u += 1
                if t > 0 and (u - t) >= SCRATCH_MIN_AIR:
                    lifts += 1
                    peaks.append(float(hz[t:u, f].max() - base))
                t = u
            else:
                t += 1
    real = [p for p in peaks if p >= SCRATCH_LIFT_M]
    return {"liftoffs_per_s": lifts / dur, "swings_per_s": len(real) / dur,
            "clearance_m": float(np.median(real)) if real else None}


def replay_one(job: tuple) -> dict:
    head, idx, ent = job
    c = _CTX
    p, M, fns = c["params"][head], c["M"], c["fns"]
    run_dir = Path(c["stamps"][head])
    cat, label = ent["category"], ent["label"]
    ep_json_path = run_dir / "metrics" / f"episode_{idx:02d}.json"
    ep_json = json.loads(ep_json_path.read_text()) if ep_json_path.is_file() else {}
    reason = ep_json.get("termination_reason") or ""
    with np.load(run_dir / "raw" / f"episode_{idx:02d}.npz") as npz:
        z = {k: npz[k] for k in npz.files if k not in ("body_pos_w", "leg_contact_force", "foot_ang_vel_w")}
    seg = np.asarray(z["segment_id"]).astype(int)
    T = seg.shape[0]
    upper = max(0, T - c["fall_trim"]) if reason in ("fall", "hard_fall") else T
    rec = np.zeros(T, dtype=bool)
    rec[:upper] = True
    rec &= seg == c["hold_labels"][head].index(label)
    widx = np.flatnonzero(rec)
    out = {"head": head, "env_index": idx, "category": cat, "hold": label, "termination_reason": reason,
           "n_window": int(widx.size)}
    if widx.size < RT.MIN_WINDOW_STEPS:
        out["skipped"] = "window_too_short"
        return out
    S = T - 1
    dt = p.dt
    step_mask = rec[:S]
    cmd = np.asarray(z["cmd_applied"], dtype=np.float64)[:S]

    def post(key):
        return np.asarray(z[key], dtype=np.float64)[1:T]

    tau, qd = post("applied_torque"), post("joint_vel")
    env_p = _ns(scene={"robot": _ns(data=_ns(applied_torque=torch.from_numpy(tau), joint_vel=torch.from_numpy(qd)))})
    g = c["groups"][head]
    e1 = fns["power"](env_p, _ns(name="robot", joint_ids=g["actuated18"])).numpy()
    e1_cam = fns["power"](env_p, _ns(name="robot", joint_ids=g["camshaft"])).numpy()
    e1_rod = fns["power"](env_p, _ns(name="robot", joint_ids=g["rod"])).numpy()
    p_all = fns["power"](env_p, _ns(name="robot", joint_ids=slice(None))).numpy()

    # F1: the stand slot emulated as (stand category own hold) & cmd == 0 on that row
    force = post("foot_force_norm")
    fvel = post("foot_lin_vel_w")
    net = np.zeros(force.shape + (3,))
    net[..., 2] = force
    grav = np.zeros((S, 3))
    grav[:, 2] = -_upright_gz(post("root_quat_w"))
    cmd_zero = np.all(cmd == 0.0, axis=1)
    stand_row = cmd_zero & (cat == "stand") & step_mask
    stand_t = torch.from_numpy(stand_row)
    env_s = _ns(scene=_SceneD({"robot": _ns(data=_ns(body_lin_vel_w=torch.from_numpy(fvel),
                                                     projected_gravity_b=torch.from_numpy(grav)))},
                              {"contact_forces": _ns(data=_ns(net_forces_w=torch.from_numpy(net)))}),
                command_manager=_ns(get_term=lambda name: _ns(stand_mask=lambda: stand_t)))
    body6 = list(range(6))
    f1 = fns["stand"](env_s, _ns(name="robot", body_ids=body6), _ns(name="contact_forces", body_ids=body6),
                      vel_ref=c["f1_params"]["vel_ref"], min_upright_gz=c["f1_params"]["min_upright_gz"]).numpy()
    # the same income with the slot gate forced open (diagnostic only: what a cmd/category-blind gate would pay)
    env_s_open = _ns(scene=env_s.scene, command_manager=_ns(get_term=lambda name: _ns(
        stand_mask=lambda: torch.ones(S, dtype=torch.bool))))
    f1_open = fns["stand"](env_s_open, _ns(name="robot", body_ids=body6), _ns(name="contact_forces", body_ids=body6),
                           vel_ref=c["f1_params"]["vel_ref"], min_upright_gz=c["f1_params"]["min_upright_gz"]).numpy()

    wb = post("root_ang_vel_b")
    inc = {
        "inc_lin": RT.track_lin_vel_xy_exp(cmd, post("root_lin_vel_b"), p.track_lin_std),
        "inc_ang": RT.track_ang_vel_z_exp(cmd, wb, p.track_ang_std),
        "inc_facing": RT.facing_exp(post("heading_err")) if "heading_err" in z else np.zeros(S),
        "inc_yaw_progress": RT.yaw_progress(cmd, wb, p.yaw_progress_cap),
    }
    sums = {k: float(v[step_mask].sum() * dt) for k, v in inc.items()}
    n_steps = int(step_mask.sum())
    dur = n_steps * dt
    sums.update({"e1": float(e1[step_mask].sum() * dt), "e1_cam": float(e1_cam[step_mask].sum() * dt),
                 "e1_rod": float(e1_rod[step_mask].sum() * dt), "power_all24": float(p_all[step_mask].sum() * dt),
                 "f1": float(f1[step_mask].sum() * dt), "f1_open": float(f1_open[step_mask].sum() * dt),
                 "f1_rows_s": float(stand_row.sum() * dt)})
    moving_nonzero = int(np.count_nonzero(f1[step_mask])) if cat != "stand" else 0
    q = M.motion_quality_metrics(
        cat, dt=dt, actions=np.asarray(z["actions"])[widx], foot_force_norm=np.asarray(z["foot_force_norm"])[widx],
        foot_pos_w=np.asarray(z["foot_pos_w"])[widx], foot_lin_vel_w=np.asarray(z["foot_lin_vel_w"])[widx],
        root_pos_w=np.asarray(z["root_pos_w"])[widx], root_quat_w=np.asarray(z["root_quat_w"])[widx],
        cmd_wz=np.asarray(z["cmd_applied"])[widx, 2],
        joint_vel=np.asarray(z["joint_vel"])[widx] if "joint_vel" in z else None,
        applied_torque=np.asarray(z["applied_torque"])[widx] if "applied_torque" in z else None,
        groups=g["metric_groups"], clock_running=np.asarray(z["clock_running"])[widx] if "clock_running" in z else None,
        rod_speed=np.asarray(z["rod_speed"])[widx] if "rod_speed" in z else None, action_clip=p.action_clip,
        hold_s_nominal=c["hold_s"][head].get(label))
    quality = {k: q.get(k) for k in ("action_jitter", "contact_switch_hz", "distance_m", "net_yaw_rad", "duration_s",
                                     "six_down_frac", "liftoffs")}
    stored = (((ep_json.get("holds") or {}).get(label) or {}).get("quality")) or {}
    xdiff = {k: abs(float(quality[k]) - float(stored[k])) for k in ("action_jitter", "contact_switch_hz",
                                                                     "distance_m", "net_yaw_rad")
             if quality.get(k) is not None and stored.get(k) is not None}
    out.update({"n_steps": n_steps, "duration_s": dur, "sums": sums,
                "income": RT.task_income(sums, p), "quality": quality, "xdiff": xdiff,
                "f1_moving_nonzero_rows": moving_nonzero,
                "moving_rows_cmd_zero": int(np.count_nonzero(cmd_zero & step_mask)) if cat != "stand" else 0,
                "stand_rows_cmd_nonzero": int(np.count_nonzero(~cmd_zero & step_mask)) if cat == "stand" else 0})
    if cat == "stand":
        hold = (ep_json.get("holds") or {}).get(label) or {}
        rig, trip = hold.get("rigidity") or {}, hold.get("tripod") or {}
        fw = np.asarray(z["foot_force_norm"], dtype=np.float64)[widx]
        hz = np.asarray(z["foot_pos_w"], dtype=np.float64)[widx, :, 2] - np.asarray(z["terrain_z"], dtype=np.float64)[widx]
        out["stand"] = {"steps_per_s": rig.get("steps_per_s"), "tripod": trip.get("tripod_score"),
                        "tripod_valid": trip.get("valid"), **_scratch_steps(fw, hz, len(widx) * dt),
                        "upright_frac": float(np.mean(-grav[step_mask, 2] > c["f1_params"]["min_upright_gz"]))}
    # 1-s windows of the own hold (step indices s in step_mask): jitter, switch hz, E1 / F1 rates
    sidx = np.flatnonzero(step_mask)
    wins = []
    acts = RT.clip_actions(z["actions"], p.action_clip)
    for k in range(sidx.size // WIN_STEPS):
        s = sidx[k * WIN_STEPS:(k + 1) * WIN_STEPS]
        if s[-1] - s[0] != WIN_STEPS - 1:
            continue
        loaded = force[s] > RT.MOTION_TOUCH_REF_N              # post rows of these steps
        lifts, downs = M.contact_transitions(loaded)
        wins.append({"jitter": M.action_jitter(acts[s]), "switch_hz": (lifts + downs) / (6 * WIN_STEPS * dt),
                     "e1_per_s": float(e1[s].mean()), "f1_per_s": float(f1[s].mean()),
                     "liftoffs_per_s": lifts / (WIN_STEPS * dt)})
    out["windows"] = wins
    return out


class _SceneD(dict):
    def __init__(self, items, sensors):
        super().__init__(items)
        self.sensors = sensors


# ============================================================================ training-log facts
def tb_tail(train_dir: Path, tags: list[str]) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    fs = sorted(glob.glob(str(train_dir / "events.out.tfevents.*")))
    if not fs:
        raise SystemExit(f"no tensorboard events in {train_dir}")
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {}
    for t in tags:
        if t in have:
            v = [e.value for e in ea.Scalars(t)]
            out[t] = float(np.mean(v[-TB_TAIL:])) if v else None
        else:
            out[t] = None
    return out


def c6x_pool() -> dict:
    txt = CANDIDATES_YAML.read_text()
    m = re.search(r'KRABBY_OMNI_CMD_POOL:\s*"([^"]+)"', txt)
    if not m:
        raise SystemExit(f"no KRABBY_OMNI_CMD_POOL in {CANDIDATES_YAML}")
    pool = {}
    for tok in m.group(1).split(","):
        k, v = tok.split(":")
        pool[POOL_TO_CAT[k.strip()]] = pool.get(POOL_TO_CAT[k.strip()], 0.0) + float(v)
    return {"pool": pool, "spec": m.group(1)}


# ============================================================================ analysis
def _rate(eps: list[dict], key: str) -> float | None:
    d = sum(e["duration_s"] for e in eps)
    return (sum(e["sums"][key] for e in eps) / d) if d > 0 else None


def _med(x):
    x = [v for v in x if v is not None and math.isfinite(v)]
    return float(np.median(x)) if x else None


def auc(a: list[float], b: list[float]) -> float | None:
    """P(a > b) + 0.5 P(a == b) over all pairs."""
    if not a or not b:
        return None
    A, B = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((A > B).sum() + 0.5 * (A == B).sum()) / (A.size * B.size))


def e1_gate1(eps_by_cat: dict) -> dict:
    rows = {}
    for c in RT.CATEGORIES:
        eps = eps_by_cat.get(c) or []
        if not eps:
            rows[c] = {"n": 0, "pass": None}
            continue
        xs, xr, jit, sw = [], [], [], []
        for e in eps:
            n = RT._norm(e, c)
            xs.append(None if n is None else e["sums"]["e1"] / n)
            xr.append(e["sums"]["e1"] / e["duration_s"])
            jit.append(e["quality"].get("action_jitter"))
            sw.append(e["quality"].get("contact_switch_hz"))
        rj, nj = RT.spearman(xs, jit)
        rs, ns = RT.spearman(xs, sw)
        rjr, _ = RT.spearman(xr, jit)
        rsr, _ = RT.spearman(xr, sw)
        best = max([r for r in (rj, rs) if r is not None], default=None)
        rows[c] = {"n": max(nj, ns), "unit": RT.NORM_UNIT[c], "rho_jitter": rj, "rho_switch": rs,
                   "rho_jitter_per_s": rjr, "rho_switch_per_s": rsr, "best": best,
                   "pass": best is not None and best >= RHO_MIN}
    moving_pass = [c for c in MOVING if rows[c].get("pass")]
    verdict = "PASS" if (rows["stand"].get("pass") and len(moving_pass) >= 4) else "FAIL"
    return {"rows": rows, "stand_pass": bool(rows["stand"].get("pass")), "moving_pass": moving_pass,
            "moving_pass_n": len(moving_pass), "verdict": verdict}


def e1_weight(eps_by_cat: dict) -> dict:
    rows = {}
    for c in MOVING:
        eps = [e for e in eps_by_cat.get(c) or [] if e["income"] > 0.0]
        inc_rate = [e["income"] / e["duration_s"] for e in eps]
        qi = RT.best_quartile(inc_rate)
        ratio_q = float(np.median([eps[i]["sums"]["e1"] / eps[i]["income"] for i in qi]))
        ratio_all = float(np.median([e["sums"]["e1"] / e["income"] for e in eps]))
        rows[c] = {"n_quartile": int(len(qi)), "n": len(eps), "ratio_quartile": ratio_q, "ratio_all": ratio_all,
                   "income_per_s_quartile": float(np.median([inc_rate[i] for i in qi])),
                   "w_1pct": E1_SHARE_BAND[0] / ratio_q, "w_5pct": E1_SHARE_BAND[1] / ratio_q}
    lo = max(r["w_1pct"] for r in rows.values())
    hi = min(r["w_5pct"] for r in rows.values())
    feasible = lo <= hi
    w = -RT.sig2(math.sqrt(lo * hi)) if feasible else None
    for r in rows.values():
        r["share_at_w_quartile"] = None if w is None else abs(w) * r["ratio_quartile"]
        r["share_at_w_all"] = None if w is None else abs(w) * r["ratio_all"]
    return {"rows": rows, "band_abs": [lo, hi], "feasible": feasible, "weight": w,
            "binding_low": max(rows, key=lambda k: rows[k]["w_1pct"]),
            "binding_high": min(rows, key=lambda k: rows[k]["w_5pct"])}


def f1_read(by_head: dict, base: float) -> dict:
    stand = {h: [e for e in by_head[h] if e["category"] == "stand" and "sums" in e] for h in by_head}
    per = {}
    for h, eps in stand.items():
        # per second of QUALIFYING stand rows (cmd vx, vy, wz all 0); an episode with none is dropped
        eps = [e for e in eps if e["sums"]["f1_rows_s"] > 0]
        stand[h] = eps
        inc = [e["sums"]["f1"] / e["sums"]["f1_rows_s"] for e in eps]
        st = [e["stand"] for e in eps]
        r_steps, n_steps = RT.spearman(inc, [s["steps_per_s"] for s in st])
        r_lift, _ = RT.spearman(inc, [s["liftoffs_per_s"] for s in st])
        r_swing, _ = RT.spearman(inc, [s["swings_per_s"] for s in st])
        r_clear, n_clear = RT.spearman(inc, [s["clearance_m"] for s in st])
        r_trip, n_trip = RT.spearman(inc, [s["tripod"] for s in st])
        rows_s = sum(e["sums"]["f1_rows_s"] for e in eps)
        own_s = sum(e["duration_s"] for e in eps)
        per[h] = {"n": len(eps), "income_per_s_median": _med(inc),
                  "income_per_s_mean": (sum(e["sums"]["f1"] for e in eps) / rows_s) if rows_s > 0 else None,
                  "qualifying_row_share": (rows_s / own_s) if own_s > 0 else None,
                  "income_per_s_p10": float(np.percentile(inc, 10)) if inc else None,
                  "income_per_s_p90": float(np.percentile(inc, 90)) if inc else None,
                  "steps_per_s_median": _med([s["steps_per_s"] for s in st]),
                  "liftoffs_per_s_median": _med([s["liftoffs_per_s"] for s in st]),
                  "swings_per_s_median": _med([s["swings_per_s"] for s in st]),
                  "clearance_m_median": _med([s["clearance_m"] for s in st]),
                  "tripod_median": _med([s["tripod"] for s in st]),
                  "upright_frac_mean": float(np.mean([s["upright_frac"] for s in st])) if st else None,
                  "rho_steps": r_steps, "n_rho_steps": n_steps, "rho_liftoffs": r_lift, "rho_swings": r_swing,
                  "rho_clearance": r_clear, "n_rho_clearance": n_clear, "rho_tripod": r_trip, "n_rho_tripod": n_trip,
                  "_inc": inc}
    s_inc = per["statue"]["_inc"]
    beats = {}
    for h in STEPPING:
        beats[h] = {"statue_median": per["statue"]["income_per_s_median"], "head_median": per[h]["income_per_s_median"],
                    "beats": per["statue"]["income_per_s_median"] > per[h]["income_per_s_median"],
                    "auc_statue_gt_head": auc(s_inc, per[h]["_inc"])}
    within = {h: {"rho_steps": per[h]["rho_steps"],
                  "pass": per[h]["rho_steps"] is not None and per[h]["rho_steps"] <= -RHO_MIN} for h in STEPPING}
    # moving rows: zero
    moving_nonzero = {h: int(sum(e.get("f1_moving_nonzero_rows", 0) for e in by_head[h])) for h in by_head}
    moving_rows = {h: int(sum(e["n_steps"] for e in by_head[h] if "sums" in e and e["category"] != "stand"))
                   for h in by_head}
    moving_cmd_zero = {h: int(sum(e.get("moving_rows_cmd_zero", 0) for e in by_head[h])) for h in by_head}
    stand_cmd_nonzero = {h: int(sum(e.get("stand_rows_cmd_nonzero", 0) for e in by_head[h])) for h in by_head}
    moving_f1 = {h: float(sum(abs(e["sums"]["f1"]) for e in by_head[h] if "sums" in e and e["category"] != "stand"))
                 for h in by_head}
    zero_ok = all(v == 0 for v in moving_nonzero.values()) and all(v == 0.0 for v in moving_f1.values())
    # weight (B head)
    i_head = per["B"]["income_per_s_mean"]
    i_still = per["statue"]["income_per_s_mean"]
    d = i_still - i_head
    grid = [round(F1_W_RANGE[0] + k * F1_W_GRID, 4) for k in range(int(round((F1_W_RANGE[1] - F1_W_RANGE[0]) / F1_W_GRID)) + 1)]
    margins = {w: base + w * d for w in grid}
    w_star = next((w for w in grid if margins[w] >= F1_MARGIN - 1e-12), None)
    w_exact = (F1_MARGIN - base) / d if d > 0 else None
    alt = {"f1_only_w_exact": (F1_MARGIN / d) if d > 0 else None,
           "ideal_still_w_exact": (F1_MARGIN - base) / (1.0 - i_head) if i_head < 1.0 else None,
           "f1_only_margin_at_1": d}
    statue_beats_all = all(b["beats"] for b in beats.values())
    within_all = all(v["pass"] for v in within.values())
    verdict = "PASS" if (statue_beats_all and within_all and zero_ok and w_star is not None) else "FAIL"
    for h in per:
        per[h].pop("_inc")
    return {"per_head": per, "statue_beats": beats, "statue_beats_all": statue_beats_all, "within": within,
            "within_all_pass": within_all, "zero_on_moving": zero_ok, "moving_nonzero_rows": moving_nonzero,
            "moving_rows_checked": moving_rows, "moving_abs_f1_integral": moving_f1,
            "moving_rows_with_cmd_zero": moving_cmd_zero, "stand_rows_with_cmd_nonzero": stand_cmd_nonzero,
            "base_margin": base, "i_head": i_head, "i_statue": i_still, "delta_i": d, "margins": margins,
            "w_exact": w_exact, "weight": w_star, "alternatives": alt, "verdict": verdict}


def windows_read(by_head: dict) -> dict:
    """Stillest vs jerkiest 1-s windows (by action_jitter) per head x category: E1 / F1 rates."""
    out = {}
    for h, eps in by_head.items():
        for c in RT.CATEGORIES:
            ws = [w for e in eps if e["category"] == c and "windows" in e for w in e["windows"]
                  if w["jitter"] is not None]
            if len(ws) < 20:
                continue
            ws.sort(key=lambda w: w["jitter"])
            k = max(1, len(ws) // 10)
            still, jerk = ws[:k], ws[-k:]
            rj, _ = RT.spearman([w["e1_per_s"] for w in ws], [w["jitter"] for w in ws])
            rs, _ = RT.spearman([w["e1_per_s"] for w in ws], [w["switch_hz"] for w in ws])
            row = {"n_windows": len(ws), "jitter_still": float(np.mean([w["jitter"] for w in still])),
                   "jitter_jerky": float(np.mean([w["jitter"] for w in jerk])),
                   "e1_still": float(np.mean([w["e1_per_s"] for w in still])),
                   "e1_jerky": float(np.mean([w["e1_per_s"] for w in jerk])),
                   "rho_e1_jitter_windows": rj, "rho_e1_switch_windows": rs}
            if c == "stand":
                ws2 = sorted(ws, key=lambda w: w["liftoffs_per_s"])
                still2, busy2 = ws2[:k], ws2[-k:]
                rf, _ = RT.spearman([w["f1_per_s"] for w in ws], [w["liftoffs_per_s"] for w in ws])
                row.update({"f1_still_jitter": float(np.mean([w["f1_per_s"] for w in still])),
                            "f1_jerky_jitter": float(np.mean([w["f1_per_s"] for w in jerk])),
                            "liftoffs_still": float(np.mean([w["liftoffs_per_s"] for w in still2])),
                            "liftoffs_busy": float(np.mean([w["liftoffs_per_s"] for w in busy2])),
                            "f1_fewest_liftoffs": float(np.mean([w["f1_per_s"] for w in still2])),
                            "f1_most_liftoffs": float(np.mean([w["f1_per_s"] for w in busy2])),
                            "rho_f1_liftoffs_windows": rf})
            out.setdefault(h, {})[c] = row
    return out


def episode_reward_basis(by_head: dict, p, tb: dict, pool: dict, e1_w: float | None, f1_w: float | None,
                         f1: dict) -> dict:
    stand_frac = tb.get("Metrics/base_velocity/stand_frac_actual")
    L = tb.get("Train/mean_episode_length")
    if stand_frac is None or L is None:
        raise SystemExit("B training log lacks stand_frac_actual / mean_episode_length")
    tmax_steps = p.episode_length_s / p.dt
    lfac = L / tmax_steps
    pl = dict(pool["pool"])
    rest = sum(v for k, v in pl.items() if k != "stand")
    mix = {k: (stand_frac if k == "stand" else v * (1.0 - stand_frac) / rest) for k, v in pl.items()}
    eps = [e for e in by_head["B"] if "sums" in e]
    by_cat = {c: [e for e in eps if e["category"] == c] for c in RT.CATEGORIES}
    rates = {c: {k: _rate(by_cat[c], k) for k in ("e1", "e1_cam", "e1_rod", "power_all24", "f1", *INCOME_TB)}
             for c in RT.CATEGORIES}

    missing = sorted(c for c in mix if any(v is None for v in rates[c].values()))

    def mixed(key):
        return sum(mix[c] * (rates[c][key] or 0.0) for c in mix)

    weights_inc = {"inc_lin": p.track_lin_w, "inc_ang": p.track_ang_w, "inc_facing": p.facing_w,
                   "inc_yaw_progress": p.yaw_progress_w}
    xcheck = {}
    for k, tbname in INCOME_TB.items():
        rep = weights_inc[k] * mixed(k) * lfac
        logged = tb.get(f"Episode_Reward/{tbname}")
        xcheck[tbname] = {"replay": rep, "logged": logged, "ratio": (rep / logged) if logged else None}
    e1_er = None if e1_w is None else e1_w * mixed("e1") * lfac
    f1_er = None if f1_w is None else f1_w * stand_frac * f1["i_head"] * lfac
    return {"stand_frac_actual": stand_frac, "mean_episode_length_steps": L, "episode_length_s": p.episode_length_s,
            "dt": p.dt, "len_factor": lfac, "mix": mix, "pool_spec": pool["spec"],
            "rates_per_s": rates, "e1_raw_mixed_per_s": mixed("e1"), "e1_episode_reward": e1_er,
            "f1_episode_reward": f1_er, "f1_episode_reward_statue_level": (None if f1_w is None else
                                                                           f1_w * stand_frac * f1["i_statue"] * lfac),
            "income_crosscheck": xcheck, "categories_missing": missing}


def power_split(by_head: dict) -> dict:
    out = {}
    for h, eps in by_head.items():
        for c in RT.CATEGORIES:
            es = [e for e in eps if e["category"] == c and "sums" in e]
            if not es:
                continue
            act = sum(e["sums"]["e1"] for e in es)
            cam = sum(e["sums"]["e1_cam"] for e in es)
            rod = sum(e["sums"]["e1_rod"] for e in es)
            a24 = sum(e["sums"]["power_all24"] for e in es)
            per_ep_cam = [e["sums"]["e1_cam"] / e["sums"]["e1"] for e in es if e["sums"]["e1"] > 0]
            out.setdefault(h, {})[c] = {
                "camshaft_share_of_18": cam / act if act > 0 else None,
                "camshaft_share_median_ep": _med(per_ep_cam),
                "rod_share_of_18": rod / act if act > 0 else None,
                "body_hip_share_of_24": (a24 - act) / a24 if a24 > 0 else None,
                "e1_per_s": _rate(es, "e1"), "all24_per_s": _rate(es, "power_all24")}
    return out


# ============================================================================ rendering
def _f(x, fmt="{:.3f}"):
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def _r(x):
    return "--" if x is None else f"{x:+.2f}"


def _pct(x):
    return "--" if x is None else f"{100 * x:.1f} %"


def render(res: dict) -> str:
    L = []
    A = L.append
    e1, f1, er = res["E1"], res["F1"], res["basis"]
    g1, wt = e1["gate1_B"], e1["weight_read"]
    A("# Replay gate: E1 (actuated power) and F1 (stand stance income)")
    A("")
    A("CHARTER §4 offline replay gate for omni phase 2. CPU only: saved eval npz, with no Isaac, training or GPU.")
    A(f"Generated by `scripts/p2_replay.py` on {res['generated']}. The numbers are also in `replay_gate.json`.")
    A("")
    A("## Verdicts")
    A("")
    A("| term | verdict | weight | Episode_Reward (per max-episode-second) | why |")
    A("|---|---|---|---|---|")
    A(f"| **E1** `KRABBY_POWER_W` + `KRABBY_POWER_JOINTS=actuated` | **{e1['verdict']}** | {_f(e1['weight'], '{:.2g}')} "
      f"| {_f(e1['episode_reward'], '{:.4g}')} | Gate 1 on B: stand {'PASS' if g1['stand_pass'] else 'FAIL'}, moving "
      f"{g1['moving_pass_n']}/6 pass ({', '.join(g1['moving_pass']) or 'none'}); the rule needs stand AND ≥ 4/6 |")
    why_f1 = []
    why_f1.append("statue beats every stepping head" if f1["statue_beats_all"] else "statue does NOT beat every head")
    why_f1.append("within-head ρ(steps/s) ≤ −0.5 on " + ", ".join(h for h in STEPPING if f1["within"][h]["pass"])
                  if f1["within_all_pass"] else "within-head ρ(steps/s) fails on " +
                  ", ".join(h for h in STEPPING if not f1["within"][h]["pass"]))
    why_f1.append("zero on every moving row" if f1["zero_on_moving"] else "NON-ZERO on moving rows")
    why_f1.append(f"weight {f1['weight']}" if f1["weight"] is not None else "no weight in [0.5, 1.0] reaches +0.5/s")
    A(f"| **F1** `KRABBY_STAND_STANCE_W` | **{f1['verdict']}** | {_f(f1['weight'], '{:.2f}')} | "
      f"{_f(f1['episode_reward'], '{:.4g}')} | {'; '.join(why_f1)} |")
    A("")
    A("Headline numbers:")
    A("")
    best = {c: g1["rows"][c].get("best") for c in RT.CATEGORIES}
    A("- **E1, B head, best ρ per category** (rule ≥ +0.5): " + ", ".join(f"{c} {_r(v)}" for c, v in best.items()) +
      ". The camshaft share of the 18-joint sum is "
      f"{_pct(min(r['camshaft_share_of_18'] for r in res['power_split']['B'].values()))}–"
      f"{_pct(max(r['camshaft_share_of_18'] for r in res['power_split']['B'].values()))} on B, so the premise holds: "
      "E1 prices the rods.")
    A("- **F1**: the statue out-earns every stepping head on every episode pair (AUC " +
      ", ".join(f"{h} {_f(f1['statue_beats'][h]['auc_statue_gt_head'], '{:.2f}')}" for h in STEPPING) +
      "). Within a head, the decisive ρ with stand steps/s is " +
      ", ".join(f"{h} {_r(f1['within'][h]['rho_steps'])}" for h in STEPPING) + " (rule ≤ −0.5). B's stand income spans only "
      f"{_f(f1['per_head']['B']['income_per_s_p10'])}–{_f(f1['per_head']['B']['income_per_s_p90'])}/s (p10–p90) across "
      "its episodes.")
    A("")
    if e1["verdict"] == "FAIL" or f1["verdict"] == "FAIL":
        A("**Per CHARTER §4, a FAIL stops and asks the user.** No arm is dropped automatically, and no rule was tuned "
          "to pass. The failing term's weight and magnitude are still recorded, because `p2_live.py --preflight` and "
          "the +200 liveness read need them if you overrule. Note that preflight checks only that the magnitude is "
          "present and non-zero. It does not read `verdict`.")
        A("")
    A("## Inputs")
    A("")
    A("| head | stamp | checkpoint | scored own holds |")
    A("|---|---|---|---|")
    for h, info in res["inputs"].items():
        A(f"| {HEAD_LABEL[h]} | `{info['stamp']}` | `{info['checkpoint']}` | {info['n_scored']} / {info['n_total']} |")
    A("")
    A(f"- C0 stamp = `driver_stamp` of `{C0_RECORD.relative_to(REPO)}`.")
    A(f"- Training params per head are read from its checkpoint's `params/env.yaml` and `agent.yaml` "
      f"(`train_params_from_dumps`). For B: {res['params_B']}.")
    A(f"- Eval motion metrics are recomputed with `gait_eval.metrics.motion_quality_metrics` on the replay window. "
      f"Max |difference| vs each stamp's stored `quality`: {res['xcheck']}.")
    A(f"- Actuated set: {res['joint_sets']['n_actuated']} joints from `_CRAB_ACTUATED_JOINT_NAMES` "
      f"{res['joint_sets']['regexes']}. This equals the eval's `motion_joint_groups` 'actuated' set: "
      f"{res['joint_sets']['equals_metric_actuated']}. Excluded: {', '.join(res['joint_sets']['excluded'])}.")
    A(f"- F1 params as `motion_knobs` builds them: vel_ref {res['f1_params']['vel_ref']} m/s, min_upright_gz "
      f"{res['f1_params']['min_upright_gz']}.")
    A("")
    # ---------------- E1
    A("## E1: actuated-joint mechanical power")
    A("")
    A("### Gate 1 on the B head (the rule)")
    A("")
    A("Spearman ρ, over a category's episodes, of the term's own-hold cost against the step-1 motion metrics. The cost "
      "is normalised as block Q and the precedent normalise it: per m on fwd/side/back/diag, per rad on turn/move_turn, "
      "per s on stand. PASS needs max(ρ jitter, ρ switch_hz) ≥ +0.5, since a cost must rise with the defect. The two "
      "per-s columns are context only.")
    A("")
    A("| category | n | unit | ρ jitter | ρ switch_hz | verdict | ρ jitter (per s) | ρ switch_hz (per s) |")
    A("|---|---|---|---|---|---|---|---|")
    for c in RT.CATEGORIES:
        r = g1["rows"][c]
        A(f"| {c} | {r['n']} | {r.get('unit', '--')} | {_r(r.get('rho_jitter'))} | {_r(r.get('rho_switch'))} | "
          f"{'PASS' if r.get('pass') else '**FAIL**'} | {_r(r.get('rho_jitter_per_s'))} | {_r(r.get('rho_switch_per_s'))} |")
    A("")
    A(f"Result: **{g1['verdict']}**. Stand {'passes' if g1['stand_pass'] else 'fails'}; {g1['moving_pass_n']}/6 moving "
      f"categories pass.")
    A("")
    A("### The same Gate 1 on the other heads (descriptive)")
    A("")
    A("| head | " + " | ".join(RT.CATEGORIES) + " | verdict |")
    A("|---|" + "---|" * (len(RT.CATEGORIES) + 1))
    for h in ("W", "C0"):
        gh = e1["gate1_other"][h]
        cells = [f"{_r(gh['rows'][c].get('rho_jitter'))} / {_r(gh['rows'][c].get('rho_switch'))}" for c in RT.CATEGORIES]
        A(f"| {HEAD_LABEL[h]} | " + " | ".join(cells) + f" | {gh['verdict']} |")
    A("")
    A("Each cell is ρ jitter / ρ switch_hz on the Q-normalised cost.")
    A("")
    A("### Where the power is: camshaft share (premise: about 0)")
    A("")
    A("Pooled over each category's scored episodes, own hold. 'of 18' is the share of the E1 sum. 'Body_Hip of 24' is "
      "the emulator's share of the all-joint sum, which E1 drops.")
    A("")
    A("| head | category | E1 /s (W) | camshaft of 18 | camshaft (median ep) | rods of 18 | Body_Hip of 24 |")
    A("|---|---|---|---|---|---|---|")
    for h in ("B", "W", "C0"):
        for c in RT.CATEGORIES:
            r = res["power_split"].get(h, {}).get(c)
            if r:
                A(f"| {HEAD_LABEL[h]} | {c} | {_f(r['e1_per_s'], '{:.1f}')} | {_pct(r['camshaft_share_of_18'])} | "
                  f"{_pct(r['camshaft_share_median_ep'])} | {_pct(r['rod_share_of_18'])} | {_pct(r['body_hip_share_of_24'])} |")
    A("")
    A("### Weight: 1–5 % of locomotion income on the moving categories (B head)")
    A("")
    A(f"Locomotion income is `replay_motion_terms.task_income`: track_lin_vel_xy_exp ({res['income_weights']['inc_lin']}) "
      f"+ track_ang_vel_z_exp ({res['income_weights']['inc_ang']}) + reward_tracking_yaw/facing "
      f"({res['income_weights']['inc_facing']}) + reward_yaw_progress ({res['income_weights']['inc_yaw_progress']}), "
      "at B's training weights. ratio = the median, over the category's best-tracking quartile by income per second, "
      "of the raw E1 integral divided by the income integral. This is the precedent's Gate-2 ratio. A weight w costs "
      "|w|·ratio of the income.")
    A("")
    A("| category | quartile n / all | income /s (quartile) | ratio (quartile) | ratio (all) | w at 1 % | w at 5 % | share at proposal (quartile) | share (all) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in MOVING:
        r = wt["rows"][c]
        A(f"| {c} | {r['n_quartile']} / {r['n']} | {r['income_per_s_quartile']:.3f} | {r['ratio_quartile']:.4g} | "
          f"{r['ratio_all']:.4g} | {-r['w_1pct']:.3g} | {-r['w_5pct']:.3g} | {_pct(r['share_at_w_quartile'])} | "
          f"{_pct(r['share_at_w_all'])} |")
    A("")
    lo, hi = wt["band_abs"]
    A(f"- Feasible band, where every moving category is in 1–5 %: |w| ∈ [{lo:.3g}, {hi:.3g}]. The low end binds on "
      f"{wt['binding_low']} and the high end on {wt['binding_high']}. Feasible: {wt['feasible']}.")
    A(f"- **Proposed `KRABBY_POWER_W` = {_f(e1['weight'], '{:.2g}')}**, the band's geometric midpoint at 2 s.f.")
    A(f"- Precedent, for scale: the all-24-joint form was proposed at −0.00023, which is 10 % on its binding category "
      "(2026-09-22 REPLAY_GATE.md).")
    A("")
    # ---------------- F1
    A("## F1: stand-slot stance income")
    A("")
    A("Stand rows only: the stand category's own hold with cmd (vx, vy, wz) = 0, which is the emulated `stand_mask`. "
      "Income is per second of qualifying stand rows at weight 1, from `reward_stand_stance_quality`, upright-gated. In W and C0 the facing relay sets wz ≠ 0 on some stand-hold rows; those rows are not stand rows and are left out, as the 'qualifying row share' column shows. Steps/s is the eval's "
      "per-episode `rigidity.steps_per_s`, STAND_STEPPING.md's 'steps/s [S]': a step is an unloaded run of ≥ 5 env "
      "steps of `foot_force_norm` at 50 N. Lift-offs/s, swings/s and clearance use the STAND_STEPPING scratch measure "
      "(`measure_stand.py`): contact means ≥ 1 N, a lift-off is ≥ 2 steps airborne, and a swing clears its own median "
      "stance height by ≥ 1 cm.")
    A("")
    A("| head | n | qualifying row share | income /s median [p10, p90] | income /s mean | steps/s | lift-offs/s | swings/s | clearance m | tripod | upright share |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for h in ("statue", *STEPPING):
        r = f1["per_head"][h]
        A(f"| {HEAD_LABEL[h]} | {r['n']} | {_f(r['qualifying_row_share'])} | {_f(r['income_per_s_median'])} [{_f(r['income_per_s_p10'])}, "
          f"{_f(r['income_per_s_p90'])}] | {_f(r['income_per_s_mean'])} | {_f(r['steps_per_s_median'])} | "
          f"{_f(r['liftoffs_per_s_median'])} | {_f(r['swings_per_s_median'])} | {_f(r['clearance_m_median'])} | "
          f"{_f(r['tripod_median'])} | {_f(r['upright_frac_mean'])} |")
    A("")
    A("### (a) The statue beats every stepping head")
    A("")
    A("| head | statue median /s | head median /s | statue > head | AUC P(statue ep > head ep) |")
    A("|---|---|---|---|---|")
    for h in STEPPING:
        b = f1["statue_beats"][h]
        A(f"| {HEAD_LABEL[h]} | {b['statue_median']:.3f} | {b['head_median']:.3f} | {'yes' if b['beats'] else '**no**'} | "
          f"{_f(b['auc_statue_gt_head'])} |")
    A("")
    A("### (b) Within each head: ρ with stand steps/s ≤ −0.5 (decisive); (c) clearance and tripod (reported only)")
    A("")
    A("| head | n | ρ steps/s (decisive) | verdict | ρ lift-offs/s | ρ swings/s | ρ clearance | ρ tripod |")
    A("|---|---|---|---|---|---|---|---|")
    for h in STEPPING:
        r = f1["per_head"][h]
        A(f"| {HEAD_LABEL[h]} | {r['n_rho_steps']} | {_r(r['rho_steps'])} | {'PASS' if f1['within'][h]['pass'] else '**FAIL**'} | "
          f"{_r(r['rho_liftoffs'])} | {_r(r['rho_swings'])} | {_r(r['rho_clearance'])} | {_r(r['rho_tripod'])} |")
    r = f1["per_head"]["statue"]
    A(f"| {HEAD_LABEL['statue']} | {r['n_rho_steps']} | {_r(r['rho_steps'])} | n/a (still control; median steps/s {_f(r['steps_per_s_median'])}) | {_r(r['rho_liftoffs'])} | "
      f"{_r(r['rho_swings'])} | {_r(r['rho_clearance'])} | {_r(r['rho_tripod'])} |")
    A("")
    A("### (d) Zero on every moving row (asserted)")
    A("")
    A("| head | moving own-hold rows checked | rows with F1 ≠ 0 | Σ abs F1 integral | moving rows with cmd exactly 0 | stand rows with cmd ≠ 0 |")
    A("|---|---|---|---|---|---|")
    for h in ("B", "W", "C0", "statue"):
        A(f"| {HEAD_LABEL[h]} | {f1['moving_rows_checked'][h]} | {f1['moving_nonzero_rows'][h]} | "
          f"{f1['moving_abs_f1_integral'][h]:.3g} | {f1['moving_rows_with_cmd_zero'][h]} | {f1['stand_rows_with_cmd_nonzero'][h]} |")
    A("")
    A(f"Assertion: **{'holds' if f1['zero_on_moving'] else 'FAILS'}**. The training term gates on the command slot "
      "(`stand_mask()`, `slot_id == SLOT_STAND`), not on the command value. 'Moving rows with cmd exactly 0' counts rows "
      "a command-value gate would wrongly pay, such as an in-place turn with its yaw command at zero. The slot gate never "
      "pays them.")
    A("")
    A("### (e) Weight: smallest w in [0.5, 1.0] at which a still stance beats B's stepping by ≥ 0.5/s on stand rows")
    A("")
    A(f"margin(w) = base + w·(I_statue − I_B). base = +{f1['base_margin']:.3f}/s, taken from {res['base_source']}: all "
      f"c6x terms other than F1, ideal still minus head. I_statue = {f1['i_statue']:.4f}/s and I_B = {f1['i_head']:.4f}/s "
      f"(pooled stand means), so Δ = {f1['delta_i']:.4f}/s.")
    A("")
    A("| w | " + " | ".join(f"{w:.2f}" for w in f1["margins"]) + " |")
    A("|---|" + "---|" * len(f1["margins"]))
    A("| margin /s | " + " | ".join(f"{m:+.3f}" for m in f1["margins"].values()) + " |")
    A("")
    al = f1["alternatives"]
    A(f"- **Proposed `KRABBY_STAND_STANCE_W` = {_f(f1['weight'], '{:.2f}')}**. The exact solution is "
      f"w = {_f(f1['w_exact'], '{:.3f}')}, rounded up to the 0.05 grid and clamped to ≥ 0.5.")
    A(f"- Readings not used: F1 alone, without the base, needs w = {_f(al['f1_only_w_exact'], '{:.3f}')}. An ideal still "
      f"stance (income 1.0) instead of the measured statue needs w = {_f(al['ideal_still_w_exact'], '{:.3f}')}.")
    A("")
    # ---------------- windows
    A("## Stillest vs jerkiest 1-s windows (descriptive)")
    A("")
    A("Non-overlapping 1-s windows of each own hold, ranked by window action_jitter (clipped). The table compares the "
      "stillest 10 % with the jerkiest 10 %. ρ is computed over all windows. E1 is raw W, before the weight.")
    A("")
    A("| head | category | windows | jitter still / jerky | E1 /s still / jerky | ρ(E1, jitter) | ρ(E1, switch_hz) |")
    A("|---|---|---|---|---|---|---|")
    for h in ("B", "W", "C0", "statue"):
        for c in RT.CATEGORIES:
            r = res["windows"].get(h, {}).get(c)
            if r:
                A(f"| {HEAD_LABEL[h]} | {c} | {r['n_windows']} | {r['jitter_still']:.3f} / {r['jitter_jerky']:.3f} | "
                  f"{r['e1_still']:.1f} / {r['e1_jerky']:.1f} | {_r(r['rho_e1_jitter_windows'])} | {_r(r['rho_e1_switch_windows'])} |")
    A("")
    A("Stand windows, F1 (income /s at weight 1):")
    A("")
    A("| head | F1 fewest-lift-off 10 % / most 10 % | lift-offs/s there | F1 stillest / jerkiest by jitter | ρ(F1, lift-offs/s) over windows |")
    A("|---|---|---|---|---|")
    for h in ("statue", *STEPPING):
        r = res["windows"].get(h, {}).get("stand")
        if r and "f1_fewest_liftoffs" in r:
            A(f"| {HEAD_LABEL[h]} | {r['f1_fewest_liftoffs']:.3f} / {r['f1_most_liftoffs']:.3f} | "
              f"{r['liftoffs_still']:.2f} / {r['liftoffs_busy']:.2f} | {r['f1_still_jitter']:.3f} / {r['f1_jerky_jitter']:.3f} | "
              f"{_r(r['rho_f1_liftoffs_windows'])} |")
    A("")
    # ---------------- basis
    A("## Episode_Reward basis (CHARTER §8 term liveness, 3× tolerance)")
    A("")
    A(f"rsl_rl's `Episode_Reward/<term>` is the mean episode sum of w·value·dt divided by episode_length_s "
      f"({er['episode_length_s']:.0f} s). The replay estimate is w · Σ_c mix_c · rate_c · L/T_max. "
      f"L/T_max = {er['mean_episode_length_steps']:.0f} / {er['episode_length_s'] / er['dt']:.0f} = {er['len_factor']:.3f}, "
      f"from B's c6x `Train/mean_episode_length`, mean of the last {TB_TAIL} iterations. The mix is the c6x pool "
      f"`{er['pool_spec']}` (omni → diag), with stand set to B's logged `stand_frac_actual` "
      f"{er['stand_frac_actual']:.3f} and the rest rescaled:")
    A("")
    A("| category | " + " | ".join(er["mix"]) + " |")
    A("|---|" + "---|" * len(er["mix"]))
    A("| mix | " + " | ".join(f"{v:.3f}" for v in er["mix"].values()) + " |")
    A("| E1 raw /s (B) | " + " | ".join(f"{er['rates_per_s'][c]['e1']:.1f}" for c in er["mix"]) + " |")
    A("")
    A(f"- **E1**: {_f(e1['weight'], '{:.2g}')} × {er['e1_raw_mixed_per_s']:.1f} W × {er['len_factor']:.3f} = "
      f"**{_f(e1['episode_reward'], '{:.4g}')}**.")
    A(f"- **F1**: {_f(f1['weight'], '{:.2f}')} × {er['stand_frac_actual']:.3f} × {f1['i_head']:.4f} × "
      f"{er['len_factor']:.3f} = **{_f(f1['episode_reward'], '{:.4g}')}**. This is at B's current stand income. At "
      f"statue-level stillness it would be {_f(er['f1_episode_reward_statue_level'], '{:.4g}')}, still inside 3×.")
    A("")
    A("Cross-check of the same conversion on the four income terms (B c6x training log vs replay):")
    A("")
    A("| term | logged Episode_Reward | replay | replay / logged |")
    A("|---|---|---|---|")
    for k, v in er["income_crosscheck"].items():
        A(f"| {k} | {_f(v['logged'], '{:.4f}')} | {_f(v['replay'], '{:.4f}')} | {_f(v['ratio'], '{:.2f}')} |")
    A("")
    A("The conversion reproduces the three large income terms to within 0.84–1.03×. reward_yaw_progress, about 0.4 % "
      "of logged income, is 3.3× high: the suite's 90° turn holds command more yaw than the training pool does. "
      "This matters little for E1, which prices every row, or for F1, which prices only stand rows.")
    A("")
    A("Mismatches that cannot be removed offline:")
    A("")
    A("- The eval rolls out the deterministic mean. Training samples actions (soft σ), which raises state-driven power.")
    A("- Phase 2 trains on the field, with obstacles, pits and ramps; these stamps are flat.")
    A("- The eval uses fixed 12-s holds, not the training slot pool.")
    A("- The stand gate is emulated from the stand hold and a zero command.")
    A("")
    A("The 3× liveness band absorbs these at the order-of-magnitude level only.")
    A("")
    A("## Sources")
    A("")
    for s in res["sources"]:
        A(f"- {s}")
    A("")
    return "\n".join(L)


# ============================================================================ main
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=CAMPAIGN)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="episodes per stamp (smoke only; never for the record)")
    args = ap.parse_args(argv)

    st = stamps()
    regs = actuated_regexes()
    f1p = f1_params_from_source()
    base, base_src = stand_stepping_base()
    M = RT.gait_metrics()
    fe = float(RT.gait_schedule_defaults()["fall_exclusion_s"])
    params, groups, hold_labels, hold_s, metas, xmeta = {}, {}, {}, {}, {}, {}
    joint_sets = None
    for h, d in st.items():
        d = Path(d)
        if not (d / "complete.json").is_file():
            raise SystemExit(f"{d}: no complete.json")
        meta = json.loads((d / "run_meta.json").read_text())
        metas[h] = meta
        env_d, agent_d, src = RT._train_dumps_for(meta, None, None)
        params[h] = RT.train_params_from_dumps(env_d, agent_d, src)
        if abs(float(meta.get("dt", params[h].dt)) - params[h].dt) > 1e-12:
            raise SystemExit(f"{h}: eval dt != training dt")
        jn = list(meta["joint_names"])
        act = [i for i, n in enumerate(jn) if any(re.fullmatch(r, n) for r in regs)]
        mg = M.motion_joint_groups(jn)
        excluded = [jn[i] for i in range(len(jn)) if i not in act]
        assert len(act) == 18, (h, len(act))
        assert sorted(act) == sorted(mg["actuated"]), h
        assert sorted(excluded) == sorted(f"{leg}_Body_Hip_RevoluteJoint" for leg in ("FL", "FR", "ML", "MR", "RL", "RR")), excluded
        assert sorted(jn[i] for i in act) == sorted(meta["action_joint_names"]), h
        groups[h] = {"actuated18": act, "camshaft": [i for i, n in enumerate(jn) if "_Body_CamShaft_" in n],
                     "rod": mg["rod"],
                     "metric_groups": {"rod": mg["rod"], "cam": mg["cam"], "actuated": mg["actuated"],
                                       "camshaft": [i for i, n in enumerate(jn) if "_Body_CamShaft_" in n],
                                       "body_hip": [i for i, n in enumerate(jn) if "_Body_Hip_" in n]}}
        assert len(groups[h]["camshaft"]) == 6 and len(groups[h]["rod"]) == 12
        joint_sets = {"n_actuated": len(act), "regexes": regs, "equals_metric_actuated": True, "excluded": excluded}
        hold_labels[h] = list(meta["hold_labels"])
        hold_s[h] = (meta.get("suite") or {}).get("hold_s") or {}
    if any(abs(params[h].dt - params["B"].dt) > 1e-12 for h in params):
        raise SystemExit("dt differs between stamps")
    fall_trim = int(round(fe / params["B"].dt))
    ctx = {"stamps": {h: str(d) for h, d in st.items()}, "params": params, "groups": groups,
           "hold_labels": hold_labels, "hold_s": hold_s, "fall_trim": fall_trim, "f1_params": f1p}
    jobs = []
    for h in st:
        emap = metas[h].get("env_index_map") or []
        for ent in emap[: (args.limit or len(emap))]:
            if ent.get("category") in RT.CATEGORIES and ent.get("label") in hold_labels[h]:
                jobs.append((h, int(ent["env_index"]), ent))
    with Pool(args.workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        results = pool.map(replay_one, jobs, chunksize=8)
    by_head = {h: [r for r in results if r["head"] == h] for h in st}
    for h in by_head:
        bad = [r for r in by_head[h] if r.get("category") == "stand" and r.get("stand_rows_cmd_nonzero")]
        if bad:
            print(f"note {h}: {len(bad)} stand episodes carry non-zero command rows (the facing relay); those rows "
                  "are not stand rows (CHARTER 4: cmd 0 and wz 0) and are left out of F1's per-second income")
    for h in by_head:
        xd = {}
        for r in by_head[h]:
            for k, v in (r.get("xdiff") or {}).items():
                xd[k] = max(xd.get(k, 0.0), v)
        xmeta[h] = xd

    scored = {h: [e for e in by_head[h] if "sums" in e] for h in by_head}
    eps_by_cat = {h: {c: [e for e in scored[h] if e["category"] == c] for c in RT.CATEGORIES} for h in scored}
    g1_B = e1_gate1(eps_by_cat["B"])
    g1_other = {h: e1_gate1(eps_by_cat[h]) for h in ("W", "C0")}
    wt = e1_weight(eps_by_cat["B"])
    f1 = f1_read(scored, base)
    tb_tags = ["Metrics/base_velocity/stand_frac_actual", "Train/mean_episode_length",
               *[f"Episode_Reward/{t}" for t in INCOME_TB.values()]]
    tb = tb_tail(B_TRAIN_DIR, tb_tags)
    pool_info = c6x_pool()
    basis = episode_reward_basis(scored, params["B"], tb, pool_info, wt["weight"], f1["weight"], f1)
    psplit = power_split(scored)
    wins = windows_read(scored)

    e1_verdict = "PASS" if (g1_B["verdict"] == "PASS" and wt["feasible"]) else "FAIL"
    pB = params["B"]
    res = {
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "inputs": {h: {"stamp": str(Path(st[h]).relative_to(REPO)), "checkpoint": metas[h].get("checkpoint"),
                       "checkpoint_sha256": metas[h].get("checkpoint_sha256"), "n_scored": len(scored[h]),
                       "n_total": len(by_head[h]), "zero_actions": metas[h].get("zero_actions")} for h in st},
        "params_B": pB.source + ("; notes: " + "; ".join(pB.notes) if pB.notes else ""),
        "income_weights": {"inc_lin": pB.track_lin_w, "inc_ang": pB.track_ang_w, "inc_facing": pB.facing_w,
                           "inc_yaw_progress": pB.yaw_progress_w},
        "xcheck": {h: {k: f"{v:.1e}" for k, v in xmeta[h].items()} for h in xmeta},
        "joint_sets": joint_sets, "f1_params": f1p, "base_source": base_src,
        "E1": {"verdict": e1_verdict, "weight": wt["weight"], "episode_reward": basis["e1_episode_reward"],
               "gate1_B": g1_B, "gate1_other": g1_other, "weight_read": wt},
        "F1": {**f1, "episode_reward": basis["f1_episode_reward"]},
        "basis": basis, "power_split": psplit, "windows": wins,
        "sources": [
            f"CHARTER §4 / §8 / §9: `{(CAMPAIGN / 'CHARTER.md').relative_to(REPO)}`",
            f"term code: `{LIB_REWARDS_PY.relative_to(REPO)}` (penalty_mechanical_power, reward_stand_stance_quality), "
            f"`{(FWD_MDP / 'crab_hex_clock_reward.py').relative_to(REPO)}` (stand_stance_income)",
            f"actuated regexes: `{FWD_MDP_CFG_PY.relative_to(REPO)}` (_CRAB_ACTUATED_JOINT_NAMES); F1 params: "
            f"`{MOTION_KNOBS_PY.relative_to(REPO)}`",
            f"shared replay functions: `{TOOL.relative_to(REPO)}` (train_params_from_dumps, income kernels, task_income, "
            "spearman, best_quartile, sig2, _norm; not edited)",
            f"precedent Gate 1 / Gate 2: `{(OMNI / 'experiments/2026-09-22_gait_motion_quality/REPLAY_GATE.md').relative_to(REPO)}`",
            f"stand stepping, base margin and steps definitions: `{STAND_STEPPING_MD.relative_to(REPO)}`",
            f"B c6x training log (stand_frac_actual, mean_episode_length, income Episode_Reward): `{B_TRAIN_DIR.relative_to(REPO)}`",
            f"command pool: `{CANDIDATES_YAML.relative_to(REPO)}` (KRABBY_OMNI_CMD_POOL)",
        ],
    }
    meta = {"inputs": res["inputs"], "generated": res["generated"], "generator": str(HERE.relative_to(REPO)),
            "report": "REPLAY_GATE.md", "eval_metric_crosscheck_max_abs_diff": xmeta,
            "basis_note": "episode_reward = rsl_rl Episode_Reward/<term> basis: mean episode sum(w*value*dt) / "
                          "episode_length_s (per max-episode-second)",
            "basis": {k: v for k, v in basis.items() if k not in ("e1_episode_reward", "f1_episode_reward")}}
    # CHARTER 9 shape: exactly two top-level keys, E1 and F1, each carrying episode_reward
    gate_json = {
        "E1": {"episode_reward": basis["e1_episode_reward"], "verdict": e1_verdict, "weight": wt["weight"],
               "knobs": {"KRABBY_POWER_W": wt["weight"], "KRABBY_POWER_JOINTS": "actuated"},
               "rule": "CHARTER 4: Gate 1 (max(rho jitter, rho switch_hz) >= +0.5 on the Q-normalised cost) on stand "
                       "AND >= 4/6 moving categories, B head; weight 1-5 % of locomotion income on every moving category",
               "gate1_B": g1_B, "gate1_other": g1_other, "weight_read": wt, "power_split": psplit,
               "raw_mixed_per_s": basis["e1_raw_mixed_per_s"], "joint_sets": joint_sets,
               "windows": {h: v for h, v in wins.items()}, "meta": meta},
        "F1": {"episode_reward": basis["f1_episode_reward"], "verdict": f1["verdict"], "weight": f1["weight"],
               "knobs": {"KRABBY_STAND_STANCE_W": f1["weight"]},
               "rule": "CHARTER 4: stand rows (cmd 0, wz 0); statue beats every stepping head; within-head rho(income/s, "
                       "steps/s) <= -0.5 (decisive); zero on every moving row; weight = smallest in [0.5, 1.0] with a "
                       "still-stance margin >= 0.5/s on stand rows",
               **{k: v for k, v in f1.items() if k not in ("verdict", "weight")},
               "f1_params": f1p, "base_source": base_src,
               "windows_stand": {h: v.get("stand") for h, v in wins.items()}, "meta": meta},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "replay_gate.json").write_text(json.dumps(_jsonable(gate_json), indent=1) + "\n")
    (args.out / "REPLAY_GATE.md").write_text(render(res))
    print(json.dumps({"E1": {"verdict": e1_verdict, "weight": wt["weight"], "episode_reward": basis["e1_episode_reward"],
                             "stand_pass": g1_B["stand_pass"], "moving_pass": g1_B["moving_pass"]},
                      "F1": {"verdict": f1["verdict"], "weight": f1["weight"], "episode_reward": basis["f1_episode_reward"],
                             "statue_beats_all": f1["statue_beats_all"],
                             "within": {h: f1["within"][h]["rho_steps"] for h in STEPPING},
                             "zero_on_moving": f1["zero_on_moving"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

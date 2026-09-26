"""The stroke/posture replay tool (stride/posture plan step 6), on CPU, without Isaac.

``experiments/tools/replay_stroke_posture.py`` path-loads the step-2 math
(``mdp/crab_hex_omni_stroke_math.py``) from THIS checkout and replays it over saved eval rollouts.
Every test here builds a small synthetic stamp in ``tmp_path`` -- a kinematic tripod walker whose
feet are exactly planted in stance, so the expected sweep, share, slip and posture numbers are known
in closed form -- and never reads the big eval directories.

Pinned here: the replay conventions the math's review named (heading from ``root_quat_w``, never
``travel_yaw``; row ``s + 1`` state with row ``s`` command, stopping at the first ``done``; stand /
turning / lat_share from the hold's CATEGORY; the sideways head's ``+travel_yaw`` rotation), the
readout definitions, the clock-free economy assertion, the delivered-pay formula, the parity dump
round trip, the paired bootstrap and the gate end to end.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
TOOL_PY = OMNI / "experiments/tools/replay_stroke_posture.py"
MATH_PY = OMNI / "mdp/crab_hex_omni_stroke_math.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t = _load(TOOL_PY, "replay_stroke_posture__test")
SM = t.SM

DT = 0.02
FOOT_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
TRIPOD_A = ("FL", "MR", "RL")
R0 = {"FL": (0.6, 1.1), "FR": (0.6, -1.1), "ML": (0.0, 1.15), "MR": (0.0, -1.15), "RL": (-0.6, 1.1),
      "RR": (-0.6, -1.1)}
JOINTS = ([f"{leg}_Body_{k}_RevoluteJoint" for leg in FOOT_ORDER for k in ("Hip", "CamShaft")]
          + [f"{leg}_Hip_Femur_RevoluteJoint" for leg in FOOT_ORDER]
          + [f"{leg}_Femur_Tibia_RevoluteJoint" for leg in FOOT_ORDER])
LABELS = ["settle", "stand", "fwd", "lat_l", "lat_r", "back", "diag", "turn_l90", "turn_r90", "move_turn",
          "move_turn_r"]
HOLD = {  # label: (category, hand, speed, dir, face, wz)
    "stand": ("stand", None, 0.0, 0.0, 0.0, 0.0),
    "fwd": ("fwd", None, 0.25, 0.0, 0.0, 0.0),
    "lat_l": ("side", "l", 0.22, math.pi / 2, 0.0, 0.0),
    "lat_r": ("side", "r", 0.22, -math.pi / 2, 0.0, 0.0),
    "back": ("back", None, 0.22, math.pi, 0.0, 0.0),
    "diag": ("diag", None, 0.25, math.pi / 4, 0.0, 0.0),
    "turn_l90": ("turn", "l", 0.0, 0.0, math.pi / 2, 0.3),
    "turn_r90": ("turn", "r", 0.0, 0.0, -math.pi / 2, -0.3),
    "move_turn": ("move_turn", "l", 0.22, 0.0, math.pi / 2, 0.3),
    "move_turn_r": ("move_turn", "r", 0.22, 0.0, -math.pi / 2, -0.3),
}
PRO, SETTLE = 10, 20
PERIOD = 60


# ============================================================================ the synthetic walker
def _body_point(root, psi, r0):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([root[0] + c * r0[0] - s * r0[1], root[1] + s * r0[0] + c * r0[1]])


def walker(T: int, *, speed: float, dir_w: float, wz: float, hold_start: int = PRO + SETTLE,
           shift=(0.0, 0.0), gait: bool | None = None, stance_force: float = 300.0, seed: int = 0):
    """A kinematic tripod walker: body at constant world velocity / yaw rate from ``hold_start``;
    stance feet exactly planted at the body point they reach at mid-stance; swing feet unloaded.
    ``shift`` moves every foot's nominal body-frame point (a posture offset)."""
    rng = np.random.default_rng(seed)
    gait = (speed > 0 or wz != 0) if gait is None else gait
    root = np.zeros((T + PERIOD * 2, 3))
    psi = np.zeros(T + PERIOD * 2)
    for k in range(1, root.shape[0]):
        mv = k > hold_start
        psi[k] = psi[k - 1] + (wz * DT if mv else 0.0)
        root[k, :2] = root[k - 1, :2] + (speed * DT * np.array([math.cos(dir_w), math.sin(dir_w)]) if mv else 0.0)
    root[:, 2] = 1.0
    feet = np.zeros((T, 6, 3))
    force = np.zeros((T, 6))
    for f, leg in enumerate(FOOT_ORDER):
        r0 = np.array(R0[leg]) + np.asarray(shift)
        feet[:hold_start + 1, f, :2] = _body_point(root[0], psi[0], r0)
        force[:hold_start + 1, f] = stance_force
        if not gait:
            for k in range(hold_start + 1, T):
                feet[k, f, :2] = feet[hold_start, f, :2]
                force[k, f] = stance_force
            continue
        off = 0 if leg in TRIPOD_A else PERIOD // 2
        starts = [hold_start + off + j * PERIOD for j in range(-1, T // PERIOD + 3)]
        prev_anchor, prev_end = feet[hold_start, f, :2].copy(), hold_start
        for s0 in starts:
            s1 = s0 + PERIOD // 2
            if s1 <= hold_start:
                continue
            a0 = max(s0, hold_start)
            # a stance already under way at the hold start stays where the foot stood
            anchor = (feet[hold_start, f, :2].copy() if s0 <= hold_start
                      else _body_point(root[(a0 + s1) // 2], psi[(a0 + s1) // 2], r0))
            for k in range(prev_end, min(a0, T)):          # swing toward this anchor
                w = (k - prev_end + 1) / max(a0 - prev_end, 1)
                feet[k, f, :2] = (1 - w) * prev_anchor + w * anchor
                feet[k, f, 2] = 0.05
            for k in range(a0, min(s1, T)):
                feet[k, f, :2] = anchor
                force[k, f] = stance_force
            prev_anchor, prev_end = anchor, s1
            if s1 >= T:
                break
    vel = np.zeros_like(feet)
    vel[1:] = (feet[1:] - feet[:-1]) / DT
    quat = np.stack([np.cos(psi[:T] / 2), 0 * psi[:T], 0 * psi[:T], np.sin(psi[:T] / 2)], axis=1)
    mv = np.arange(T) > hold_start
    vw = np.where(mv[:, None], speed * np.array([math.cos(dir_w), math.sin(dir_w)])[None, :], 0.0)
    c, s = np.cos(psi[:T]), np.sin(psi[:T])
    vb = np.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1], 0 * c], axis=1)
    wb = np.stack([0 * c, 0 * c, np.where(mv, wz, 0.0)], axis=1)
    return {"foot_force_norm": force.astype(np.float32), "foot_pos_w": feet.astype(np.float32),
            "foot_lin_vel_w": vel.astype(np.float32), "root_pos_w": root[:T].astype(np.float32),
            "root_quat_w": quat.astype(np.float32), "root_lin_vel_b": vb.astype(np.float32),
            "root_ang_vel_b": wb.astype(np.float32), "psi": psi[:T],
            "actions": rng.normal(0, 0.1, (T, 18)).astype(np.float32),
            "applied_torque": rng.normal(0, 3, (T, 24)).astype(np.float32),
            "joint_pos": rng.normal(0, 0.2, (T, 24)).astype(np.float32)}


ENV_YAML = """decimation: 4
sim: {{dt: 0.005}}
episode_length_s: 40.0
omni_clock_turn_radius: 1.2
rewards:
  track_lin_vel_xy_exp: {{weight: 1.25, params: {{std: 0.2}}}}
  track_ang_vel_z_exp: {{weight: 1.0, params: {{std: 0.5}}}}
  reward_clock_schedule: {{weight: {clock}, params: {{combine: product}}}}
  reward_clock_swing_apex: {{weight: {clock}}}
  reward_yaw_progress: {{weight: 1.0, params: {{clip: null}}}}
  reward_stride_length: {{weight: 0.5}}
  penalty_tibia_deviation_in_stance:
    weight: -0.28
    params: {{contact_force_threshold: 0.1}}
scene:
  robot:
    init_state:
      joint_pos: {{".*_Femur_Tibia_RevoluteJoint": 0.3}}
"""


def _head(root: Path, clock: float = 0.001) -> Path:
    head = root / "head"
    (head / "params").mkdir(parents=True, exist_ok=True)
    (head / "params" / "env.yaml").write_text(ENV_YAML.format(clock=clock))
    (head / "params" / "agent.yaml").write_text("seed: 3\nclip_actions: 1.0\n")
    return head


def omni_stamp(root: Path, *, labels=("stand", "fwd", "lat_l", "lat_r", "turn_l90", "turn_r90"), per: int = 2,
               T: int = 300, shift_side: float = 0.0, done_at: dict | None = None, name: str = "run",
               speed_scale: float = 1.0) -> Path:
    """A synthetic ``suite__omni`` stamp: one wave per hold label, ``per`` envs per wave."""
    run = root / name / "suite__omni/seed001/2026-01-01_00-00-00"
    (run / "raw").mkdir(parents=True)
    (run / "metrics").mkdir()
    head = _head(root)
    emap = []
    idx = 0
    for w, label in enumerate(labels):
        cat, hand, speed, dir_w, face, wz = HOLD[label]
        speed *= speed_scale
        hid = LABELS.index(label)
        for e in range(per):
            shift = (0.0, 0.0)
            if cat in ("side", "diag", "fwd", "back") and shift_side:
                shift = tuple(-shift_side * np.array([math.cos(dir_w), math.sin(dir_w)]))
            z = walker(T, speed=speed, dir_w=dir_w, wz=wz, shift=shift, seed=idx)
            psi = z.pop("psi")
            seg = np.full(T, hid, dtype=np.int16)
            seg[:PRO] = -1
            seg[PRO:PRO + SETTLE] = 0
            face_row = np.where(seg == hid, face, 0.0)
            vw = np.where((np.arange(T) > PRO + SETTLE)[:, None],
                          speed * np.array([math.cos(dir_w), math.sin(dir_w)])[None, :], 0.0)
            c, s = np.cos(psi), np.sin(psi)
            cmd = np.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1],
                            np.where(seg == hid, wz, 0.0)], axis=1)
            done = np.zeros(T, dtype=bool)
            if done_at and idx in done_at:
                done[done_at[idx]] = True
                keep = done_at[idx] + 1
                z = {k: v[:keep] for k, v in z.items()}
                seg, cmd, face_row, psi, done = seg[:keep], cmd[:keep], face_row[:keep], psi[:keep], done[:keep]
            n = seg.shape[0]
            steady = np.zeros(n, dtype=bool)
            steady[PRO + SETTLE + 10:] = True
            z.update({"cmd_applied": cmd.astype(np.float32), "segment_id": seg, "steady_mask": steady,
                      "done": done, "heading_err": np.angle(np.exp(1j * (face_row - psi))).astype(np.float32),
                      "travel_yaw": np.full(n, 0.123, dtype=np.float32), "dt": np.float64(DT)})
            np.savez(run / "raw" / f"episode_{idx:02d}.npz", **z)
            (run / "metrics" / f"episode_{idx:02d}.json").write_text(json.dumps({
                "category": cat, "stride": {"pooled_along_command": {"mean": 0.1 + 0.01 * e}},
                "holds": {label: {"quality": {"contact_switch_hz": 1.0 + 0.1 * ((idx * 7) % 5)}}}}))
            emap.append({"category": cat, "env": e, "env_index": idx, "hand": hand, "label": label, "wave": w})
            idx += 1
    meta = {"checkpoint": str(head / "model_0.pt"), "scenario_id": "suite__omni", "dt": DT, "joint_names": JOINTS,
            "foot_body_names": [f"{leg}_Footpad" for leg in FOOT_ORDER], "hold_labels": LABELS,
            "hold_dirs": [0.0] + [HOLD[x][3] for x in LABELS[1:]],
            "hold_faces": [0.0] + [HOLD[x][4] for x in LABELS[1:]], "env_index_map": emap,
            "episodes_total": idx, "n_steps_per_env": [T] * idx, "travel_yaw": 0.0,
            "suite": {"hold_s": {x: (T - PRO - SETTLE) * DT for x in LABELS[1:]}}}
    (run / "run_meta.json").write_text(json.dumps(meta))
    (run / "complete.json").write_text(json.dumps({"episodes_total": idx}))
    return run


def side_stamp(root: Path, *, n: int = 3, T: int = 300, travel_yaw: float = math.pi / 2, partial: bool = False) -> Path:
    """A synthetic sideways-task stamp: cmd ``(0.25, 0, 0)`` means body +y; the body walks +y."""
    run = root / "side" / "slow__side/seed001/2026-01-01_00-00-00"
    (run / "raw").mkdir(parents=True)
    (run / "metrics").mkdir()
    head = _head(root)
    for i in range(n if not partial else n - 1):
        z = walker(T, speed=0.25, dir_w=math.pi / 2, wz=0.0, seed=i)
        z.pop("psi")
        seg = np.full(T, 1, dtype=np.int16)
        seg[:PRO] = -1
        seg[PRO:PRO + SETTLE] = 0
        cmd = np.zeros((T, 3), dtype=np.float32)
        cmd[PRO + SETTLE + 1:, 0] = 0.25
        steady = np.zeros(T, dtype=bool)
        steady[PRO + SETTLE + 10:] = True
        z.update({"cmd_applied": cmd, "segment_id": seg, "steady_mask": steady, "done": np.zeros(T, dtype=bool),
                  "dt": np.float64(DT)})
        np.savez(run / "raw" / f"episode_{i:02d}.npz", **z)
        (run / "metrics" / f"episode_{i:02d}.json").write_text(json.dumps(
            {"stride": {"pooled_along_command": {"mean": 0.3}}}))
    meta = {"checkpoint": str(head / "model_0.pt"), "scenario_id": "slow__side", "dt": DT, "joint_names": JOINTS,
            "foot_body_names": [f"{leg}_Footpad" for leg in FOOT_ORDER], "hold_labels": ["stand", "creep"],
            "num_envs": n, "n_steps_per_env": [T] * n, "travel_yaw": travel_yaw}
    (run / "run_meta.json").write_text(json.dumps(meta))
    (run / "scenario_metrics.json").write_text("{}")
    return run


def _readout(run):
    st = t.load_stamp(run)
    rp = t.replay(st)
    vals = t.episode_values(st, rp)
    return st, rp, vals, t.readout_table(st, vals)


# ============================================================================ loading and conventions
def test_the_math_is_path_loaded_from_this_checkout_and_no_isaaclab_is_imported():
    assert Path(SM.__file__).resolve() == MATH_PY.resolve()
    code = (f"import importlib.util, sys; s = importlib.util.spec_from_file_location('x', {str(TOOL_PY)!r}); "
            "m = importlib.util.module_from_spec(s); sys.modules['x'] = m; s.loader.exec_module(m); "
            "bad = [k for k in sys.modules if k.split('.')[0] in ('isaaclab', 'isaacsim', 'omni', 'carb')]; "
            "print(bad); assert not bad, bad")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_heading_is_the_quaternion_yaw_never_travel_yaw(tmp_path):
    q = np.array([[math.cos(0.35), 0.0, 0.0, math.sin(0.35)]])
    assert t.heading_from_quat(q)[0] == pytest.approx(0.7)
    run = omni_stamp(tmp_path, labels=("turn_l90",), per=1)
    st = t.load_stamp(run)
    with np.load(st.raw_paths[0]) as z:
        psi = t.heading_from_quat(z["root_quat_w"])
        assert np.allclose(st.psi[0], psi)
        assert not np.allclose(st.psi[0][-50:], z["travel_yaw"][-50:])   # the npz's travel_yaw is ignored
    assert st.guard_max is not None and st.guard_max < 1e-5


def test_the_heading_guard_refuses_an_inconsistent_heading_err(tmp_path):
    run = omni_stamp(tmp_path, labels=("fwd",), per=1)
    path = run / "raw" / "episode_00.npz"
    z = dict(np.load(path))
    z["heading_err"] = z["heading_err"] + 0.01
    np.savez(path, **z)
    with pytest.raises(SystemExit, match="heading guard"):
        t.load_stamp(run)


def test_refuses_partial_stamps_and_sample_actions(tmp_path):
    run = omni_stamp(tmp_path, labels=("fwd",), per=1)
    (run / "complete.json").unlink()
    with pytest.raises(SystemExit, match="complete.json"):
        t.load_stamp(run)
    (run / "complete.json").write_text("{}")
    meta = json.loads((run / "run_meta.json").read_text())
    meta["sample_actions"] = True
    (run / "run_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="sample-actions"):
        t.load_stamp(run)
    with pytest.raises(SystemExit, match="partial sideways"):
        t.load_stamp(side_stamp(tmp_path / "s", partial=True))


def test_slot_flags_come_from_the_hold_category_not_the_command(tmp_path):
    run = omni_stamp(tmp_path, labels=("stand", "fwd", "lat_l", "diag", "back", "turn_l90", "move_turn"), per=1)
    # a stand hold with a noisy nonzero command is still STAND
    p = run / "raw" / "episode_00.npz"
    z = dict(np.load(p))
    z["cmd_applied"][PRO + SETTLE:, 0] = 0.3
    np.savez(p, **z)
    st = t.load_stamp(run)
    own = lambda i: st.own[i]  # noqa: E731
    assert st.stand[0][own(0)].all()
    for i in range(1, 7):
        assert not st.stand[i][own(i)].any()
    assert st.stand[:, :PRO + SETTLE].all()                       # prologue + settle are STAND
    assert not st.turning[:4][st.own[:4]].any() and st.turning[5][own(5)].all() and st.turning[6][own(6)].all()
    lat = [float(np.unique(st.lat_share[i][own(i)])[0]) for i in range(7)]
    assert lat[0] == 0 and lat[1] == 0 and lat[2] == pytest.approx(1.0)
    assert lat[3] == pytest.approx(math.sqrt(0.5)) and lat[4] == pytest.approx(0.0, abs=1e-7)
    assert lat[5] == 0 and lat[6] == 0
    assert st.resampled[0].sum() == 2                             # -1 -> settle -> own hold
    rp = t.replay(st)
    assert rp.pay[0].sum() == 0.0                                 # the stand hold pays nothing


def test_env_step_s_reads_row_s_plus_1_state_with_row_s_command_and_stops_at_done(tmp_path, monkeypatch):
    run = omni_stamp(tmp_path, labels=("fwd",), per=2, T=120, done_at={1: 80})
    st = t.load_stamp(run)
    assert list(st.S) == [119, 80]
    calls = []
    orig = SM.foot_stroke_step

    def spy(force, foot_pos_w, root_pos_w, heading, cmd, stand, *a, **kw):
        calls.append((force.clone(), heading.clone(), cmd.clone(), stand.clone()))
        return orig(force, foot_pos_w, root_pos_w, heading, cmd, stand, *a, **kw)

    monkeypatch.setattr(SM, "foot_stroke_step", spy)
    rp = t.replay(st)
    assert len(calls) == 119
    for s in (0, 40, 79, 100):
        f, h, c, _ = calls[s]
        assert torch.equal(f[0], torch.as_tensor(st.arr["foot_force_norm"][0, s + 1]))
        assert torch.equal(c[0], torch.as_tensor(st.arr["cmd_applied"][0, s]))
        assert h[0].item() == pytest.approx(st.psi[0, s + 1], abs=1e-6)
    assert not rp.gate[1, 80:].any() and rp.pay[1, 80:].sum() == 0.0   # nothing after the first done


# ============================================================================ readout definitions
def test_a_planted_tripod_walker_reads_its_closed_form_numbers(tmp_path):
    run = omni_stamp(tmp_path, labels=("stand", "fwd", "lat_l"), per=2, T=400)
    st, rp, vals, table = _readout(run)
    for cat, v in (("fwd", 0.25), ("side", 0.22)):
        row = table[cat]
        # a stance lasts PERIOD/2 steps; the first loaded step is not credited (prev reading)
        assert row["stance_stroke"] == pytest.approx(v * (PERIOD // 2 - 1) * DT, rel=0.08)
        assert row["stroke_share"] == pytest.approx(1.0, abs=0.02)
        assert row["slip_ratio"] == pytest.approx(0.0, abs=1e-3)
        assert row["body_offset"] == pytest.approx(0.0, abs=0.02)
        assert row["stroke_per_s"] > 0 and row["lead_scuff_frac"] == pytest.approx(0.0, abs=1e-6)
        assert row["stride_along"] == pytest.approx(0.105)           # MEAN of the per-episode json means
    assert table["stand"]["stroke_per_s"] == 0.0


def test_body_offset_and_lead_trail_rows_follow_the_command_not_the_leg_name(tmp_path):
    run = omni_stamp(tmp_path, labels=("lat_l", "lat_r"), per=2, T=400, shift_side=0.2)
    _st, _rp, _vals, table = _readout(run)
    for row in ("side/l", "side/r", "side"):
        v = table[row]
        assert v["body_offset"] == pytest.approx(0.2, abs=0.03)       # feet centroid 0.2 m behind the body
        # leading row at |y| ~ 1.1 - 0.2 (the middle legs 1.15 - 0.2), trailing ~ 1.1 + 0.2
        assert 0.85 <= v["lead_y"] <= 0.97 and 1.28 <= v["trail_y"] <= 1.37
        assert v["lead_trail_asym"] == pytest.approx(v["trail_y"] - v["lead_y"])


def test_the_sideways_head_rotates_the_command_by_travel_yaw(tmp_path):
    run = side_stamp(tmp_path)
    st = t.load_stamp(run)
    assert st.kind == "side" and {e["category"] for e in st.episodes} == {"side"}
    assert np.allclose(st.arr["cmd_applied"][0, -1], [0.0, 0.25, 0.0], atol=1e-6)
    assert (st.lat_share == 1.0).all() and not st.stand.any() and not st.turning.any()
    _st, _rp, _vals, table = _readout(run)
    assert table["side"]["stance_stroke"] == pytest.approx(0.25 * (PERIOD // 2 - 1) * DT, rel=0.08)
    assert table["side"]["centring_per_s"] > 0.85
    with pytest.raises(SystemExit, match="travel_yaw"):
        t.load_stamp(side_stamp(tmp_path / "bad", travel_yaw=0.0))


def test_turn_in_place_pays_the_stroke_term_through_its_turning_flag(tmp_path):
    run = omni_stamp(tmp_path, labels=("turn_l90", "turn_r90"), per=1, T=400)
    _st, _rp, _vals, table = _readout(run)
    for row in ("turn/l", "turn/r"):
        assert table[row]["stroke_per_s"] > 0.05
        assert table[row]["stroke_share"] == pytest.approx(1.0, abs=0.05)
        assert table[row]["centring_per_s"] == 0.0


def test_duplicated_contact_readings_leave_a_planted_gait_nearly_unchanged(tmp_path):
    run = omni_stamp(tmp_path, labels=("fwd",), per=2, T=400)
    st = t.load_stamp(run)
    base = t.replay(st)
    dup = t.replay(st, dup=t.dup_pattern(st.arr["segment_id"].shape[1]))
    # a repeated reading shifts an edge by at most one step and never creates or removes a stance;
    # a stance kept open one step into this walker's fast swing (3x body speed) loses that step's sweep
    assert dup.stances["env"].size == base.stances["env"].size
    assert dup.stances["payable"].sum() == base.stances["payable"].sum()
    step = 3 * 0.25 * DT
    assert np.abs(dup.stances["B"] - base.stances["B"]).max() <= step + 1e-6
    a, b = base.pay.sum(), dup.pay.sum()
    assert b == pytest.approx(a, rel=0.2)
    assert t.dup_pattern(10).tolist() == [False] * 4 + [True] + [False] * 4 + [True]
    f32 = t.f32_stale_pattern(2000, physics_dt=0.005, decimation=4, update_period=0.02)
    assert not f32[:500].any() and f32[800:].any()                  # stale refreshes only after ~16 s


def test_patched_math_restores_the_module_constants():
    before = (SM.FORCE_HOLD_N, SM.N_OFF)
    with t.patched_math(t.StrokeParams(force_hold=15.0, n_off=3)):
        assert (SM.FORCE_HOLD_N, SM.N_OFF) == (15.0, 3)
    assert (SM.FORCE_HOLD_N, SM.N_OFF) == before
    assert t.parse_params(["rho0=0.4", "n_off=3"]).n_off == 3
    with pytest.raises(SystemExit, match="unknown parameter"):
        t.parse_params(["nope=1"])


# ============================================================================ economy / income
def test_the_economy_is_clock_free_and_asserted_against_the_dump(tmp_path):
    head = _head(tmp_path)
    econ, info = t.load_economy(head / "params" / "env.yaml")
    assert tuple(econ.clock) == (0.001, 0.001) and info["dump_clock"] == [0.001, 0.001]
    assert dict(econ.extra)["tibia"] == -0.28
    held = _head(tmp_path / "held", clock=1.0)
    with pytest.raises(SystemExit, match="clock"):
        t.load_economy(held / "params" / "env.yaml")


def test_delivered_pay_is_the_managers_post_clip_share():
    x = np.array([1.0, 1.0, 1.0, 0.0])
    ctx = np.array([0.5, -0.4, -2.0, -1.0])
    assert t.delivered(x, ctx).tolist() == pytest.approx([1.0, 0.6, 0.0, 0.0])


def test_tibia_deviation_prices_only_loaded_knees(tmp_path):
    head = _head(tmp_path)
    _econ, info = t.load_economy(head / "params" / "env.yaml")
    q = np.zeros((2, 24))
    q[:, 18:] = 0.5                                              # every knee 0.2 off its 0.3 default
    force = np.array([[100.0] * 6, [100.0, 0, 0, 0, 0, 0]])
    out = t.tibia_deviation({"joint_pos": q, "foot_force_norm": force}, JOINTS, info)
    assert out.tolist() == pytest.approx([6 * 0.04, 0.04])


def test_income_rows_absorb_and_bind_as_the_context_says(tmp_path):
    run = omni_stamp(tmp_path, labels=("fwd",), per=2, T=300)
    st = t.load_stamp(run)
    rp = t.replay(st)
    econ, _info = t.load_economy(Path(st.meta["checkpoint"]).parent / "params" / "env.yaml")
    big = np.full((st.N, st.arr["segment_id"].shape[1]), 100.0)
    rows = t.income_table(st, rp, w_stroke=0.1, w_centring=0.0, econ=econ, ctx=big)
    assert rows["fwd"]["stroke_absorbed"] == pytest.approx(0.0)
    assert rows["fwd"]["clip_bind_base"] == 0.0
    rows = t.income_table(st, rp, w_stroke=0.1, w_centring=0.0, econ=econ, ctx=-big)
    assert rows["fwd"]["stroke_absorbed"] == pytest.approx(1.0)
    assert rows["fwd"]["clip_bind_with"] == 1.0
    s = rows["fwd"]["stances"]
    # each episode's first stance began loaded at reset (prologue) -> reset-tainted, never paid; every
    # other stance is exactly planted, so the gates trim nothing else
    assert s["rej_taint"] == 3 * 2 and s["scored"] > 0
    assert s["lost_anchor"] == s["lost_slip"] == s["lost_prog"] == pytest.approx(0.0, abs=1e-6)
    assert s["kept"] * s["ungated_credit"] + s["lost_n_taint"] == pytest.approx(s["ungated_credit"], rel=1e-5)


def test_income_cli_runs_on_a_synthetic_stamp(tmp_path, capsys):
    run = omni_stamp(tmp_path, labels=("stand", "fwd", "turn_l90"), per=1, T=250)
    env_yaml = Path(json.loads((run / "run_meta.json").read_text())["checkpoint"]).parent / "params" / "env.yaml"
    out = tmp_path / "income.json"
    assert t.main(["income", str(run), "--w-stroke", "0.1", "--w-centring", "0.2", "--econ-env-yaml",
                   str(env_yaml), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "stances per gate" in text and "tibia" in text
    rows = json.loads(out.read_text())["rows"]
    assert rows["stand"]["stroke_nominal_per_s"] == 0.0 and rows["turn"]["stroke_nominal_per_s"] > 0


# ============================================================================ bootstrap / validate
def test_paired_bootstrap_of_a_run_against_itself_is_zero_with_a_finite_mde(tmp_path):
    run = omni_stamp(tmp_path, labels=("fwd", "lat_l", "lat_r"), per=5, T=260)
    st, _rp, vals, _table = _readout(run)
    boot = t.paired_bootstrap(st, vals, st, vals, B=200)
    for row in ("fwd", "side", "side/l"):
        d = boot[row]["stance_stroke"]
        assert d["delta"] == 0.0 and d["ci"] == [0.0, 0.0] and d["mde"] == 0.0
    shifted = {k: v + (0.01 if k == "stance_stroke" else 0.0) for k, v in vals.items()}
    boot = t.paired_bootstrap(st, shifted, st, vals, B=200)
    assert boot["side"]["stance_stroke"]["delta"] == pytest.approx(0.01)


def test_validate_names_every_missed_target_and_exits_nonzero(tmp_path, capsys):
    run = omni_stamp(tmp_path, labels=("fwd", "lat_l", "lat_r"), per=1, T=260)
    head_yaml = Path(json.loads((run / "run_meta.json").read_text())["checkpoint"]).parent / "params" / "env.yaml"
    assert t.main(["readout", str(run), "--validate", "--econ-env-yaml", str(head_yaml)]) == 1
    out = capsys.readouterr().out
    assert "FAIL side.body_offset" in out and "FAIL side.lead_y" in out and "[validate] FAILED" in out
    side = side_stamp(tmp_path / "s")
    assert t.main(["readout", str(side), "--validate", "--econ-env-yaml", str(head_yaml)]) == 1   # |y| 1.1, not 0.93


def test_readout_vs_prints_the_paired_delta_table(tmp_path, capsys):
    a = omni_stamp(tmp_path, labels=("fwd", "lat_l"), per=3, T=260, name="a")
    b = omni_stamp(tmp_path, labels=("fwd", "lat_l"), per=3, T=260, name="b", speed_scale=0.8)
    head_yaml = Path(json.loads((a / "run_meta.json").read_text())["checkpoint"]).parent / "params" / "env.yaml"
    assert t.main(["readout", str(a), "--vs", str(b), "--econ-env-yaml", str(head_yaml)]) == 0
    assert "delta (run - vs)" in capsys.readouterr().out


# ============================================================================ parity
def _make_dump(tmp_path, *, perturb: float = 0.0) -> Path:
    run = omni_stamp(tmp_path, labels=("fwd", "lat_l", "turn_l90"), per=1, T=200)
    st = t.load_stamp(run)
    S = int(st.S.min())
    a = st.arr
    N = st.N
    stt, cst = SM.init_stroke_state(N, 6), SM.init_centring_state(N)
    reset = np.zeros((S, N), dtype=bool)
    reset[120, 1] = True
    rows = {k: [] for k in t.PARITY_KEYS + ("reset", "pay", "o")}
    for s in range(S):
        if reset[s].any():
            ids = torch.as_tensor(np.flatnonzero(reset[s]))
            SM.reset_stroke_state(stt, ids)
            SM.reset_centring_state(cst, ids)
        inp = {"force": a["foot_force_norm"][:, s + 1], "foot_pos_w": a["foot_pos_w"][:, s + 1],
               "root_pos_w": a["root_pos_w"][:, s + 1], "heading": st.psi[:, s + 1].astype(np.float32),
               "cmd": a["cmd_applied"][:, s], "stand": st.stand[:, s],
               "root_lin_vel_b_xy": a["root_lin_vel_b"][:, s + 1, :2],
               "root_ang_vel_b_z": a["root_ang_vel_b"][:, s + 1, 2], "turning": st.turning[:, s],
               "lat_share": st.lat_share[:, s].astype(np.float32), "resampled": st.resampled[:, s]}
        tt = {k: torch.as_tensor(np.ascontiguousarray(v)) for k, v in inp.items()}
        R, stt, pay = SM.foot_stroke_step(tt["force"], tt["foot_pos_w"], tt["root_pos_w"], tt["heading"], tt["cmd"],
                                          tt["stand"], tt["root_lin_vel_b_xy"], tt["root_ang_vel_b_z"], stt, DT,
                                          1.2, tt["turning"])
        b, cst, o = SM.body_centring_step(tt["foot_pos_w"], tt["root_pos_w"], tt["heading"], tt["cmd"], tt["stand"],
                                          tt["root_lin_vel_b_xy"], cst, DT, tt["lat_share"], tt["resampled"])
        for k, v in inp.items():
            rows[k].append(v)
        rows["reward"].append(R.numpy() + (perturb if s == 150 else 0.0))
        rows["b"].append(b.numpy())
        rows["reset"].append(reset[s])
        rows["pay"].append(pay.numpy())
        rows["o"].append(o.numpy())
    dump = {k: np.stack(v) for k, v in rows.items()}
    dump.update(dt=np.float64(DT), turn_radius=np.float64(1.2))
    path = tmp_path / f"dump_{perturb}.npz"
    np.savez(path, **dump)
    return path


def test_parity_recomputes_a_wrapper_dump_exactly_and_catches_a_drift(tmp_path, capsys):
    good = _make_dump(tmp_path / "g")
    r = t.parity_check(dict(np.load(good)))
    assert r["ok"] and set(r["checked"]) == {"pay_sum", "b", "pay", "o"} and r["max_abs"]["pay_sum"] == 0.0
    assert t.main(["parity", str(good)]) == 0
    bad = _make_dump(tmp_path / "b", perturb=0.01 / DT)
    assert t.main(["parity", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out
    d = dict(np.load(good))
    d.pop("b")
    with pytest.raises(SystemExit, match="lacks"):
        t.parity_check(d)


# ============================================================================ the gate
def test_the_gate_runs_end_to_end_and_never_hides_a_fail(tmp_path):
    labels = ("stand", "fwd", "lat_l", "lat_r", "back", "diag", "turn_l90", "turn_r90", "move_turn", "move_turn_r")
    healthy = omni_stamp(tmp_path, labels=labels, per=2, T=260, name="healthy")
    posture = omni_stamp(tmp_path, labels=labels, per=2, T=260, name="posture", shift_side=0.2)
    degen = omni_stamp(tmp_path, labels=labels, per=2, T=260, name="degen", speed_scale=0.3)
    side = side_stamp(tmp_path / "sw", n=2, T=260)
    cfg = {"primary": "H", "healthy": {"H": str(healthy)}, "sideways": {"SW": str(side)},
           "posture": {"Z": str(posture)}, "degenerate": {"D": str(degen)}, "x08": "H", "statue": "D",
           "creeper": "D", "grid": {"rho0": [0.1]}, "dropped": {"v4": "no npz"}}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "gate"
    assert t.main(["gate", "--config", str(cfg_path), "--out", str(out)]) == 0
    rep = json.loads((out / "replay_gate.json").read_text())
    md = (out / "REPLAY_GATE.md").read_text()
    ids = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "P1", "P2", "P3", "P4")
    for k in ids:
        assert k in rep["criteria"] and f"| {k} |" in md
        verdict = t._verdict(rep["criteria"][k]["pass"])
        assert f"| {k} | **{verdict}** |" in md
    assert rep["w_S"] > 0 and "v4: no npz" in md
    # the planted walker keeps every stance: R1 passes; its perfectly-planted side keeps ~1.0 > 0.40: R2 FAILS
    assert rep["criteria"]["R1"]["pass"] is True
    assert rep["criteria"]["R2"]["pass"] is False and "| R2 | **FAIL** |" in md
    assert rep["criteria"]["R4"]["shares_at_w_S"]["H"]["fwd"] <= 0.10 + 1e-9

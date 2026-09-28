"""The wrapper -> replay parity recorder (stride/posture plan, step 12(ii), 2026-09-26), without Isaac.

``crab_hex_omni_stroke_parity.py`` is the shared recorder ``RewardFootStroke`` and
``RewardBodyCentring`` (``crab_hex_omni_stroke_rewards.py``) write into when
``KRABBY_OMNI_STROKE_PARITY_DUMP`` is set, keyed by env step so a row is only written once both
terms have recorded it. Covered here:

* unset / blank is a complete no-op: no file, no stored state, every call returns immediately;
* the singleton reads the environment once, at its first construction, and is cached after that;
* a round trip -- drive the math directly with synthetic inputs, feed the SAME inputs/outputs through
  the recorder, save, and confirm the tool's own ``parity`` check (``replay_stroke_posture.py``)
  reproduces every recorded step to well under its tolerance;
* a deliberately corrupted dump fails that same check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
PARITY_PY = OMNI / "mdp/crab_hex_omni_stroke_parity.py"
MATH_PY = OMNI / "mdp/crab_hex_omni_stroke_math.py"
TOOL_PY = OMNI / "experiments/tools/replay_stroke_posture.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load(PARITY_PY, "crab_hex_omni_stroke_parity__test")
SM = _load(MATH_PY, "crab_hex_omni_stroke_parity__test_math")
TOOL = _load(TOOL_PY, "crab_hex_omni_stroke_parity__test_tool")

DT = 0.02
TURN_R = 1.2
NUM_FEET = 6


# =============================================================================== no-op when unset
def test_disabled_recorder_is_a_complete_noop(tmp_path):
    rec = P.ParityRecorder(None, 16, 300)
    assert not rec.enabled
    rec.note_reset([0, 1])
    rec.record_stroke(
        0, force=None, foot_pos_w=None, root_pos_w=None, heading=None, cmd=None, stand=None,
        root_lin_vel_b_xy=None, root_ang_vel_b_z=None, turning=None, resampled=None, reward=None,
        pay=None, dt=DT, turn_radius=TURN_R,
    )
    rec.record_centring(
        0, foot_pos_w=None, root_pos_w=None, heading=None, cmd=None, stand=None, lat_share=None,
        resampled=None, b=None, o=None, dt=DT,
    )
    rec.close()
    assert rec._store == {}
    assert not (tmp_path / "would_never_exist.npz").exists()


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_or_unset_path_disables_the_recorder(blank):
    assert not P.ParityRecorder(blank, 16, 300).enabled


def test_get_recorder_reads_the_environment_once_and_caches(monkeypatch, tmp_path):
    monkeypatch.delenv(P.DUMP_PATH_VAR, raising=False)
    monkeypatch.delenv(P.ENVS_VAR, raising=False)
    monkeypatch.delenv(P.STEPS_VAR, raising=False)
    P._reset_singleton_for_tests()
    assert not P.get_recorder().enabled

    P._reset_singleton_for_tests()
    dump = tmp_path / "d.npz"
    monkeypatch.setenv(P.DUMP_PATH_VAR, str(dump))
    rec = P.get_recorder()
    assert rec.enabled and rec.k == P.DEFAULT_ENVS and rec.m == P.DEFAULT_STEPS
    assert P.get_recorder() is rec  # cached: a second call does not re-read the environment

    monkeypatch.setenv(P.ENVS_VAR, "3")  # changing the env now must not reach the cached instance
    assert P.get_recorder().k == P.DEFAULT_ENVS
    P._reset_singleton_for_tests()


def test_unset_dump_never_parses_the_size_vars(monkeypatch):
    """Unset DUMP is a complete no-op: garbage in ENVS/STEPS must not raise in a wrapper's __init__."""
    monkeypatch.delenv(P.DUMP_PATH_VAR, raising=False)
    monkeypatch.setenv(P.ENVS_VAR, "not-an-int")
    monkeypatch.setenv(P.STEPS_VAR, "x")
    P._reset_singleton_for_tests()
    assert not P.get_recorder().enabled
    P._reset_singleton_for_tests()


# =============================================================================== synthetic driver
def _synthetic_steps(num_envs: int, num_steps: int, reset_at: dict[int, list[int]]):
    """A small deterministic scenario: every env walks forward at 0.25 m/s; foot 0 plants (fixed in
    world) for 14 steps of every 20 and swings (rides with the body) for the other 6; the other five
    feet always ride with the body. ``reset_at`` maps a step index to the env ids reset before it."""
    foot_rel = torch.zeros(num_envs, NUM_FEET, 2)
    offsets = [(0.5, 0.7), (0.5, -0.7), (0.0, 0.75), (0.0, -0.75), (-0.5, 0.7), (-0.5, -0.7)]
    for f, off in enumerate(offsets):
        foot_rel[:, f] = torch.tensor(off)
    cmd = torch.zeros(num_envs, 3)
    cmd[:, 0] = 0.25
    root = torch.zeros(num_envs, 3)
    touchdown = torch.zeros(num_envs, 2)
    was_loaded = torch.zeros(num_envs, dtype=torch.bool)
    steps = []
    for s in range(num_steps):
        root[:, :2] += cmd[:, :2] * DT
        loaded_now = torch.full((num_envs,), (s % 20) < 14)
        newly_loaded = loaded_now & ~was_loaded
        touchdown[newly_loaded] = (root[:, :2] + foot_rel[:, 0])[newly_loaded]
        was_loaded = loaded_now

        foot_pos_w = torch.zeros(num_envs, NUM_FEET, 3)
        foot_pos_w[:, 1:, :2] = root[:, None, :2] + foot_rel[:, 1:]
        foot_pos_w[:, 0, :2] = torch.where(loaded_now[:, None], touchdown, root[:, :2] + foot_rel[:, 0])
        force = torch.zeros(num_envs, NUM_FEET)
        force[:, 0] = torch.where(loaded_now, torch.full((num_envs,), 300.0), torch.zeros(num_envs))

        steps.append(dict(
            reset_ids=reset_at.get(s),
            force=force, foot_pos_w=foot_pos_w, root_pos_w=root.clone(),
            heading=torch.zeros(num_envs), cmd=cmd.clone(), stand=torch.zeros(num_envs, dtype=torch.bool),
            root_lin_vel_b_xy=cmd[:, :2].clone(), root_ang_vel_b_z=torch.zeros(num_envs),
            turning=torch.zeros(num_envs, dtype=torch.bool), resampled=torch.zeros(num_envs, dtype=torch.bool),
            lat_share=torch.full((num_envs,), 0.3),
        ))
    return steps


def _drive_and_record(rec, num_envs: int, num_steps: int, reset_at: dict[int, list[int]]):
    """Drive the math directly, exactly as the wrappers would, through ``rec``. Returns the raw
    (reward, pay, b, o) trajectories for a sanity check independent of the saved dump."""
    stt = SM.init_stroke_state(num_envs, NUM_FEET)
    cst = SM.init_centring_state(num_envs)
    rewards, pays, bs, os_ = [], [], [], []
    for s, inp in enumerate(_synthetic_steps(num_envs, num_steps, reset_at)):
        if inp["reset_ids"]:
            ids = torch.as_tensor(inp["reset_ids"])
            SM.reset_stroke_state(stt, ids)
            SM.reset_centring_state(cst, ids)
            rec.note_reset(ids)
        R, stt, pay = SM.foot_stroke_step(
            inp["force"], inp["foot_pos_w"], inp["root_pos_w"], inp["heading"], inp["cmd"], inp["stand"],
            inp["root_lin_vel_b_xy"], inp["root_ang_vel_b_z"], stt, DT, TURN_R, inp["turning"], inp["resampled"])
        rec.record_stroke(
            s, force=inp["force"], foot_pos_w=inp["foot_pos_w"], root_pos_w=inp["root_pos_w"],
            heading=inp["heading"], cmd=inp["cmd"], stand=inp["stand"],
            root_lin_vel_b_xy=inp["root_lin_vel_b_xy"], root_ang_vel_b_z=inp["root_ang_vel_b_z"],
            turning=inp["turning"], resampled=inp["resampled"], reward=R, pay=pay, dt=DT, turn_radius=TURN_R)
        b, cst, o = SM.body_centring_step(
            inp["foot_pos_w"], inp["root_pos_w"], inp["heading"], inp["cmd"], inp["stand"],
            inp["root_lin_vel_b_xy"], cst, DT, inp["lat_share"], inp["resampled"])
        rec.record_centring(
            s, foot_pos_w=inp["foot_pos_w"], root_pos_w=inp["root_pos_w"], heading=inp["heading"],
            cmd=inp["cmd"], stand=inp["stand"], lat_share=inp["lat_share"], resampled=inp["resampled"],
            b=b, o=o, dt=DT)
        rewards.append(R)
        pays.append(pay)
        bs.append(b)
        os_.append(o)
    return torch.stack(rewards), torch.stack(pays), torch.stack(bs), torch.stack(os_)


def _build_dump(tmp_path, *, num_envs=5, num_steps=40, reset_at=None) -> Path:
    reset_at = reset_at if reset_at is not None else {12: [1]}
    dump_path = tmp_path / "dump.npz"
    rec = P.ParityRecorder(str(dump_path), num_envs, num_steps)
    rewards, pays, bs, os_ = _drive_and_record(rec, num_envs, num_steps, reset_at)
    rec.close()
    assert dump_path.exists(), "the recorder must save on close() even without hitting the auto-save condition"
    assert pays.abs().sum() > 0, "the synthetic gait must exercise at least one non-zero stance pay"
    assert bs.abs().sum() > 0, "the synthetic gait must exercise at least one non-zero centring bonus"
    return dump_path


# =============================================================================== round trip
def test_round_trip_reproduces_exactly_and_passes_the_tools_parity_check(tmp_path):
    dump_path = _build_dump(tmp_path)
    with np.load(dump_path) as z:
        dump = {k: z[k] for k in z.files}
    for key in TOOL.PARITY_KEYS:
        assert key in dump, f"the saved dump is missing {key!r}, which the tool's parity check requires"
    assert dump["reward"].shape == dump["pay"].shape[:-1]
    assert dump["pay"].shape[-1] == NUM_FEET

    result = TOOL.parity_check(dump)
    assert result["ok"], result
    assert set(result["checked"]) == {"pay_sum", "b", "pay", "o"}
    for key in result["checked"]:
        assert result["max_abs"][key] == pytest.approx(0.0, abs=1e-8)

    assert TOOL.main(["parity", str(dump_path)]) == 0


def test_a_step_without_both_terms_is_never_saved_as_a_complete_row(tmp_path):
    """If a step were only ever recorded by one term, the tool would recompute a row from a fully
    zeroed partner term. Steps 0..k-1 here get only ``record_stroke`` and are held back."""
    num_envs, num_steps = 3, 10
    dump_path = tmp_path / "dump.npz"
    rec = P.ParityRecorder(str(dump_path), num_envs, num_steps)
    for s, inp in enumerate(_synthetic_steps(num_envs, num_steps, {})):
        stt = SM.init_stroke_state(num_envs, NUM_FEET)
        R, _stt, pay = SM.foot_stroke_step(
            inp["force"], inp["foot_pos_w"], inp["root_pos_w"], inp["heading"], inp["cmd"], inp["stand"],
            inp["root_lin_vel_b_xy"], inp["root_ang_vel_b_z"], stt, DT, TURN_R, inp["turning"], inp["resampled"])
        rec.record_stroke(
            s, force=inp["force"], foot_pos_w=inp["foot_pos_w"], root_pos_w=inp["root_pos_w"],
            heading=inp["heading"], cmd=inp["cmd"], stand=inp["stand"],
            root_lin_vel_b_xy=inp["root_lin_vel_b_xy"], root_ang_vel_b_z=inp["root_ang_vel_b_z"],
            turning=inp["turning"], resampled=inp["resampled"], reward=R, pay=pay, dt=DT, turn_radius=TURN_R)
        if s >= 3:  # the centring term joins in late -- rows 0-2 never complete
            cst = SM.init_centring_state(num_envs)
            b, _cst, o = SM.body_centring_step(
                inp["foot_pos_w"], inp["root_pos_w"], inp["heading"], inp["cmd"], inp["stand"],
                inp["root_lin_vel_b_xy"], cst, DT, inp["lat_share"], inp["resampled"])
            rec.record_centring(
                s, foot_pos_w=inp["foot_pos_w"], root_pos_w=inp["root_pos_w"], heading=inp["heading"],
                cmd=inp["cmd"], stand=inp["stand"], lat_share=inp["lat_share"], resampled=inp["resampled"],
                b=b, o=o, dt=DT)
    assert not rec._saved  # never hit the auto-save condition (10 slots used, but 3 of them incomplete)
    rec.close()
    with np.load(dump_path) as z:
        dump = {k: z[k] for k in z.files}
    # every column still has num_steps rows (the recorder never shrinks to the complete prefix), but
    # rows 0-2's "b" is a zero default, not a real centring recompute -- the tool's own check on this
    # constructed-by-hand dump would only be meaningful over rows 3.. (parity_check has no row mask,
    # so this test only pins the recorder's own behaviour, not a tool-side guard).
    assert dump["b"].shape[0] == num_steps
    assert (dump["b"][:3] == 0.0).all()


# =========================================================== per-category (cat/cat_scale), 2026-09-27
def test_record_stroke_without_cat_scale_never_writes_cat_columns(tmp_path):
    """Knob off: ``RewardFootStroke`` passes ``cat=None, cat_scale=None`` (the defaults) every step
    -- exactly what ``_build_dump``'s driver already does -- so the dump must carry neither new key,
    byte-identical to a dump saved before this knob existed."""
    dump_path = _build_dump(tmp_path, num_envs=3, reset_at={})
    with np.load(dump_path) as z:
        keys = set(z.files)
    assert "cat" not in keys
    assert "cat_scale" not in keys


def test_record_stroke_with_cat_scale_writes_both_new_columns(tmp_path):
    """Armed: ``cat`` ((steps, envs) int8, this step's TELEMETRY_CATEGORIES index) and ``cat_scale``
    ((7,) float, the resolved table) both land in the saved dump."""
    num_envs, num_steps = 4, 6
    dump_path = tmp_path / "cat_dump.npz"
    rec = P.ParityRecorder(str(dump_path), num_envs, num_steps)
    # stand, fwd, side, back, diag, turn, move_turn (TELEMETRY_CATEGORIES order) -- side multiplied 4.8x
    cat_scale = torch.tensor([1.0, 1.0, 4.8, 1.0, 1.2, 1.0, 1.0])
    for s, inp in enumerate(_synthetic_steps(num_envs, num_steps, {})):
        stt = SM.init_stroke_state(num_envs, NUM_FEET)
        R, _stt, pay = SM.foot_stroke_step(
            inp["force"], inp["foot_pos_w"], inp["root_pos_w"], inp["heading"], inp["cmd"], inp["stand"],
            inp["root_lin_vel_b_xy"], inp["root_ang_vel_b_z"], stt, DT, TURN_R, inp["turning"], inp["resampled"])
        cat = torch.full((num_envs,), 2, dtype=torch.int64)  # every env reads as 'side' this step
        rec.record_stroke(
            s, force=inp["force"], foot_pos_w=inp["foot_pos_w"], root_pos_w=inp["root_pos_w"],
            heading=inp["heading"], cmd=inp["cmd"], stand=inp["stand"],
            root_lin_vel_b_xy=inp["root_lin_vel_b_xy"], root_ang_vel_b_z=inp["root_ang_vel_b_z"],
            turning=inp["turning"], resampled=inp["resampled"], reward=R, pay=pay, dt=DT, turn_radius=TURN_R,
            cat=cat, cat_scale=cat_scale)
        cst = SM.init_centring_state(num_envs)
        b, _cst, o = SM.body_centring_step(
            inp["foot_pos_w"], inp["root_pos_w"], inp["heading"], inp["cmd"], inp["stand"],
            inp["root_lin_vel_b_xy"], cst, DT, inp["lat_share"], inp["resampled"])
        rec.record_centring(
            s, foot_pos_w=inp["foot_pos_w"], root_pos_w=inp["root_pos_w"], heading=inp["heading"],
            cmd=inp["cmd"], stand=inp["stand"], lat_share=inp["lat_share"], resampled=inp["resampled"],
            b=b, o=o, dt=DT)
    rec.close()
    with np.load(dump_path) as z:
        dump = {k: z[k] for k in z.files}
    assert dump["cat"].shape == (num_steps, num_envs)
    assert dump["cat"].dtype == np.int8
    assert (dump["cat"] == 2).all()
    assert dump["cat_scale"].shape == (7,)
    assert np.allclose(dump["cat_scale"], cat_scale.numpy())


# =============================================================================== corruption
def test_a_corrupted_dump_fails_the_tools_parity_check(tmp_path):
    dump_path = _build_dump(tmp_path)
    d = dict(np.load(dump_path))
    d["reward"] = d["reward"] + 1.0  # far beyond any float32 rounding tolerance
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, **d)

    with np.load(bad_path) as z:
        bad_dump = {k: z[k] for k in z.files}
    result = TOOL.parity_check(bad_dump)
    assert not result["ok"]
    assert result["max_abs"]["pay_sum"] > 1e-3
    assert TOOL.main(["parity", str(bad_path)]) == 1

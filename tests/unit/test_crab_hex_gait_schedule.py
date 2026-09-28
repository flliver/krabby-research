"""Unit tests for the crab-hex gait eval command schedule (Milestone 18, Task 0).

Pure -- no Isaac Sim. Covers the manifest validations that exist to stop a scenario from silently
measuring something other than what it says (dead-zone commands, yaw probes that the policy cannot
observe, schedules longer than their episode).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "scripts"
)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import schedule as S  # noqa: E402

DT = 0.02


def _scenario(**overrides) -> S.Scenario:
    kwargs = dict(
        id="t",
        task="Isaac-Crab-Hex-Teacher-v0",
        checkpoint="/tmp/model.pt",
        schedule=[
            S.Hold(hold_s=6.0, vx=0.45, label="low"),
            S.Hold(hold_s=6.0, vx=0.65, label="mid"),
            S.Hold(hold_s=6.0, vx=0.85, label="high"),
        ],
        episodes=4,
    )
    kwargs.update(overrides)
    return S.Scenario(**kwargs)


def test_compile_lengths_and_segments():
    sc = _scenario()
    comp = S.compile_schedule(sc, dt=DT)
    prologue = int(round(sc.get_default("prologue_s") / DT))
    assert comp.n_steps == prologue + 3 * int(round(6.0 / DT))
    assert comp.cmd.shape == (comp.n_steps, 4, 3)
    # prologue is never scored
    assert (comp.segment_id[:prologue] == -1).all()
    assert not comp.steady_mask[:prologue].any()
    assert set(np.unique(comp.segment_id[prologue:]).tolist()) == {0, 1, 2}


def test_settle_window_is_excluded_from_steady_mask():
    """The first second after a command change is the controller's step response, not its gait."""
    sc = _scenario()
    comp = S.compile_schedule(sc, dt=DT)
    prologue = int(round(sc.get_default("prologue_s") / DT))
    settle = int(round(sc.get_default("settle_s") / DT))
    assert not comp.steady_mask[prologue : prologue + settle, 0].any()
    assert comp.steady_mask[prologue + settle, 0]


def test_commands_match_schedule_values():
    sc = _scenario()
    comp = S.compile_schedule(sc, dt=DT)
    prologue = int(round(sc.get_default("prologue_s") / DT))
    hold_steps = int(round(6.0 / DT))
    assert comp.cmd[prologue + 1, 0, 0] == pytest.approx(0.45)
    assert comp.cmd[prologue + hold_steps + 1, 0, 0] == pytest.approx(0.65)
    assert comp.cmd[prologue + 2 * hold_steps + 1, 0, 0] == pytest.approx(0.85)


def test_rotate_holds_by_env_decorrelates_speed_from_course_position():
    sc = _scenario(rotate_holds_by_env=True)
    comp = S.compile_schedule(sc, dt=DT)
    prologue = int(round(sc.get_default("prologue_s") / DT))
    first_of_each_env = comp.segment_id[prologue + 1, :]
    assert first_of_each_env.tolist() == [0, 1, 2, 0]


def test_no_rotation_by_default():
    comp = S.compile_schedule(_scenario(), dt=DT)
    prologue = int(round(S.DEFAULTS["prologue_s"] / DT))
    assert (comp.segment_id[prologue + 1, :] == 0).all()


def test_prologue_holds_the_first_scored_command():
    """So the robot is already settled at the speed its first steady window will score."""
    sc = _scenario(rotate_holds_by_env=True)
    comp = S.compile_schedule(sc, dt=DT)
    assert comp.cmd[0, 1, 0] == pytest.approx(0.65)  # env 1 starts on hold index 1


def test_dead_zone_command_is_rejected():
    """|vx| under lin_vel_clip is silently zeroed to a stop by the command term."""
    sc = _scenario(schedule=[S.Hold(hold_s=5.0, vx=0.1, label="creep")])
    with pytest.raises(S.ManifestError, match="dead zone"):
        S.validate_scenario(sc)


def test_dead_zone_allowed_when_explicitly_labelled_stop():
    S.validate_scenario(_scenario(schedule=[S.Hold(hold_s=5.0, vx=0.0, label="stop")]))


def test_yaw_probe_requires_delta_yaw_injection():
    """Writing vel_command_b[:,2] alone is a no-op: wz never reaches the observation vector."""
    sc = _scenario(probe="yaw", schedule=[S.Hold(hold_s=5.0, vx=0.45, wz=0.3, label="yaw")])
    with pytest.raises(S.ManifestError, match="delta_yaw_inject"):
        S.validate_scenario(sc)


def test_yaw_mode_requires_a_delta_yaw_value():
    sc = _scenario(probe="yaw", yaw_mode="delta_yaw_inject")
    with pytest.raises(S.ManifestError, match="no hold sets delta_yaw"):
        S.validate_scenario(sc)


def test_delta_yaw_channel_compiled_when_present():
    sc = _scenario(
        probe="yaw",
        yaw_mode="delta_yaw_inject",
        schedule=[
            S.Hold(hold_s=4.0, vx=0.45, delta_yaw=0.0, label="a"),
            S.Hold(hold_s=4.0, vx=0.45, delta_yaw=0.4, label="b"),
        ],
    )
    S.validate_scenario(sc)
    comp = S.compile_schedule(sc, dt=DT)
    assert comp.delta_yaw is not None
    prologue = int(round(sc.get_default("prologue_s") / DT))
    hold = int(round(4.0 / DT))
    assert comp.delta_yaw[prologue + hold + 1, 0] == pytest.approx(0.4)


def test_committed_v2_manifest_parses_with_plan_f_probes():
    """The shipped scenarios_v2.yaml must load, including the PLAN F turn/speed probes
    (turn_walk_v1 exercises the yaw-probe fields end-to-end)."""
    manifest = (
        SCRIPTS_DIR.parent / "experiments" / "eval" / "scenarios_v2.yaml"
    )
    scenarios, defaults = S.load_manifest(manifest)
    by_id = {s.id: s for s in scenarios}
    assert {"flat_walk_forward_v2", "flat_walk_slow_v2", "turn_walk_v1",
            "flat_walk_speed_v1"} <= set(by_id)
    turn = by_id["turn_walk_v1"]
    assert turn.yaw_mode == "delta_yaw_inject" and turn.probe == "yaw"
    deltas = [h.delta_yaw for h in turn.schedule]
    assert deltas == [None, 0.4, -0.4, 0.8]
    speed = by_id["flat_walk_speed_v1"]
    assert [h.vx for h in speed.schedule] == [0.35, 0.45, 0.55]


def test_episode_shorter_than_schedule_is_rejected():
    with pytest.raises(S.ManifestError, match="shorter than"):
        S.validate_scenario(_scenario(episode_length_s=5.0))


def test_empty_schedule_is_rejected():
    with pytest.raises(S.ManifestError, match="empty schedule"):
        S.validate_scenario(_scenario(schedule=[]))


def test_load_manifest_roundtrip(tmp_path):
    payload = {
        "manifest_version": 1,
        "defaults": {"prologue_s": 2.0},
        "scenarios": [
            {
                "id": "s1",
                "task": "Isaac-Crab-Hex-Teacher-v0",
                "checkpoint": "/tmp/a.pt",
                "episodes": 3,
                "schedule": [{"vx": 0.45, "hold_s": 6.0, "label": "low"}],
            }
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))
    scenarios, defaults = S.load_manifest(path)
    assert len(scenarios) == 1
    assert scenarios[0].id == "s1"
    assert defaults["prologue_s"] == 2.0
    assert scenarios[0].get_default("prologue_s") == 2.0


def test_load_manifest_rejects_bad_version(tmp_path):
    path = tmp_path / "m.json"
    # Versions 1 and 2 are both live (v2 = gait-formation scenarios, 2026-08-20).
    path.write_text(json.dumps({"manifest_version": 3, "scenarios": []}))
    with pytest.raises(S.ManifestError, match="manifest_version"):
        S.load_manifest(path)


def test_load_manifest_rejects_duplicate_ids(tmp_path):
    entry = {
        "id": "dup",
        "task": "t",
        "checkpoint": "/tmp/a.pt",
        "schedule": [{"vx": 0.45, "hold_s": 3.0}],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_version": 1, "scenarios": [entry, dict(entry)]}))
    with pytest.raises(S.ManifestError, match="duplicate scenario ids"):
        S.load_manifest(path)

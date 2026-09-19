"""Unit tests for the crab-hex gait eval command schedule (Milestone 18, Task 0).

Pure -- no Isaac Sim. Covers the manifest validations that exist to stop a scenario from silently
measuring something other than what it says (dead-zone commands, yaw probes that the policy cannot
observe, schedules longer than their episode).

The second half covers the omni task's WORLD-frame rows (plan D10 C1): speed/dir/face holds, the
direction-scaled speed cap, turn feasibility, spawn-yaw pinning -- and that every body-row
behaviour above is untouched by them.
"""

import json
import math
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


def test_committed_side_manifest_parses():
    """The shipped scenarios_side.yaml (sideways-formation campaign) must load: slow__side holds
    the stand-first band along the travel axis (vx), 100 episodes, no env block (frozen shape),
    and step__side carries exactly the four light-shallow-terrain knobs."""
    manifest = (
        SCRIPTS_DIR.parents[1]
        / "crab_hex_sideways_task"
        / "experiments"
        / "eval"
        / "scenarios_side.yaml"
    )
    scenarios, defaults = S.load_manifest(manifest)
    by_id = {s.id: s for s in scenarios}
    assert {"slow__side", "step__side", "flat_walk_slow_side"} <= set(by_id)
    slow = by_id["slow__side"]
    assert slow.task == "Isaac-Crab-Hex-Side-Walk-v0"
    assert [h.vx for h in slow.schedule] == [0.0, 0.25, 0.35]
    assert slow.episodes == 100
    assert slow.env_vars == {}
    S.validate_scenario(slow)
    step = by_id["step__side"]
    assert set(step.env_vars) == {
        "KRABBY_FLAT_TERRAIN_MODE",
        "KRABBY_FLAT_TERRAIN_FLAT_FRAC",
        "KRABBY_FLAT_TERRAIN_GEOM",
        "KRABBY_FLAT_TERRAIN_DIFF",
    }


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


# ---------------------------------------------------------------------------
# world-frame rows (omni task, 2026-09-16): plan D10 C1. Holds carry speed/dir/face instead of
# vx/vy/wz and are handed to the command term's schedule_hook rather than written into
# vel_command_b; every rule below exists so a manifest cannot ask for a command the sampler would
# never produce or a turn the plant cannot finish inside its hold.
# ---------------------------------------------------------------------------
HALF_PI = math.pi / 2


def _world_scenario(**overrides) -> S.Scenario:
    kwargs = dict(
        id="w",
        task="Isaac-Crab-Hex-Omni-Walk-Play-v0",
        checkpoint="/tmp/model.pt",
        schedule=[
            S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=0.0, label="stand"),
            S.Hold(hold_s=8.0, speed=0.25, dir=0.0, face=0.0, label="fwd"),
            S.Hold(hold_s=8.0, speed=0.22, dir=HALF_PI, face=0.0, label="lat_l"),
            S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=HALF_PI, label="turn_l90"),
        ],
        episodes=4,
        speed_ellipse={"fwd": 0.35, "lat": 0.25, "back": 0.25},
        ang_vel_max=0.3,
    )
    kwargs.update(overrides)
    return S.Scenario(**kwargs)


def test_ellipse_cap_axes_and_interpolation():
    fwd, lat, back = 0.35, 0.25, 0.25
    assert S.ellipse_cap(0.0, fwd, lat, back) == pytest.approx(fwd)
    assert S.ellipse_cap(HALF_PI, fwd, lat, back) == pytest.approx(lat)
    assert S.ellipse_cap(-HALF_PI, fwd, lat, back) == pytest.approx(lat)
    assert S.ellipse_cap(math.pi, fwd, lat, back) == pytest.approx(back)
    # Elliptic interpolation at 45 deg: 1/hypot(cos/a, sin/b), strictly between the two axes.
    diag = S.ellipse_cap(math.pi / 4, fwd, lat, back)
    assert diag == pytest.approx(1.0 / math.hypot(math.cos(math.pi / 4) / fwd, math.sin(math.pi / 4) / lat))
    assert lat < diag < fwd
    # An asymmetric ellipse: the rear half uses `back`.
    assert S.ellipse_cap(math.pi, 0.35, 0.25, 0.10) == pytest.approx(0.10)


def test_command_frame_is_derived_from_the_holds():
    assert _scenario().command_frame == "body"
    assert _world_scenario().command_frame == "world"


def test_world_holds_compile_to_per_step_arrays():
    sc = _world_scenario()
    comp = S.compile_schedule(sc, dt=DT)
    prologue = int(round(sc.get_default("prologue_s") / DT))
    hold = int(round(8.0 / DT))
    assert comp.command_frame == "world"
    for arr in (comp.world_speed, comp.world_dir, comp.world_face):
        assert arr.shape == (comp.n_steps, 4)
    # Nothing is written into the body command on a world row.
    assert not comp.cmd.any()
    # Prologue holds the first scored command, exactly as body rows do.
    assert comp.world_speed[0, 0] == pytest.approx(0.0)
    assert comp.world_speed[prologue + hold + 1, 0] == pytest.approx(0.25)
    assert comp.world_dir[prologue + 2 * hold + 1, 0] == pytest.approx(HALF_PI)
    assert comp.world_face[prologue + 3 * hold + 1, 0] == pytest.approx(HALF_PI)
    # Segments/steady mask are unchanged machinery.
    assert set(np.unique(comp.segment_id[prologue:]).tolist()) == {0, 1, 2, 3}


def test_body_rows_carry_no_world_arrays():
    comp = S.compile_schedule(_scenario(), dt=DT)
    assert comp.command_frame == "body"
    assert comp.world_speed is None and comp.world_dir is None and comp.world_face is None


def test_schedule_row_is_the_hook_contract():
    """The command term's schedule_hook reads .speed/.dir/.face (and may read it as a mapping)."""
    comp = S.compile_schedule(_world_scenario(), dt=DT)
    prologue = int(round(S.DEFAULTS["prologue_s"] / DT))
    row = comp.row(prologue + int(round(8.0 / DT)) + 1)
    # Uniform schedule -> plain floats, the scalar half of the contract.
    assert comp.world_uniform
    assert isinstance(row.speed, float) and row.speed == pytest.approx(0.25)
    assert isinstance(row.dir, float) and row.dir == pytest.approx(0.0)
    assert isinstance(row.face, float) and row.face == pytest.approx(0.0)
    assert row.label == "fwd"
    assert row["speed"] == row.speed and row.get("face") == row.face
    assert set(dict(row)) == {"speed", "dir", "face", "label", "step"}
    assert comp.row(0).label == "prologue"


def test_schedule_row_is_per_env_when_envs_run_different_holds():
    """rotate_holds_by_env puts each env on a different hold at the same step, so the row has to
    carry one value per env -- the array half of the contract."""
    sc = _world_scenario(
        rotate_holds_by_env=True,
        schedule=[
            S.Hold(hold_s=8.0, speed=0.25, dir=0.0, face=0.0, label="fwd"),
            S.Hold(hold_s=8.0, speed=0.22, dir=HALF_PI, face=0.0, label="lat_l"),
        ],
        episodes=2,
    )
    comp = S.compile_schedule(sc, dt=DT, num_envs=2)
    assert not comp.world_uniform
    row = comp.row(int(round(S.DEFAULTS["prologue_s"] / DT)) + 1)
    assert row.speed.tolist() == pytest.approx([0.25, 0.22])
    assert row.dir.tolist() == pytest.approx([0.0, HALF_PI])


def test_mixed_command_frames_are_rejected():
    sc = _world_scenario(
        schedule=[
            S.Hold(hold_s=8.0, speed=0.25, dir=0.0, face=0.0, label="fwd"),
            S.Hold(hold_s=8.0, vx=0.25, label="body"),
        ]
    )
    with pytest.raises(S.ManifestError, match="mixed command frames"):
        S.validate_scenario(sc)


def test_world_hold_may_not_carry_body_fields():
    sc = _world_scenario(
        schedule=[S.Hold(hold_s=8.0, speed=0.25, dir=0.0, face=0.0, wz=0.3, label="both")]
    )
    with pytest.raises(S.ManifestError, match="world-frame holds carry only"):
        S.validate_scenario(sc)


def test_world_hold_in_a_manifest_may_not_carry_body_keys(tmp_path):
    """Key PRESENCE is the manifest-level rule -- `vx: 0.0` is still a body key."""
    payload = {
        "manifest_version": 2,
        "scenarios": [
            {
                "id": "w",
                "task": "t",
                "checkpoint": "/tmp/a.pt",
                "schedule": [{"speed": 0.25, "dir": 0.0, "face": 0.0, "vx": 0.0, "hold_s": 8.0}],
            }
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(S.ManifestError, match="world-frame holds carry only"):
        S.load_manifest(path)


def test_partial_world_hold_is_rejected(tmp_path):
    payload = {
        "manifest_version": 2,
        "scenarios": [
            {
                "id": "w",
                "task": "t",
                "checkpoint": "/tmp/a.pt",
                "schedule": [{"speed": 0.25, "dir": 0.0, "hold_s": 8.0}],
            }
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(S.ManifestError, match="needs all of"):
        S.load_manifest(path)


def test_speed_above_the_direction_scaled_cap_is_rejected():
    """0.30 m/s sideways is inside the fore-aft band but outside the lateral one."""
    sc = _world_scenario(
        schedule=[S.Hold(hold_s=8.0, speed=0.30, dir=HALF_PI, face=0.0, label="lat_fast")]
    )
    with pytest.raises(S.ManifestError, match="exceeds the direction-scaled cap"):
        S.validate_scenario(sc)
    # The same speed straight ahead is fine (fore-aft semi-axis 0.35).
    S.validate_scenario(
        _world_scenario(schedule=[S.Hold(hold_s=8.0, speed=0.30, dir=0.0, face=0.0, label="fwd")])
    )


def test_speed_cap_is_measured_in_the_body_frame():
    """dir and face rotate together -> still a forward command, so the fore-aft cap applies."""
    S.validate_scenario(
        _world_scenario(
            schedule=[S.Hold(hold_s=8.0, speed=0.30, dir=HALF_PI, face=HALF_PI, label="fwd_rotated")]
        )
    )
    with pytest.raises(S.ManifestError, match="exceeds the direction-scaled cap"):
        S.validate_scenario(
            _world_scenario(
                schedule=[S.Hold(hold_s=8.0, speed=0.30, dir=math.pi, face=HALF_PI, label="lat_rotated")]
            )
        )


def test_turn_that_cannot_finish_inside_its_hold_is_rejected():
    """settle_s 1.0 + |dface|/0.3: a 90 deg turn needs 6.24 s, a 180 deg one 11.47 s."""
    sc = _world_scenario(
        schedule=[
            S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=HALF_PI, label="turn_l90"),
            S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=-HALF_PI, label="turn_180"),
        ]
    )
    with pytest.raises(S.ManifestError, match="cannot finish inside the hold"):
        S.validate_scenario(sc)
    # 6.5 s is enough for 90 deg, 6.0 s is not.
    S.validate_scenario(
        _world_scenario(
            schedule=[
                S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=0.0, label="a"),
                S.Hold(hold_s=6.5, speed=0.0, dir=0.0, face=HALF_PI, label="b"),
            ]
        )
    )
    with pytest.raises(S.ManifestError, match="cannot finish inside the hold"):
        S.validate_scenario(
            _world_scenario(
                schedule=[
                    S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=0.0, label="a"),
                    S.Hold(hold_s=6.0, speed=0.0, dir=0.0, face=HALF_PI, label="b"),
                ]
            )
        )


def test_turn_feasibility_uses_the_manifest_ang_vel_max():
    """A slower yaw cap makes the same 90 deg turn infeasible."""
    holds = [
        S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=0.0, label="a"),
        S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=HALF_PI, label="b"),
    ]
    S.validate_scenario(_world_scenario(schedule=holds))
    with pytest.raises(S.ManifestError, match="ang_vel_max"):
        S.validate_scenario(_world_scenario(schedule=holds, ang_vel_max=0.2))


def test_rotated_hold_order_also_checks_the_wrap_around_turn():
    """With rotate_holds_by_env some env runs last -> first, so that pair is a real transition."""
    holds = [
        S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=0.0, label="a"),
        S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=HALF_PI, label="b"),
        S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=math.pi, label="c"),
    ]
    S.validate_scenario(_world_scenario(schedule=holds))  # a->b, b->c are 90 deg each
    with pytest.raises(S.ManifestError, match="cannot finish inside the hold"):
        S.validate_scenario(_world_scenario(schedule=holds, rotate_holds_by_env=True))  # c->a is 180


def test_yaw_probe_is_rejected_on_world_rows():
    with pytest.raises(S.ManifestError, match="body-row probe"):
        S.validate_scenario(_world_scenario(probe="yaw", yaw_mode="delta_yaw_inject"))


def test_world_row_pins_the_spawn_yaw_to_the_first_face_by_default():
    sc = _world_scenario()
    assert sc.spawn_yaw == "first_face"
    assert sc.spawn_yaw_rad() == pytest.approx(0.0)
    turn_first = _world_scenario(
        schedule=[S.Hold(hold_s=8.0, speed=0.0, dir=0.0, face=HALF_PI, label="t")]
    )
    assert turn_first.spawn_yaw_rad() == pytest.approx(HALF_PI)
    assert _world_scenario(spawn_yaw=-0.5).spawn_yaw_rad() == pytest.approx(-0.5)
    with pytest.raises(S.ManifestError, match="spawn_yaw"):
        S.validate_scenario(_world_scenario(spawn_yaw="wherever"))


def test_body_rows_never_pin_the_spawn_yaw():
    """Forward/sideways rows must not acquire a spawn override they never had."""
    sc = _scenario()
    assert sc.spawn_yaw is None and sc.spawn_yaw_rad() is None


def test_world_episode_shorter_than_schedule_is_rejected():
    with pytest.raises(S.ManifestError, match="shorter than"):
        S.validate_scenario(_world_scenario(episode_length_s=5.0))


def test_committed_omni_manifest_parses():
    """The shipped scenarios_omni.yaml: three world rows, 8 holds (6 x 8 s + the two turn holds
    `turn_l90`/`turn_r90` x 12 s -- step-16 review decision, 2026-09-17), the ring env on
    light__omni only, and every validator rule satisfied as committed. Plus the Part A6
    curriculum-chain telemetry row `curriculum__omni` (10 holds, 2026-09-18)."""
    manifest = (
        SCRIPTS_DIR.parents[1]
        / "crab_hex_omni_task"
        / "experiments"
        / "eval"
        / "scenarios_omni.yaml"
    )
    scenarios, _ = S.load_manifest(manifest)
    by_id = {s.id: s for s in scenarios}
    assert set(by_id) == {"slow__omni", "light__omni", "ring__omni", "curriculum__omni"}

    slow = by_id["slow__omni"]
    assert slow.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0"
    assert slow.command_frame == "world"
    assert slow.episodes == 100 and slow.episode_length_s == 80.0
    assert slow.env_vars == {}  # frozen shape: no env block on the canary
    assert slow.get_speed_ellipse() == (0.35, 0.25, 0.25)
    assert slow.ang_vel_max == 0.3
    assert slow.spawn_yaw == "first_face" and slow.spawn_yaw_rad() == pytest.approx(0.0)
    assert [h.label for h in slow.schedule] == [
        "stand", "fwd", "lat_l", "diag", "back", "turn_l90", "turn_r90", "move_turn"
    ]
    expected_hold_s = [8.0, 8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 8.0]  # turn_l90/turn_r90 are 12 s
    assert [h.hold_s for h in slow.schedule] == expected_hold_s
    assert [h.speed for h in slow.schedule] == [0.0, 0.25, 0.22, 0.25, 0.22, 0.0, 0.0, 0.22]
    dirs = [0.0, 0.0, HALF_PI, math.pi / 4, math.pi, 0.0, 0.0, 0.0]
    faces = [0.0, 0.0, 0.0, 0.0, 0.0, HALF_PI, 0.0, HALF_PI]
    assert [h.dir for h in slow.schedule] == pytest.approx(dirs, abs=1e-6)
    assert [h.face for h in slow.schedule] == pytest.approx(faces, abs=1e-6)
    # move_turn travels along world +x while facing +y == a lateral command in the body frame.
    move_turn = slow.schedule[-1]
    assert S.wrap_to_pi(move_turn.dir - move_turn.face) == pytest.approx(-HALF_PI, abs=1e-6)

    # Every row in the manifest must validate, whatever the set of rows becomes.
    for sc in scenarios:
        S.validate_scenario(sc)
    # curriculum__omni carries a different (longer) schedule, so the "every row is byte-identical
    # to slow__omni" check applies only to the three legacy rows (frozen shape, header note above).
    for sc in (by_id["slow__omni"], by_id["light__omni"], by_id["ring__omni"]):
        assert [h.as_world() for h in sc.schedule] == [h.as_world() for h in slow.schedule]
        assert [h.hold_s for h in sc.schedule] == expected_hold_s
    assert by_id["light__omni"].env_vars == {
        "KRABBY_FLAT_TERRAIN_MODE": "light",
        "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0",
        "KRABBY_FLAT_TERRAIN_GEOM": "ring",
        "KRABBY_FLAT_TERRAIN_DIFF": "0.05:0.2",
    }
    assert by_id["ring__omni"].env_vars == {}
    assert by_id["ring__omni"].task == "Isaac-Crab-Hex-Omni-Walk-Play-v0"

    # Part A6: curriculum__omni -- the curriculum-chain telemetry canary. Prefix-ordered by
    # enabling stage (c1 stand/fwd, c2 lat_l/lat_r/back, c3 turn_l90/turn_r90, c4
    # move_turn/move_turn_r, c5 diag); WATCH only, scoring stays on slow__omni (decision 5).
    curriculum = by_id["curriculum__omni"]
    assert curriculum.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0"
    assert curriculum.role == "teacher"
    assert curriculum.terrain_label == "flat"
    assert curriculum.command_frame == "world"
    assert curriculum.episodes == 100 and curriculum.episode_length_s == 92.0
    assert curriculum.env_seed == 1
    assert curriculum.env_vars == {}  # flat terrain, same convention as slow__omni
    # The row carries no speed_ellipse/ang_vel_max of its own: both are inherited from the
    # file-level blocks, and the hold speeds below are only legal under these two.
    assert curriculum.get_speed_ellipse() == (0.35, 0.25, 0.25)
    assert curriculum.ang_vel_max == 0.3
    assert curriculum.spawn_yaw == "first_face" and curriculum.spawn_yaw_rad() == pytest.approx(0.0)
    assert [h.label for h in curriculum.schedule] == [
        "stand", "fwd", "lat_l", "lat_r", "back", "turn_l90", "turn_r90",
        "move_turn", "move_turn_r", "diag",
    ]
    expected_curr_hold_s = [8.0, 8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 8.0, 8.0, 8.0]
    assert [h.hold_s for h in curriculum.schedule] == expected_curr_hold_s
    # 8 holds x 8 s + 2 turn holds x 12 s = 88 s schedule + the 1.5 s prologue = 89.5 s.
    assert curriculum.schedule_duration_s == pytest.approx(88.0)
    assert curriculum.total_duration_s == pytest.approx(89.5)
    curr_speeds = [0.0, 0.25, 0.22, 0.22, 0.22, 0.0, 0.0, 0.22, 0.22, 0.25]
    assert [h.speed for h in curriculum.schedule] == pytest.approx(curr_speeds)
    curr_dirs = [0.0, 0.0, HALF_PI, -HALF_PI, math.pi, 0.0, 0.0, 0.0, 0.0, math.pi / 4]
    curr_faces = [0.0, 0.0, 0.0, 0.0, 0.0, HALF_PI, 0.0, HALF_PI, 0.0, 0.0]
    assert [h.dir for h in curriculum.schedule] == pytest.approx(curr_dirs, abs=1e-6)
    assert [h.face for h in curriculum.schedule] == pytest.approx(curr_faces, abs=1e-6)
    # stand/fwd/lat_l/back/turn_l90/turn_r90/move_turn/diag are copied VERBATIM from slow__omni.
    slow_by_label = {h.label: h for h in slow.schedule}
    shared = {"stand": 0, "fwd": 1, "lat_l": 2, "back": 4, "turn_l90": 5, "turn_r90": 6,
              "move_turn": 7, "diag": 9}
    for label, curr_idx in shared.items():
        assert curriculum.schedule[curr_idx].as_world() == pytest.approx(
            slow_by_label[label].as_world(), abs=1e-9
        )
        # VERBATIM includes the duration: pinned against slow__omni's own, so the two rows
        # cannot de-sync if slow__omni's 12-s turn holds are ever retuned.
        assert curriculum.schedule[curr_idx].hold_s == slow_by_label[label].hold_s
    # lat_r / move_turn_r are new holds (not present in slow__omni at all): lat_r is lat_l
    # mirrored; move_turn_r is move_turn's RIGHT-HANDED counterpart -- it travels world +x while
    # turning back to face 0, NOT a reflection to face -pi/2 (which would be a 180-degree swing
    # off move_turn and would fail the 8-s turn-feasibility rule).
    assert curriculum.schedule[3].as_world() == pytest.approx((0.22, -HALF_PI, 0.0), abs=1e-6)
    assert curriculum.schedule[8].as_world() == pytest.approx((0.22, 0.0, 0.0), abs=1e-6)


def test_spawn_pin_that_disagrees_with_the_first_face_must_also_be_feasible():
    """A numeric spawn_yaw makes the first hold a turn; "first_face" makes it a zero turn."""
    holds = [S.Hold(hold_s=6.5, speed=0.0, dir=0.0, face=0.0, label="a")]
    S.validate_scenario(_world_scenario(schedule=holds))  # spawn == first face -> no turn
    S.validate_scenario(_world_scenario(schedule=holds, spawn_yaw=-1.0))  # 1.0 rad needs 4.33 s
    with pytest.raises(S.ManifestError, match="cannot finish inside the hold"):
        S.validate_scenario(_world_scenario(schedule=holds, spawn_yaw=math.pi))

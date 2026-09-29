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
    curriculum-chain telemetry row `curriculum__omni` (10 holds, 2026-09-18) and the per-category
    suites `suite__omni` (plan D1, 2026-09-21) and `side22_25__omni` (the side-speed bisection
    campaign's gate row, side-speed-bisection.md plan step 3, 2026-09-28)."""
    manifest = (
        SCRIPTS_DIR.parents[1]
        / "crab_hex_omni_task"
        / "experiments"
        / "eval"
        / "scenarios_omni.yaml"
    )
    scenarios, _ = S.load_manifest(manifest)
    by_id = {s.id: s for s in scenarios}
    assert set(by_id) == {
        "slow__omni", "light__omni", "ring__omni", "curriculum__omni", "suite__omni",
        "side22_25__omni",
    }

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

    # Plan D1 (2026-09-21): suite__omni -- ten waves, one category hold per wave, settle first.
    suite = by_id["suite__omni"]
    assert suite.kind == S.SUITE_KIND and S.uses_waves(suite)
    assert suite.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0"
    assert suite.role == "teacher" and suite.terrain_label == "flat"
    assert suite.command_frame == "world"
    assert suite.episodes == 100 and suite.waves == 10 and suite.env_seed == 1
    assert suite.episode_length_s == 18.0
    assert suite.env_vars == {}
    assert suite.spawn_yaw == "first_face" and suite.spawn_yaw_rad() == pytest.approx(0.0)
    assert suite.get_speed_ellipse() == (0.35, 0.25, 0.25) and suite.ang_vel_max == 0.3
    assert suite.suite.settle_hold_s == 2.0 and suite.suite.settle_label == "settle"
    assert suite.suite.hold_s == 12.0
    assert suite.schedule_duration_s == pytest.approx(14.0)
    assert suite.total_duration_s == pytest.approx(15.5)
    assert [(c.name, c.kind) for c in suite.suite.categories] == [
        ("stand", "stand"), ("fwd", "translate"), ("side", "translate"), ("back", "translate"),
        ("diag", "translate"), ("turn", "turn"), ("move_turn", "move_turn"),
    ]
    holds = suite.suite.holds()
    assert [h.label for h in holds] == [
        "stand", "fwd", "lat_l", "lat_r", "back", "diag", "turn_l90", "turn_r90", "move_turn", "move_turn_r",
    ]
    assert [h.hand for h in holds] == [None, None, "l", "r", None, None, "l", "r", "l", "r"]
    assert [h.speed for h in holds] == pytest.approx([0.0, 0.25, 0.22, 0.22, 0.22, 0.25, 0.0, 0.0, 0.22, 0.22])
    assert [h.dir for h in holds] == pytest.approx(
        [0.0, 0.0, HALF_PI, -HALF_PI, math.pi, math.pi / 4, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert [h.face for h in holds] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, HALF_PI, -HALF_PI, HALF_PI, -HALF_PI], abs=1e-6)
    # Every hold whose command is the legacy one keeps the legacy label AND command; turn_r90 and
    # move_turn_r are the two new meanings (a right turn from face 0 / the mirror of move_turn).
    for h in holds:
        if h.label in slow_by_label and h.label != "turn_r90":
            assert (h.speed, h.dir, h.face) == pytest.approx(slow_by_label[h.label].as_world(), abs=1e-9)
    assert slow_by_label["turn_r90"].face == pytest.approx(0.0)
    assert [h.label for h in suite.schedule] == ["settle"] + [h.label for h in holds]
    assert suite.schedule[0].as_world() == (0.0, 0.0, 0.0) and suite.schedule[0].hold_s == 2.0


def test_spawn_pin_that_disagrees_with_the_first_face_must_also_be_feasible():
    """A numeric spawn_yaw makes the first hold a turn; "first_face" makes it a zero turn."""
    holds = [S.Hold(hold_s=6.5, speed=0.0, dir=0.0, face=0.0, label="a")]
    S.validate_scenario(_world_scenario(schedule=holds))  # spawn == first face -> no turn
    S.validate_scenario(_world_scenario(schedule=holds, spawn_yaw=-1.0))  # 1.0 rad needs 4.33 s
    with pytest.raises(S.ManifestError, match="cannot finish inside the hold"):
        S.validate_scenario(_world_scenario(schedule=holds, spawn_yaw=math.pi))


# ---------------------------------------------------------------------------
# category suites (omni per-category eval, plan D1/D2/A1, 2026-09-21): one hold per wave, the waves
# run in one launch, every episode carries exactly one category hold. The schedule is DERIVED (a
# label union, settle first) and compiles per wave; every legacy row compiles exactly as before.
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = SCRIPTS_DIR / "eval_crab_hex_gait.py"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SUITE_LABELS = ["stand", "fwd", "lat_l", "lat_r", "back", "diag", "turn_l90", "turn_r90", "move_turn", "move_turn_r"]


def _suite_row(**overrides) -> dict:
    """The plan-D1 suite row as a manifest mapping (JSON, so no PyYAML is needed)."""
    row = {
        "id": "suite",
        "kind": "category_suite",
        "task": "Isaac-Crab-Hex-Omni-Walk-Play-v0",
        "checkpoint": "/tmp/model.pt",
        "episodes": 100,
        "waves": 10,
        "env_seed": 1,
        "episode_length_s": 18.0,
        "spawn_yaw": "first_face",
        "settle": {"hold_s": 2.0, "label": "settle"},
        "hold_s": 12.0,
        "categories": [
            {"name": "stand", "kind": "stand", "holds": [{"label": "stand", "speed": 0.0, "dir": 0.0, "face": 0.0}]},
            {"name": "fwd", "kind": "translate", "holds": [{"label": "fwd", "speed": 0.25, "dir": 0.0, "face": 0.0}]},
            {"name": "side", "kind": "translate", "holds": [
                {"label": "lat_l", "speed": 0.22, "dir": HALF_PI, "face": 0.0, "hand": "l"},
                {"label": "lat_r", "speed": 0.22, "dir": -HALF_PI, "face": 0.0, "hand": "r"},
            ]},
            {"name": "back", "kind": "translate", "holds": [{"label": "back", "speed": 0.22, "dir": math.pi, "face": 0.0}]},
            {"name": "diag", "kind": "translate", "holds": [{"label": "diag", "speed": 0.25, "dir": math.pi / 4, "face": 0.0}]},
            {"name": "turn", "kind": "turn", "holds": [
                {"label": "turn_l90", "speed": 0.0, "dir": 0.0, "face": HALF_PI, "hand": "l"},
                {"label": "turn_r90", "speed": 0.0, "dir": 0.0, "face": -HALF_PI, "hand": "r"},
            ]},
            {"name": "move_turn", "kind": "move_turn", "holds": [
                {"label": "move_turn", "speed": 0.22, "dir": 0.0, "face": HALF_PI, "hand": "l"},
                {"label": "move_turn_r", "speed": 0.22, "dir": 0.0, "face": -HALF_PI, "hand": "r"},
            ]},
        ],
    }
    row.update(overrides)
    return row


def _load_rows(tmp_path, *rows) -> list:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({
        "manifest_version": 2,
        "speed_ellipse": {"fwd": 0.35, "lat": 0.25, "back": 0.25},
        "ang_vel_max": 0.3,
        "scenarios": list(rows),
    }))
    return S.load_manifest(path)[0]


def _suite(tmp_path, **overrides) -> S.Scenario:
    return _load_rows(tmp_path, _suite_row(**overrides))[0]


def test_category_suite_row_derives_the_union_schedule(tmp_path):
    sc = _suite(tmp_path)
    assert sc.kind == S.SUITE_KIND and sc.waves == 10
    # 11 labels: the settle first (segment 0), then one hold per category hold in manifest order.
    assert [h.label for h in sc.schedule] == ["settle"] + SUITE_LABELS
    assert sc.suite.hold_labels() == ["settle"] + SUITE_LABELS
    assert sc.schedule[0].as_world() == (0.0, 0.0, 0.0) and sc.schedule[0].hold_s == 2.0
    assert all(h.hold_s == 12.0 for h in sc.schedule[1:])
    # Per-WAVE durations, not the sum of the union: 2 + 12 = 14 s; + the 1.5 s prologue = 15.5 s.
    assert sc.schedule_duration_s == pytest.approx(14.0)
    assert sc.total_duration_s == pytest.approx(15.5)
    assert sc.command_frame == "world" and sc.spawn_yaw_rad() == 0.0
    assert sc.suite.hold_s_by_label() == {lab: 12.0 for lab in SUITE_LABELS}
    assert sc.suite.category_of_hold()["lat_r"] == "side"
    assert sc.suite.hand_of_hold()["move_turn_r"] == "r" and sc.suite.hand_of_hold()["fwd"] is None


def test_wave_plan_gives_every_hold_100_episodes_and_unique_env_indices(tmp_path):
    sc = _suite(tmp_path)
    plan = S.wave_plan(sc)
    assert len(plan) == 1000
    assert [a.env_index for a in plan] == list(range(1000))
    assert all(a.env_index == a.wave * 100 + a.env for a in plan)
    by_label = {lab: sum(1 for a in plan if a.label == lab) for lab in SUITE_LABELS}
    assert by_label == {lab: 100 for lab in SUITE_LABELS}
    assert all(a.label != "settle" for a in plan)             # the settle is never an own hold
    for cat in ("side", "turn", "move_turn"):
        eps = [a for a in plan if a.category == cat]
        assert len(eps) == 200
        assert sum(a.hand == "l" for a in eps) == 100 and sum(a.hand == "r" for a in eps) == 100
    for cat in ("stand", "fwd", "back", "diag"):
        eps = [a for a in plan if a.category == cat]
        assert len(eps) == 100 and all(a.hand is None for a in eps)
    # Each wave is exactly one hold, run by every env.
    for wave, label in enumerate(SUITE_LABELS):
        assert {a.label for a in plan if a.wave == wave} == {label}
        assert sorted(a.env for a in plan if a.wave == wave) == list(range(100))
    assert len(S.wave_plan(sc, num_envs=7)) == 70


def test_compile_wave_is_uniform_and_row_global(tmp_path):
    sc = _suite(tmp_path, episodes=3)
    prologue = int(round(1.5 / DT))
    settle = int(round(2.0 / DT))
    settle_s = int(round(sc.get_default("settle_s") / DT))
    for wave, label in enumerate(SUITE_LABELS):
        comp = S.compile_wave(sc, wave, dt=DT)
        assert comp.n_steps == 775 == prologue + settle + int(round(12.0 / DT))
        assert comp.cmd.shape == (775, 3, 3) and not comp.cmd.any()      # world rows write no vel_command_b
        assert comp.hold_labels == ["settle"] + SUITE_LABELS
        assert comp.hold_labels[wave + 1] == label
        assert set(np.unique(comp.segment_id).tolist()) == {-1, 0, wave + 1}
        assert (comp.segment_id[:prologue] == -1).all()
        assert (comp.segment_id[prologue : prologue + settle] == 0).all()
        assert (comp.segment_id[prologue + settle :] == wave + 1).all()
        # Steady windows: each hold's first settle_s is its transient, never scored.
        steady = comp.steady_mask[:, 0]
        assert not steady[: prologue + settle_s].any()
        assert steady[prologue + settle_s : prologue + settle].all()
        assert not steady[prologue + settle : prologue + settle + settle_s].any()
        assert steady[prologue + settle + settle_s :].all()
        assert comp.world_uniform and comp.delta_yaw is None
        # The prologue and the settle hold the settle command; the hold holds its own row.
        hold = sc.schedule[wave + 1]
        assert (comp.world_speed[: prologue + settle] == 0.0).all() and (comp.world_face[: prologue + settle] == 0.0).all()
        assert (comp.world_speed[prologue + settle :] == hold.speed).all()
        assert (comp.world_dir[prologue + settle :] == hold.dir).all()
        assert (comp.world_face[prologue + settle :] == hold.face).all()
        assert comp.world_speed.dtype == np.float64
        assert comp.wave_of_env.tolist() == [wave] * 3 and comp.hold_of_env.tolist() == [wave + 1] * 3
        row = comp.row(prologue + settle + 5)
        assert row.label == label and isinstance(row.speed, float)
        assert S.own_hold_end_step(prologue_s=1.5, settle_hold_s=2.0, hold_s=12.0, dt=DT) == comp.n_steps
    with pytest.raises(ValueError, match="out of range"):
        S.compile_wave(sc, 10, dt=DT)
    with pytest.raises(ValueError, match="category_suite"):
        S.compile_wave(_world_scenario(), 0, dt=DT)


def test_stack_waves_concatenates_and_labels_every_env(tmp_path):
    sc = _suite(tmp_path, episodes=4)
    parts = [S.compile_wave(sc, k, dt=DT) for k in range(sc.waves)]
    stacked = S.stack_waves(parts)
    assert stacked.n_steps == 775 and stacked.hold_labels == ["settle"] + SUITE_LABELS
    for name in ("cmd", "segment_id", "steady_mask", "world_speed", "world_dir", "world_face"):
        arr = getattr(stacked, name)
        assert arr.shape[:2] == (775, 40)
        for k, part in enumerate(parts):
            assert np.array_equal(arr[:, 4 * k : 4 * k + 4], getattr(part, name))
    assert stacked.wave_of_env.tolist() == [k for k in range(10) for _ in range(4)]
    assert stacked.hold_of_env.tolist() == [k + 1 for k in range(10) for _ in range(4)]
    # every env's own hold is the one its segment_id carries after the settle
    for env in range(40):
        assert set(np.unique(stacked.segment_id[:, env]).tolist()) == {-1, 0, int(stacked.hold_of_env[env])}
    assert stacked.delta_yaw is None
    assert not stacked.world_uniform                   # different envs run different holds
    with pytest.raises(ValueError):
        S.stack_waves([])
    with pytest.raises(ValueError, match="not a compile_wave"):
        S.stack_waves([S.compile_schedule(_world_scenario(), dt=DT)])


def test_compile_schedule_refuses_a_suite_row(tmp_path):
    with pytest.raises(ValueError, match="category_suite rows compile per wave: use compile_wave"):
        S.compile_schedule(_suite(tmp_path), dt=DT)


def test_suite_transition_pairs_are_from_the_settle_face(tmp_path):
    sc = _suite(tmp_path)
    faces = [h.face for h in sc.schedule]
    assert sc.transition_pairs() == [(0, 0.0, 0.0)] + [(i, 0.0, faces[i]) for i in range(1, 11)]
    # turn_r90 is a 90-degree RIGHT turn from the settle face, not a swing off turn_l90.
    assert (8, 0.0, pytest.approx(-HALF_PI)) in sc.transition_pairs()
    # Legacy rows: exactly today's pairs -- consecutive, + the spawn pin first, + the wrap-around
    # under rotation (which replaces the spawn pair).
    w = _world_scenario()
    wf = [h.face for h in w.schedule]
    assert w.transition_pairs() == [(0, 0.0, wf[0])] + list(zip(range(1, 4), wf[:-1], wf[1:]))
    rot = _world_scenario(rotate_holds_by_env=True)
    assert rot.transition_pairs() == list(zip(range(1, 4), wf[:-1], wf[1:])) + [(0, wf[-1], wf[0])]


def _set_hold(row: dict, which: str, **fields) -> dict:
    for cat in row["categories"]:
        for hold in cat["holds"]:
            if hold["label"] == which:
                hold.update(fields)
    return row


def _drop_hold(row: dict, label: str) -> dict:
    for cat in row["categories"]:
        cat["holds"] = [h for h in cat["holds"] if h["label"] != label]
    return row


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: _set_hold(r, "lat_r", label="lat_l"), "duplicate hold labels"),
        (lambda r: _set_hold(r, "fwd", label="settle"), "duplicate hold labels"),
        (lambda r: dict(r, schedule=[{"speed": 0.0, "dir": 0.0, "face": 0.0, "hold_s": 8.0}]), "remove the explicit"),
        (lambda r: dict(r, kind="bogus_suite"), "unknown kind"),
        (lambda r: (r["categories"][1].update(kind="spin"), r)[1], "unknown kind 'spin'"),
        (lambda r: dict(_drop_hold(r, "lat_r"), waves=9), "unbalanced hands"),
        (lambda r: _set_hold(r, "turn_r90", hand="l"), "unbalanced hands"),
        (lambda r: _set_hold(r, "turn_r90", hand="x"), "hand must be one of"),
        # 0.30 m/s along +pi/2 while facing +pi/2 clears the END cap (fore-aft 0.35) but not the
        # START one: the body still faces the settle face 0 when the hold begins (lateral 0.25).
        (lambda r: _set_hold(r, "move_turn", speed=0.30, dir=HALF_PI), "START body direction"),
        (lambda r: dict(r, hold_s=1.9, episode_length_s=18.0), "shorter than settle_s . min_window_s"),
        (lambda r: dict(r, episode_length_s=15.4), "shorter than prologue . schedule"),
        (lambda r: dict(r, waves=9), "waves=9"),
        (lambda r: dict(r, rotate_holds_by_env=True), "rotate_holds_by_env"),
        (lambda r: (r["categories"][5].update(watch_only=True), r)[1], "remove .'watch_only'."),
        (lambda r: (r["categories"][0].update(notes="x"), r)[1], "remove .'notes'."),
        (lambda r: _set_hold(r, "fwd", hold_s=8.0), "remove .'hold_s'."),
        (lambda r: dict(r, settle={"hold_s": 2.0, "label": "settle", "face": 1.0}), "remove .'face'."),
        # turn feasibility runs from the settle face: at ang_vel_max 0.1 a 90-degree turn needs
        # settle_s + (pi/2)/0.1 = 16.7 s > the 12-s hold (at the committed 0.3 it needs 6.24 s).
        (lambda r: dict(r, ang_vel_max=0.1), "cannot finish inside the hold"),
    ],
)
def test_suite_validator_rejects(tmp_path, mutate, match):
    with pytest.raises(S.ManifestError, match=match):
        _load_rows(tmp_path, mutate(_suite_row()))


def test_suite_validator_accepts_the_plan_row_and_rejects_suite_keys_on_a_schedule_row(tmp_path):
    _suite(tmp_path)  # the plan-D1 row validates as written
    legacy = {"id": "s", "task": "t", "checkpoint": "/tmp/a.pt",
              "schedule": [{"speed": 0.0, "dir": 0.0, "face": 0.0, "hold_s": 8.0}]}
    for key, value in (("waves", 1), ("settle", {"hold_s": 2.0}), ("hold_s", 12.0), ("categories", [])):
        with pytest.raises(S.ManifestError, match="category_suite"):
            _load_rows(tmp_path, dict(legacy, **{key: value}))
    assert _load_rows(tmp_path, dict(legacy, kind="schedule"))[0].kind == "schedule"
    # A schedule row built in code may not carry a suite either.
    with pytest.raises(S.ManifestError, match="belong to a 'category_suite' row"):
        S.validate_scenario(_world_scenario(waves=2))


def _committed_manifests() -> list[Path]:
    fwd_eval = SCRIPTS_DIR.parent / "experiments" / "eval"
    tasks = SCRIPTS_DIR.parents[1]
    return [
        fwd_eval / "scenarios_v1.yaml",
        fwd_eval / "scenarios_v2.yaml",
        fwd_eval / "scenarios_morph.yaml",
        tasks / "crab_hex_sideways_task" / "experiments" / "eval" / "scenarios_side.yaml",
        tasks / "crab_hex_omni_task" / "experiments" / "eval" / "scenarios_omni.yaml",
    ]


def test_only_a_suite_row_takes_the_wave_path():
    """``uses_waves`` -- the one predicate the harness loop branches on -- is False for every committed
    legacy row, so no legacy eval gains an ``env.seed`` call, an extra reset or a re-install."""
    suite_rows = []
    for manifest in _committed_manifests():
        for sc in S.load_manifest(manifest)[0]:
            if S.uses_waves(sc):
                suite_rows.append(sc.id)
            else:
                assert sc.kind == "schedule" and sc.suite is None and sc.waves == 1, sc.id
    assert suite_rows == ["suite__omni", "side22_25__omni"]
    # And the harness only calls env.seed / env.reset / re-installs the override under the suite
    # branch: every such call site sits inside an ``if`` whose test reads ``suite_mode``.
    import ast  # noqa: PLC0415

    tree = ast.parse(EVAL_SCRIPT.read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_env_call = (
            isinstance(func, ast.Attribute)
            and func.attr in ("seed", "reset")
            and isinstance(func.value, ast.Name)
            and func.value.id == "env"
        )
        is_reinstall = isinstance(func, ast.Name) and func.id == "_install_schedule_override_world"
        if not (is_env_call or is_reinstall):
            continue
        guarded, cur = False, node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If) and "suite_mode" in ast.unparse(cur.test):
                guarded = True
                break
        sites.append((ast.unparse(node), guarded))
    assert any(src.startswith("env.seed(") for src, _ in sites)
    assert any(src.startswith("env.reset(") for src, _ in sites)
    # the one unguarded install is the pre-loop one every world row has always run
    unguarded = [src for src, ok in sites if not ok]
    assert unguarded == ["_install_schedule_override_world(cmd_term, compiled, device, compiled.hold_labels)"]


def test_a_legacy_row_refuses_waves_and_trained_categories(tmp_path):
    legacy = S.load_manifest(_committed_manifests()[-1])[0][0]
    assert legacy.id == "slow__omni"
    assert S.suite_cli_args(legacy) is None
    for kwargs in ({"waves": 1}, {"trained_categories": "fwd"}, {"trained_categories": ""},
                   {"waves": 10, "trained_categories": "stand,fwd"}):
        with pytest.raises(S.ManifestError, match="category_suite' row only"):
            S.suite_cli_args(legacy, **kwargs)
    sc = _suite(tmp_path)
    assert S.suite_cli_args(sc) is None
    assert S.suite_cli_args(sc, waves=10) is None
    assert S.suite_cli_args(sc, trained_categories="stand, fwd,side") == ["stand", "fwd", "side"]
    assert S.suite_cli_args(sc, trained_categories="") == []
    with pytest.raises(S.ManifestError, match="--waves 9"):
        S.suite_cli_args(sc, waves=9)
    with pytest.raises(S.ManifestError, match="not categories"):
        S.suite_cli_args(sc, trained_categories="stand,lat_l")
    with pytest.raises(S.ManifestError, match="duplicate"):
        S.suite_cli_args(sc, trained_categories="fwd,fwd")
    # The harness routes both flags through this check before building anything.
    assert "S.suite_cli_args(" in EVAL_SCRIPT.read_text()


def test_suite_run_meta_records_the_waves_and_the_run_time_geometry(tmp_path):
    sc = _suite(tmp_path, episodes=7)
    seeds = [S.wave_seed(sc, k) for k in range(sc.waves)]
    assert seeds[0] == 1_000_003 and seeds[9] == 1_000_012
    meta = S.suite_run_meta(sc, dt=DT, num_envs=7, wave_seeds=seeds, trained_categories=["stand", "fwd"])
    assert set(meta) == {"waves", "wave_seeds", "episodes_total", "env_index_map", "suite",
                         "trained_categories", "hold_labels_kind"}
    assert meta["episodes_total"] == 70 and meta["waves"] == 10 and meta["hold_labels_kind"] == "union"
    assert [m["env_index"] for m in meta["env_index_map"]] == list(range(70))
    assert set(meta["suite"]) == {"dt", "prologue_s", "settle_label", "settle_hold_s", "hold_s", "categories"}
    assert meta["suite"]["hold_s"] == {lab: 12.0 for lab in SUITE_LABELS}
    assert [c["name"] for c in meta["suite"]["categories"]] == ["stand", "fwd", "side", "back", "diag", "turn", "move_turn"]
    json.dumps(meta)  # JSON-able as is


def _write_run_order(tmp_path, monkeypatch, scenario, compiled, run_meta, n_raw):
    """Run report.write_run on a tiny synthetic run and return the ordered list of written names."""
    from gait_eval import report as R  # noqa: PLC0415

    T = compiled.n_steps
    contact = np.zeros((T, n_raw, 6))
    raw = {
        "foot_force_norm": contact,
        "cmd_applied": np.zeros((T, n_raw, 3)),
        "root_lin_vel_b": np.zeros((T, n_raw, 3)),
        "root_ang_vel_b": np.zeros((T, n_raw, 3)),
        "root_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, n_raw, 1)),
        "root_pos_w": np.zeros((T, n_raw, 3)),
        "foot_pos_w": np.zeros((T, n_raw, 6, 3)),
        "foot_lin_vel_w": np.zeros((T, n_raw, 6, 3)),
        "foot_ang_vel_w": np.zeros((T, n_raw, 6, 3)),
        "terrain_z": np.zeros((T, n_raw, 6)),
        "actions": np.zeros((T, n_raw, 4)),
        "joint_vel": np.zeros((T, n_raw, 4)),
        "crab_failure": np.zeros((T, n_raw), dtype=bool),
    }
    if compiled.command_frame == "world":
        raw["heading_err"] = np.zeros((T, n_raw))
        raw["travel_yaw"] = np.zeros((T, n_raw))
    eps = R.score_run(raw, compiled=compiled, scenario=scenario, dt=DT, term_reason=run_meta["termination_reason"],
                      n_steps_env=run_meta["n_steps_per_env"],
                      travel_yaw=(raw["travel_yaw"] if "travel_yaw" in raw else 0.0))
    order: list[str] = []
    real_write_text = Path.write_text
    real_savez = np.savez_compressed

    def _write_text(self, *a, **k):
        order.append(self.name)
        return real_write_text(self, *a, **k)

    def _savez(path, *a, **k):
        order.append(Path(path).name)
        return real_savez(path, *a, **k)

    monkeypatch.setattr(Path, "write_text", _write_text)
    monkeypatch.setattr(np, "savez_compressed", _savez)
    run_dir = tmp_path / scenario.id
    R.write_run(run_dir, run_meta=run_meta, episode_metrics=eps, raw=raw, scenario=scenario,
                compiled=compiled, history_root=tmp_path / "history")
    monkeypatch.undo()
    return order, run_dir


def test_write_run_writes_the_completion_marker_last_on_suite_rows_only(tmp_path, monkeypatch):
    sc = _suite(tmp_path, episodes=2)
    parts = [S.compile_wave(sc, k, dt=DT) for k in range(sc.waves)]
    stacked = S.stack_waves(parts)
    n_raw = 20
    run_meta = {
        "scenario_id": sc.id, "task": sc.task, "checkpoint": "/tmp/model.pt", "checkpoint_sha256": "ab" * 32,
        "command_override_max_deviation": 0.0, "dt": DT, "num_envs": 2,
        "termination_reason": ["schedule_complete"] * n_raw, "n_steps_per_env": [stacked.n_steps] * n_raw,
        **S.suite_run_meta(sc, dt=DT, num_envs=2, wave_seeds=[S.wave_seed(sc, k) for k in range(10)],
                           trained_categories=None),
    }
    order, run_dir = _write_run_order(tmp_path, monkeypatch, sc, stacked, run_meta, n_raw)
    assert order[-1] == "complete.json" and order.count("complete.json") == 1
    # one raw per EPISODE (W*E = 20, n_raw), all before the marker; episode_{env_index:02d}
    raws = [name for name in order if name.endswith(".npz")]
    assert raws == [f"episode_{i:02d}.npz" for i in range(n_raw)]
    assert order.index("scenario_metrics.json") < order.index(raws[0])
    marker = json.loads((run_dir / "complete.json").read_text())
    assert marker == {"episodes_total": 20, "checkpoint_sha256": "ab" * 32, "raw_written": True}
    agg = json.loads((run_dir / "scenario_metrics.json").read_text())["aggregate"]
    assert agg["suite"]["episodes_total"] == 20 and agg["suite"]["n_by_hold"] == {lab: 2 for lab in SUITE_LABELS}
    assert "by category" in (run_dir / "summary.md").read_text()

    # A legacy (world) row: no marker, no suite block -- the files it always wrote, in the same order.
    legacy = _world_scenario(episodes=2)
    comp = S.compile_schedule(legacy, dt=DT)
    meta = {"scenario_id": legacy.id, "task": legacy.task, "checkpoint": "", "checkpoint_sha256": None,
            "command_override_max_deviation": 0.0, "dt": DT, "num_envs": 2,
            "termination_reason": ["schedule_complete"] * 2, "n_steps_per_env": [comp.n_steps] * 2}
    order, run_dir = _write_run_order(tmp_path, monkeypatch, legacy, comp, meta, 2)
    assert "complete.json" not in order and not (run_dir / "complete.json").exists()
    assert order == ["run_meta.json", "episode_00.json", "episode_01.json", "scenario_metrics.json",
                     "summary.md", "episode_00.npz", "episode_01.npz"]
    assert "suite" not in json.loads((run_dir / "scenario_metrics.json").read_text())["aggregate"]


def test_legacy_rows_compile_byte_identically():
    """Every legacy row of every committed manifest compiles to EXACTLY the step-0 snapshot
    (``fixtures/gait_eval_compile_snapshot.json``, captured from the unedited code). Never regenerate
    the fixture to make this pass: a diff here means a forward/sideways/legacy-omni schedule moved."""
    if str(FIXTURES_DIR) not in sys.path:
        sys.path.insert(0, str(FIXTURES_DIR))
    import gen_gait_eval_compile_snapshot as GEN  # noqa: PLC0415

    snapshot = json.loads(GEN.FIXTURE_PATH.read_text())
    assert [p.resolve() for p in GEN.MANIFESTS] == [p.resolve() for p in _committed_manifests()]
    current: dict[str, dict] = {}
    for manifest in GEN.MANIFESTS:
        for sc in S.load_manifest(manifest)[0]:
            if S.uses_waves(sc):
                continue  # the suite compiles per wave; it is not a legacy row
            comp = S.compile_schedule(sc, dt=GEN.DT, num_envs=sc.episodes)
            current[f"{manifest.name}::{sc.id}"] = {
                "n_steps": comp.n_steps,
                "hold_labels": list(comp.hold_labels),
                "arrays": {f: GEN._array_identity(getattr(comp, f)) for f in GEN.ARRAY_FIELDS},
            }
    # The same row set (the four legacy omni rows included) and the same bytes, row by row.
    assert set(current) == set(snapshot)
    assert len(snapshot) == 50
    assert {k for k in snapshot if k.startswith("scenarios_omni.yaml::")} == {
        "scenarios_omni.yaml::slow__omni", "scenarios_omni.yaml::light__omni",
        "scenarios_omni.yaml::ring__omni", "scenarios_omni.yaml::curriculum__omni",
    }
    for key in sorted(snapshot):
        assert current[key] == snapshot[key], key

"""Menu option 9 (python -m firmware): the key loop must not break a calibration run."""

import math
from unittest.mock import Mock

import pytest

pytest.importorskip("serial")

from firmware import __main__ as menu
from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.krabby_mcu import JOINT_GROUP_NAMES, KrabbyMCUSDK


class _FakeMcu:
    def __init__(self):
        self.latest_cal_line = None
        self.cal_line_count = 0
        self.lines = []

    def say(self, line):
        self.latest_cal_line = line
        self.cal_line_count += 1
        self.lines.append(line)

    def recent_cal_lines(self, limit=None):
        return list(self.lines if limit is None else self.lines[-limit:])


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _watch():
    mcu = _FakeMcu()
    clock = _Clock()
    return menu.CalibrationWatch(mcu, clock=clock), mcu, clock


class TestCalibrationWatch:
    def test_inactive_until_started(self):
        watch, _, _ = _watch()
        assert not watch.active
        assert watch.poll() is None

    def test_active_after_start(self):
        watch, _, _ = _watch()
        watch.start()
        assert watch.active

    @pytest.mark.parametrize(
        "final",
        [
            "CAL ROBOT DONE",
            "CAL ROBOT FAIL LEFT: no CAL report for 135 s",
            "CAL ROBOT ABORTED H",
            "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150",
        ],
    )
    def test_ends_on_robot_end_line(self, final):
        watch, mcu, _ = _watch()
        watch.start()
        mcu.say("CAL ROBOT START pwm=80 passes=SEEK,SWEEP")
        assert watch.poll() is None
        assert watch.active
        mcu.say(final)
        assert watch.poll() == final
        assert not watch.active
        assert watch.poll() is None

    def test_robot_end_line_followed_by_a_follower_line_still_ends_the_run(self):
        # The leader prints "CAL ROBOT FAIL" after sending H; a still-running
        # follower answers "CAL LEFT ABORTED H" right after, within one poll.
        watch, mcu, _ = _watch()
        watch.start()
        mcu.say("CAL ROBOT START pwm=80 passes=SEEK,SWEEP")
        assert watch.poll() is None
        mcu.say("CAL ROBOT FAIL LEFT: no CAL report for 135 s")
        mcu.say("CAL LEFT ABORTED H")
        assert watch.poll() == "CAL ROBOT FAIL LEFT: no CAL report for 135 s"
        assert not watch.active

    def test_board_lines_do_not_end_the_run(self):
        watch, mcu, clock = _watch()
        watch.start()
        for line in (
            "CAL FRONT PASS SEEK pwm=80",
            "CAL FRONT FAIL FLKL pot not connected",
            "CAL LEFT PASS_DONE SWEEP saved",
        ):
            mcu.say(line)
            assert watch.poll() is None
        assert watch.active

    def test_busy_keeps_waiting(self):
        watch, mcu, _ = _watch()
        watch.start()
        mcu.say("CAL ROBOT BUSY")
        assert watch.poll() is None
        assert watch.active

    def test_long_silence_after_progress_does_not_end_the_run(self):
        # A single move can take 120 s with no line; only "no reply at all" times out.
        watch, mcu, clock = _watch()
        watch.start()
        mcu.say("CAL ROBOT START pwm=80 passes=SEEK,SWEEP")
        clock.now += 600
        assert watch.poll() is None
        assert watch.active

    def test_no_reply_at_all_resumes_after_timeout(self):
        watch, _, clock = _watch()
        watch.start()
        clock.now += menu.CalibrationWatch.NO_REPLY_S - 1
        assert watch.poll() is None
        clock.now += 2
        message = watch.poll()
        assert message is not None and "no CAL reply" in message
        assert not watch.active

    def test_old_lines_do_not_count_as_a_reply(self):
        watch, mcu, clock = _watch()
        mcu.say("CAL ROBOT DONE")  # left over from an earlier run
        watch.start()
        assert watch.poll() is None  # stale terminal line is not this run's end
        clock.now += menu.CalibrationWatch.NO_REPLY_S + 1
        assert watch.poll() is not None

    def test_stop(self):
        watch, _, _ = _watch()
        watch.start()
        watch.stop()
        assert not watch.active


class _FakeMenuMcu:
    """Stands in for KrabbyMCUSDK in menu.main(); records what the key loop sends."""

    def __init__(self):
        self.latest_cal_line = None
        self.cal_line_count = 0
        self.lines = []
        self.events = []  # ("cal", cmd) / ("jog", frame) / ("hold",) / ("T", targets)
        self.neutral = ({}, [])  # what neutral_targets() answers: (targets, skipped)

    def say(self, line):
        self.latest_cal_line = line
        self.cal_line_count += 1
        self.lines.append(line)

    def recent_cal_lines(self, limit=None):
        return list(self.lines if limit is None else self.lines[-limit:])

    def connect(self):
        return True

    def close(self):
        pass

    def send_command_calibrate(self, pwm=None, passes=None):
        self.events.append(("cal", pwm, passes))
        return "C"

    def neutral_targets(self, value=0.5):
        self.events.append(("neutral_targets", value))
        return self.neutral

    def send_command_joints(self, targets):
        self.events.append(("T", dict(targets)))

    def send_commands_jog(self, frame):
        self.events.append(("jog", dict(frame)))

    def send_command_joints_hold(self):
        self.events.append(("hold",))

    def jog_frames(self):
        return [e[1] for e in self.events if e[0] == "jog"]


def _run_menu(monkeypatch, steps, initial_pressed=(), mcu=None):
    """Run menu.main() headlessly.

    ``steps`` is a list of callables(mcu, pressed); one runs each time the loop sleeps,
    so step N edits the world seen by loop iteration N+1. The loop quits when steps run
    out. Returns the fake MCU.
    """
    mcu = mcu if mcu is not None else _FakeMenuMcu()
    pressed = set(initial_pressed)
    script = iter(steps)

    def fake_sleep(_seconds):
        step = next(script, None)
        if step is None:
            monkeypatch.setattr(menu, "_quit", True)
        else:
            step(mcu, pressed)

    monkeypatch.setattr(menu, "KrabbyMCUSDK", lambda: mcu)
    monkeypatch.setattr(menu, "_read_keys", lambda: None)
    monkeypatch.setattr(menu, "is_pressed", lambda key: key in pressed)
    monkeypatch.setattr(menu, "_quit", False)
    monkeypatch.setattr(menu.time, "sleep", fake_sleep)
    monkeypatch.setattr(menu.sys, "argv", ["krabby-firmware"])
    monkeypatch.setattr(menu.sys, "platform", "win32")  # skip termios setup
    menu.main()
    return mcu


class TestMenuLoopDuringCalibration:
    def test_no_jog_frames_while_a_run_is_active(self, monkeypatch):
        steps = [
            lambda m, p: (p.clear(), m.say("CAL ROBOT START pwm=80 passes=SEEK,SWEEP")),
            lambda m, p: m.say("CAL FRONT MOVE FLKL curl"),
            lambda m, p: m.say("CAL FRONT END FLKL curl pot=1015 travel=0"),
            lambda m, p: m.say("CAL ROBOT DONE"),
            lambda m, p: None,  # the iteration that sees DONE
            lambda m, p: None,  # first iteration after the run: jog frames resume
        ]
        mcu = _run_menu(monkeypatch, steps, initial_pressed={"9"})
        assert mcu.events[0] == ("cal", None, None)  # bare C = default speed
        # Only the two iterations after DONE send (all-zero) frames; none while it ran.
        frames = mcu.jog_frames()
        assert len(frames) == 2
        assert all(all(v == 0 for v in f.values()) for f in frames)
        assert all(e[0] == "jog" for e in mcu.events[1:])
        assert ("hold",) not in mcu.events

    def test_run_that_never_answers_resumes_jog_control(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(menu.time, "monotonic", clock)

        def tick(m, p):
            clock.now += 4.0

        mcu = _run_menu(
            monkeypatch,
            [lambda m, p: p.clear()] + [tick] * 6,
            initial_pressed={"9"},
        )
        assert mcu.jog_frames()  # resumed after the 10 s no-reply timeout

    def test_jog_key_takes_over_and_sends_its_frame(self, monkeypatch):
        steps = [
            lambda m, p: (p.clear(), m.say("CAL ROBOT START pwm=80 passes=SEEK,SWEEP")),
            lambda m, p: p.add("q"),  # Extend key for FLKL
            lambda m, p: None,
        ]
        mcu = _run_menu(monkeypatch, steps, initial_pressed={"9"})
        frames = mcu.jog_frames()
        assert frames, "jog key pressed during a run must reach the firmware"
        assert frames[0]["FLKL"] == +menu.JOG_PWM

    def test_esc_during_a_run_sends_hold(self, monkeypatch):
        steps = [
            lambda m, p: (p.clear(), m.say("CAL ROBOT START pwm=80 passes=SEEK,SWEEP")),
            lambda m, p: None,
        ]
        mcu = _run_menu(monkeypatch, steps, initial_pressed={"9"})
        assert mcu.events[-1] == ("hold",)
        assert mcu.jog_frames() == []

    def test_esc_without_a_run_sends_no_hold(self, monkeypatch):
        mcu = _run_menu(monkeypatch, [lambda m, p: None])
        assert ("hold",) not in mcu.events


class TestMenuJogDirection:
    """Menu keys follow the firmware convention: +PWM extends, -PWM retracts."""

    def test_extend_key_sends_positive_pwm(self, monkeypatch):
        mcu = _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"q"})
        assert mcu.jog_frames()[0]["FLKL"] == +menu.JOG_PWM

    def test_retract_key_sends_negative_pwm(self, monkeypatch):
        mcu = _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"a"})
        assert mcu.jog_frames()[0]["FLKL"] == -menu.JOG_PWM


ALL_NAMES = [n for _, names in JOINT_GROUP_NAMES for n in names]
HIPS_AND_KNEES = [n for n in ALL_NAMES if n.endswith(("HL", "KL"))]
YAWS = [n for n in ALL_NAMES if n.endswith("HY")]


class _SdkBackedMcu(_FakeMenuMcu):
    """Fake menu MCU whose neutral_targets() is the real SDK logic over set telemetry."""

    def __init__(self, positions, states=None):
        super().__init__()
        self._sdk = KrabbyMCUSDK(target=Mock())
        for name, pos in positions.items():
            self._sdk.joints[name] = JointTelemetry(
                name=name, pos=pos, pot=500, current=0, en=(0, 0), pwm=(0, 0), saf=0
            )
        self._sdk.joint_sensor_state.update(states or {})

    def neutral_targets(self, value=0.5):
        self.events.append(("neutral_targets", value))
        return self._sdk.neutral_targets(value)

    def sensor_states(self):
        return self._sdk.sensor_states()


def _neutral_sends(mcu):
    return [e[1] for e in mcu.events if e[0] == "T"]


class TestMenuNeutralKey:
    """Key 0: Neutral goes to hips and knees that have a position, never to a yaw."""

    def test_sends_exactly_the_ready_targets(self, monkeypatch):
        mcu = _FakeMenuMcu()
        mcu.neutral = ({"FLHL": 0.5, "FLKL": 0.5}, ["FRHL"])
        _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"0"}, mcu=mcu)
        assert _neutral_sends(mcu)
        assert all(sent == {"FLHL": 0.5, "FLKL": 0.5} for sent in _neutral_sends(mcu))
        assert ("neutral_targets", 0.5) in mcu.events

    def test_all_ready_never_includes_a_yaw(self, monkeypatch):
        mcu = _SdkBackedMcu({n: 0.4 for n in ALL_NAMES}, {n: "E" for n in HIPS_AND_KNEES})
        _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"0"}, mcu=mcu)
        sent = _neutral_sends(mcu)[0]
        assert set(sent) == set(HIPS_AND_KNEES)
        assert not set(sent) & set(YAWS)
        assert set(sent.values()) == {0.5}

    def test_unhomed_encoder_joints_are_skipped_and_named(self, monkeypatch, capsys):
        states = {n: "E" for n in HIPS_AND_KNEES}
        states.update({"FLHL": "e", "RRKL": "x"})
        mcu = _SdkBackedMcu({n: 0.4 for n in ALL_NAMES}, states)
        _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"0"}, mcu=mcu)
        sent = _neutral_sends(mcu)[0]
        assert "FLHL" not in sent and "RRKL" not in sent
        assert len(sent) == 10
        out = capsys.readouterr().out
        assert "Neutral sent to 10 joints; no position, skipped: FLHL, RRKL" in out

    def test_no_position_anywhere_sends_nothing_and_says_so(self, monkeypatch, capsys):
        mcu = _SdkBackedMcu({n: math.nan for n in ALL_NAMES})
        _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"0"}, mcu=mcu)
        assert _neutral_sends(mcu) == []
        assert "no hip or knee has a position yet" in capsys.readouterr().out

    def test_unhomed_after_power_up_is_sent_to_curl_only_not_to_9(self, monkeypatch, capsys):
        # R1-21: homing after a power-up needs only Curl only; key 9 sweeps and saves.
        mcu = _SdkBackedMcu(
            {n: math.nan for n in ALL_NAMES}, {n: "e" for n in HIPS_AND_KNEES}
        )
        _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"0"}, mcu=mcu)
        out = capsys.readouterr().out
        assert menu.NEUTRAL_NEEDS_CURL_ONLY in out
        assert menu.NEUTRAL_NEEDS_FULL_RUN not in out
        assert ("cal", None, None) not in mcu.events

    def test_all_ready_prints_no_note(self, monkeypatch, capsys):
        mcu = _FakeMenuMcu()
        mcu.neutral = ({n: 0.5 for n in HIPS_AND_KNEES}, [])
        _run_menu(monkeypatch, [lambda m, p: None], initial_pressed={"0"}, mcu=mcu)
        out = capsys.readouterr().out
        assert "skipped" not in out
        assert "no hip or knee has a position" not in out
        assert len(_neutral_sends(mcu)[0]) == 12

    def test_holding_the_key_prints_the_note_once(self, monkeypatch, capsys):
        mcu = _FakeMenuMcu()
        mcu.neutral = ({"FLHL": 0.5}, ["FLKL"])
        steps = [lambda m, p: None] * 3
        _run_menu(monkeypatch, steps, initial_pressed={"0"}, mcu=mcu)
        assert len(_neutral_sends(mcu)) == 4  # sent on every loop while held
        out = capsys.readouterr().out
        assert out.count("Neutral sent to 1 joints; no position, skipped: FLKL") == 1

    def test_pressing_again_after_a_release_prints_again(self, monkeypatch, capsys):
        mcu = _FakeMenuMcu()
        mcu.neutral = ({"FLHL": 0.5}, ["FLKL"])
        steps = [lambda m, p: p.clear(), lambda m, p: p.add("0")]
        _run_menu(monkeypatch, steps, initial_pressed={"0"}, mcu=mcu)
        out = capsys.readouterr().out
        assert out.count("Neutral sent to 1 joints; no position, skipped: FLKL") == 2

    def test_help_text_describes_the_new_key_and_the_nudge(self, monkeypatch, capsys):
        _run_menu(monkeypatch, [lambda m, p: None])
        out = capsys.readouterr().out
        assert "0: Neutral (0.5)" in out
        assert "never yaw" in out
        assert "a hip or knee" in out and "nudged out a little and straight back" in out
        assert "off its curl end" not in out  # no knee jog before 9 (knee nudge default)
        assert "stand" in out

    def test_help_does_not_send_power_up_homing_to_key_9(self, monkeypatch, capsys):
        # R1-21: key 9 is the full SEEK + SWEEP run that rewrites EEPROM.
        _run_menu(monkeypatch, [lambda m, p: None])
        out = capsys.readouterr().out
        assert "need 9 after every power-up" not in out
        assert "homed after every power-up with" in out
        assert "Calibrate > Curl only in the GUI" in out
        assert "9 is the FULL calibration" in out
        assert "Run 9 only with the robot on its stand" in out

    def test_help_describes_the_fixed_hip_sweep_and_flashing(self, monkeypatch, capsys):
        _run_menu(monkeypatch, [lambda m, p: None])
        out = " ".join(capsys.readouterr().out.split())
        assert "each knee fully out and back with its hip curled" in out
        assert "each hip out about 70% and back (a curled tibia meets the body" in out
        assert "end to end" not in out
        assert "Flash all three boards together with the same firmware." in out

    def test_calibrate_warning_mentions_the_nudge_and_the_stand(self, monkeypatch, capsys):
        steps = [lambda m, p: p.clear(), lambda m, p: m.say("CAL ROBOT DONE")]
        _run_menu(monkeypatch, steps, initial_pressed={"9"})
        out = capsys.readouterr().out
        assert "robot on its stand" in out
        assert "A hip or knee resting fully curled" in out
        assert "nudged out a little and straight back so its sensor can be checked" in out
        assert "(at most 24 hall counts for a hip, 12 for a knee;" in out
        assert "springs out a little more, to about 20" in out
        assert "Hips go out about 70% and back, knees fully out and back" in out
        assert "flash all three boards with the same firmware, or wait" in out
        assert "a few seconds after connecting" in out
        assert "A joint that still" in out and "shows no movement stops the run" in out
        assert "jogged" not in out and "off its curl end" not in out
        assert " mm" not in out


class TestNeutralMessage:
    def test_nothing_sent(self):
        assert menu.neutral_message(0, ["FLHL"]) == (
            "Neutral: no hip or knee has a position yet - run auto-calibrate (9) with "
            "the robot on its stand"
        )

    def test_nothing_sent_and_only_unhomed_points_to_curl_only(self):
        message = menu.neutral_message(0, ["FLHL", "FLKL"], {"FLHL": "e", "FLKL": "x"})
        assert message == menu.NEUTRAL_NEEDS_CURL_ONLY
        assert "Curl only" in message and "in the GUI" in message
        assert "run auto-calibrate" not in message

    @pytest.mark.parametrize("state", ["s", "?"])
    def test_nothing_sent_with_no_stroke_or_unknown_needs_the_full_run(self, state):
        message = menu.neutral_message(0, ["FLHL", "FLKL"], {"FLHL": "e", "FLKL": state})
        assert message == menu.NEUTRAL_NEEDS_FULL_RUN
        assert "on its stand" in message

    def test_sensor_states_helper_tolerates_missing_or_odd_accessors(self):
        assert menu._sensor_states(object()) == {}
        assert menu._sensor_states(Mock(sensor_states=lambda: Mock())) == {}
        assert menu._sensor_states(Mock(sensor_states=lambda: {"FLHL": "e"})) == {
            "FLHL": "e"
        }

    def test_some_skipped(self):
        assert menu.neutral_message(11, ["MLKL"]) == (
            "Neutral sent to 11 joints; no position, skipped: MLKL"
        )

    def test_none_skipped(self):
        assert menu.neutral_message(12, []) == ""

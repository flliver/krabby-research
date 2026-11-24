"""arduino.ino's calibration wiring, checked in the sketch source itself.

The calibration logic is unit-tested in pure headers (leg_calibration.h,
robot_calibration.h, calibration_command_dispatch.h) and the host sim calls the
same dispatch, but the sketch's own loop() is compiled only for the Mega. These
checks fail if an edit drops one of the safety-relevant calls from the sketch:
the H/J/B/T aborts, follower-line supervision, the leader-boot H and the V guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKETCH = Path(__file__).resolve().parents[3] / "firmware" / "arduino" / "arduino.ino"


@pytest.fixture(scope="module")
def sketch() -> str:
    return SKETCH.read_text(encoding="utf-8")


def _function_body(source: str, signature: str) -> str:
    """Text of the brace-balanced body that follows the first `signature`."""
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def _command_branch(loop_body: str, command: str) -> str:
    """The loop()'s `cmdType == '<command>'` branch, up to the next branch."""
    match = re.search(rf"if \(cmdType == '{command}'\)", loop_body)
    assert match, f"no cmdType == '{command}' branch in loop()"
    rest = loop_body[match.end() :]
    following = re.search(r"else if \(cmdType == '|\n        else\n", rest)
    return rest[: following.start()] if following else rest


@pytest.mark.parametrize("command", ["T", "B", "J", "H"])
def test_stop_commands_abort_calibration_before_they_run(sketch, command):
    branch = _command_branch(_function_body(sketch, "void loop()"), command)
    abort_call = f"abortCalibrationForCommand('{command}');"
    assert abort_call in branch
    # The abort comes before the command is applied or forwarded.
    abort_at = branch.index(abort_call)
    for later in ("applyCommands", "handleJog", "holdAll", "leftSerial->"):
        if later in branch:
            assert abort_at < branch.index(later), f"{later} runs before the abort"


def test_abort_helper_stops_the_robot_and_this_board(sketch):
    body = _function_body(sketch, "static void abortCalibrationForCommand(char command)")
    assert (
        "abortBoardCalibrationForCommand(robotCalibration, robotCalibrationHooks, "
        "*actuatorManager, command);" in body
    )


def test_calibration_command_uses_the_shared_dispatch(sketch):
    body = _function_body(sketch, "static void handleActuatorCalibrationCommand(")
    assert "dispatchActuatorCalibrationCommand(currentRole, kind, request, out," in body
    assert "robotCalibration, robotCalibrationHooks, *actuatorManager);" in body


def test_every_forwarded_follower_line_reaches_the_robot_calibration(sketch):
    body = _function_body(sketch, "void forwardFullLines(")
    forward_at = body.index("to->println(partial);")
    note_at = body.index("noteFollowerLine(boardRole, partial);")
    assert forward_at < note_at
    note = _function_body(sketch, "static void noteFollowerLine(BoardRole role, const char *line)")
    assert "robotCalibration.onFollowerLine(robotCalibrationHooks, role, line);" in note


def test_loop_polls_the_robot_calibration(sketch):
    body = _function_body(sketch, "void loop()")
    assert "robotCalibration.update(robotCalibrationHooks);" in body
    assert body.index("actuatorManager->updateAll();") < body.index(
        "robotCalibration.update(robotCalibrationHooks);"
    )


def test_leader_boot_holds_the_followers(sketch):
    body = _function_body(sketch, "void setup()")
    leader = body[body.index("if (currentRole == ROLE_FRONT || currentRole == ROLE_UNKNOWN)") :]
    assert 'if (leftSerial)  leftSerial->println("H");' in leader
    assert 'if (rightSerial) rightSerial->println("H");' in leader
    assert body.index("determineRole();") < body.index('leftSerial->println("H");')


def test_version_skips_the_follower_query_while_calibrating(sketch):
    branch = _command_branch(_function_body(sketch, "void loop()"), "V")
    assert "const bool queryFollowers = shouldQueryFollowerVersions(robotCalibration);" in branch
    assert "if (leftSerial && queryFollowers)" in branch
    assert "if (rightSerial && queryFollowers)" in branch
    assert branch.count("readVerLine(") == 2


def test_leader_labels_its_lines_with_the_elected_role(sketch):
    hooks = _function_body(sketch, "class SketchRobotCalibrationHooks final : public RobotCalibrationHooks")
    assert "return boardRoleLabel(currentRole);" in hooks
    assert "actuatorManager->setCalibrationOutput(mainSerial, currentRole);" in sketch

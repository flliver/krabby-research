#pragma once

#include <stdint.h>

#include "../cli/actuator_calibration_command.h"
#include "../controller/board_role.h"
#include "calibration_text_sink.h"
#include "robot_calibration.h"

// The sketch's calibration command glue, shared by arduino.ino and the host
// sim so that tests exercise the code the robot runs. `Board` is the board's
// own calibration: anything with isCalibrating(), startCalibration(passes, pwm)
// and abortCalibration(cause), i.e. ActuatorManager.

// FRONT leads; UNKNOWN is a leader on a single-board bench.
inline bool isCalibrationLeaderRole(BoardRole role)
{
    return role == ROLE_FRONT || role == ROLE_UNKNOWN;
}

// H, J, B and T stop calibration before they run: the leader's robot sequence
// (it sends H to the followers first), then this board's own pass. No-op when
// idle (a follower's robot sequence is always idle).
template <typename Board>
inline void abortBoardCalibrationForCommand(
    RobotCalibration &robot,
    RobotCalibrationHooks &hooks,
    Board &board,
    char command)
{
    robot.abort(hooks, command);
    board.abortCalibration(command);
}

// C [ACTUATOR] [SEEK|SWEEP] [pwm] on a board with role `role`, after parsing.
// Invalid: "CAL ROBOT ERROR <usage>" on a leader, "CAL <ROLE> ERROR <usage>"
// on a follower; nothing moves. Run: the leader starts the whole robot (which
// answers "CAL ROBOT BUSY" while one runs); a follower runs the passes on its
// own six actuators, or answers "CAL <ROLE> BUSY" while it is calibrating.
// Followers never forward calibration commands. PowerSensor is not handled here.
template <typename Board>
inline void dispatchActuatorCalibrationCommand(
    BoardRole role,
    ActuatorCalibrationCommandKind kind,
    const ActuatorCalibrationRequest &request,
    CalibrationTextSink &out,
    RobotCalibration &robot,
    RobotCalibrationHooks &hooks,
    Board &board)
{
    const bool isLeader = isCalibrationLeaderRole(role);
    if (kind == ActuatorCalibrationCommandKind::Invalid)
    {
        out.printFlash(KRABBY_FLASH_LITERAL("CAL "));
        if (isLeader)
            out.printFlash(KRABBY_FLASH_LITERAL("ROBOT"));
        else
            out.printText(boardRoleLabel(role));
        out.printFlash(KRABBY_FLASH_LITERAL(" ERROR "));
        printActuatorCalibrationUsage(out);
        out.endLine();
        return;
    }
    if (kind != ActuatorCalibrationCommandKind::Run)
        return;

    if (isLeader)
    {
        robot.start(hooks, request.passes, request.pwm);
        return;
    }
    if (board.isCalibrating())
    {
        out.printFlash(KRABBY_FLASH_LITERAL("CAL "));
        out.printText(boardRoleLabel(role));
        out.printFlash(KRABBY_FLASH_LITERAL(" BUSY"));
        out.endLine();
        return;
    }
    board.startCalibration(request.passes, request.pwm);
}

// Leader V: ask the followers for their versions only while no robot
// calibration runs. Asking blocks the loop for up to 300 ms per follower and
// reads the follower links outside the line forwarding, which can split or
// drop a follower's CAL line (and with it a PASS_DONE the robot waits for);
// during calibration the leader answers with its own version and "-" for each
// follower instead.
inline bool shouldQueryFollowerVersions(const RobotCalibration &robot)
{
    return !robot.isActive();
}

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../controller/board_role.h"
#include "calibration_constants.h"
#include "calibration_text_sink.h"
#include "leg_calibration.h"

// Leader-side robot calibration: runs each requested pass (SEEK, then SWEEP)
// on FRONT (the leader's own actuators), then LEFT, then RIGHT, strictly one
// board at a time, so only one joint on the robot moves at any instant. A
// follower pass is started with "C SEEK <pwm>" / "C SWEEP <pwm>" on its link
// and finishes when the leader forwards that follower's
// "CAL <ROLE> PASS_DONE <pass>" line (see onFollowerLine()).
//
// Non-blocking: call update() every loop and onFollowerLine() for every
// complete line forwarded from a follower.
//
// Mixed firmware: on a quadrature leader the robot calibration refuses to
// start while a linked follower has sent no "SENS <ROLE> ..." line since the
// leader booted. Only this firmware on a quadrature build prints SENS (every
// ACTUATOR_SENS_PERIOD_MS), so a silent follower runs other firmware (an older
// build, or a build without quadrature halls) that would treat hall joints as
// pots, or has only just been connected.

// hall_hw.h: true when this board counts signed quadrature hall edges.
bool hallHwHasQuadrature();

// What the robot calibration drives: the leader's own pass and its links.
class RobotCalibrationHooks
{
public:
    virtual uint32_t nowMs() = 0;
    // The host link (leader mainSerial).
    virtual CalibrationTextSink &out() = 0;
    // boardRoleLabel() of the leader: FRONT, or UNKWN on a single-board bench.
    virtual const char *localRoleLabel() = 0;
    // ROLE_LEFT / ROLE_RIGHT link; nullptr when that follower is absent.
    virtual CalibrationTextSink *followerLink(BoardRole role) = 0;
    // Start one pass (a CAL_PASS_* bit) on the leader; false if it could not start.
    virtual bool startLocalPass(uint8_t pass, int pwm) = 0;
    virtual bool isLocalRunning() = 0;
    // Outcome of the last local pass.
    virtual LegCalibrationOutcome localOutcome() = 0;
    // LegCalibration::printFailureReason() of that pass.
    virtual void printLocalFailure(CalibrationTextSink &out) = 0;
    // Abort a running local pass (it emits its own ABORTED line).
    virtual void abortLocal(char cause) = 0;
    // Every local actuator off, no position target.
    virtual void stopLocalMotors() = 0;
    // The leader counts quadrature hall edges (and so do its followers' builds
    // of this firmware, which print SENS lines).
    virtual bool hasQuadratureHall() { return hallHwHasQuadrature(); }

protected:
    ~RobotCalibrationHooks() = default;
};

enum class RobotCalibrationPhase : uint8_t
{
    Idle,
    Local,     // leader's own pass running
    Follower,  // waiting for a follower's PASS_DONE
};

// Boards in calibration order: 0 = the leader itself (FRONT), 1 = LEFT, 2 = RIGHT.
static constexpr uint8_t ROBOT_CALIBRATION_BOARD_COUNT = 3;

inline BoardRole robotCalibrationBoardRole(uint8_t board)
{
    if (board == 1)
        return ROLE_LEFT;
    if (board == 2)
        return ROLE_RIGHT;
    return ROLE_FRONT;
}

// Text after `prefix` when `text` starts with it, else nullptr. The prefix is a
// KRABBY_FLASH_LITERAL (flash on AVR).
inline const char *calibrationTextAfterFlashPrefix(const char *text, const char *flashPrefix)
{
    for (;; ++text, ++flashPrefix)
    {
        const char expected = KRABBY_FLASH_CHAR(flashPrefix);
        if (expected == '\0')
            return text;
        if (*text != expected)
            return nullptr;
    }
}

// As calibrationTextAfterFlashPrefix, for a prefix held in RAM.
inline const char *calibrationTextAfterPrefix(const char *text, const char *prefix)
{
    for (;; ++text, ++prefix)
    {
        if (*prefix == '\0')
            return text;
        if (*text != *prefix)
            return nullptr;
    }
}

// Event text of a "CAL <roleLabel> <event...>" line, else nullptr.
inline const char *calibrationLineEvent(const char *line, const char *roleLabel)
{
    const char *cursor = calibrationTextAfterFlashPrefix(line, KRABBY_FLASH_LITERAL("CAL "));
    if (cursor != nullptr)
        cursor = calibrationTextAfterPrefix(cursor, roleLabel);
    if (cursor != nullptr)
        cursor = calibrationTextAfterFlashPrefix(cursor, KRABBY_FLASH_LITERAL(" "));
    return cursor;
}

class RobotCalibration
{
public:
    RobotCalibration() { reset(); }

    // Silently returns to Idle without touching any board, and forgets the
    // followers' SENS lines (host-sim reset: the boards boot again).
    void reset()
    {
        phase_ = RobotCalibrationPhase::Idle;
        passes_ = 0;
        pass_ = 0;
        board_ = 0;
        pwm_ = 0;
        sensSeenMask_ = 0;
        lastFollowerLineMs_ = 0;
    }

    // Starts a robot calibration. While one runs, emits "CAL ROBOT BUSY" and
    // returns false; an invalid request emits "CAL ROBOT ERROR invalid request"
    // and returns false. On a quadrature leader, a linked follower (LEFT first)
    // that has sent no SENS line since boot halts the robot with "CAL ROBOT FAIL
    // <role>: no SENS: flash all three boards with the same firmware, or wait a
    // few seconds after connecting" and returns false: nothing moves. Otherwise
    // emits "CAL ROBOT START pwm=<p> passes=<SEEK|SWEEP|SEEK,SWEEP>", sends H to
    // every follower link so nothing holds position elsewhere, starts the first
    // pass on the leader and returns true.
    bool start(RobotCalibrationHooks &hooks, uint8_t passes, int pwm)
    {
        CalibrationTextSink &out = hooks.out();
        if (phase_ != RobotCalibrationPhase::Idle)
        {
            printRobotPrefix(out);
            out.printFlash(KRABBY_FLASH_LITERAL("BUSY"));
            out.endLine();
            return false;
        }
        if (!isValidCalibrationRequest(passes, pwm))
        {
            printRobotPrefix(out);
            out.printFlash(KRABBY_FLASH_LITERAL("ERROR invalid request"));
            out.endLine();
            return false;
        }
        const BoardRole silent = followerWithoutSensLine(hooks);
        if (silent != ROLE_UNKNOWN)
        {
            CalibrationTextSink &failure = beginRobotFailure(hooks, boardRoleLabel(silent));
            failure.printFlash(KRABBY_FLASH_LITERAL(
                "no SENS: flash all three boards with the same firmware, "
                "or wait a few seconds after connecting"));
            failure.endLine();
            return false;
        }

        passes_ = passes;
        pwm_ = static_cast<uint8_t>(pwm);
        printRobotPrefix(out);
        out.printFlash(KRABBY_FLASH_LITERAL("START pwm="));
        out.printNumber(pwm_);
        out.printFlash(KRABBY_FLASH_LITERAL(" passes="));
        if (passes_ == CAL_PASS_ALL)
            out.printFlash(KRABBY_FLASH_LITERAL("SEEK,SWEEP"));
        else if (passes_ == CAL_PASS_SEEK)
            out.printFlash(KRABBY_FLASH_LITERAL("SEEK"));
        else
            out.printFlash(KRABBY_FLASH_LITERAL("SWEEP"));
        out.endLine();

        sendHoldToFollowers(hooks);
        pass_ = (passes_ & CAL_PASS_SEEK) != 0 ? CAL_PASS_SEEK : CAL_PASS_SWEEP;
        board_ = 0;
        runStep(hooks);
        return true;
    }

    // Polls the leader's own pass and the follower silence watchdog.
    void update(RobotCalibrationHooks &hooks)
    {
        if (phase_ == RobotCalibrationPhase::Local)
        {
            if (hooks.isLocalRunning())
                return;
            if (hooks.localOutcome() == LegCalibrationOutcome::Completed)
            {
                ++board_;
                runStep(hooks);
                return;
            }
            CalibrationTextSink &out = beginRobotFailure(hooks, hooks.localRoleLabel());
            hooks.printLocalFailure(out);
            out.endLine();
            return;
        }

        if (phase_ == RobotCalibrationPhase::Follower &&
            hooks.nowMs() - lastFollowerLineMs_ >= CAL_FOLLOWER_SILENCE_MS)
        {
            CalibrationTextSink &out = beginRobotFailure(hooks, boardRoleLabel(currentBoard()));
            out.printFlash(KRABBY_FLASH_LITERAL("no CAL report for "));
            out.printNumber(static_cast<long>(CAL_FOLLOWER_SILENCE_MS / 1000UL));
            out.printFlash(KRABBY_FLASH_LITERAL(" s"));
            out.endLine();
        }
    }

    // Feed every complete line forwarded from follower `role` (after it has
    // been forwarded). A "SENS <role> ..." line marks that follower as running
    // this firmware (see start()). From the follower being waited on, each "CAL <role> ..."
    // line restarts the silence watchdog; PASS_DONE of the current pass
    // advances; FAIL, ABORTED, BUSY or ERROR fail the robot with
    // "CAL ROBOT FAIL <role>: <the follower's reason>" (at most
    // CAL_ROBOT_REASON_MAX_CHARS of it). A "CAL <role> MOVE" from any other
    // follower, or while no follower is being waited on, is a board moving
    // outside its turn (for example a pass still running after the leader
    // reset): it fails a running robot calibration, and when none is running
    // sends that follower H. Either way it emits
    // "CAL ROBOT FAIL <role>: MOVE outside its turn".
    void onFollowerLine(RobotCalibrationHooks &hooks, BoardRole role, const char *line)
    {
        if (line == nullptr)
            return;
        noteSensLine(role, line);
        const char *event = calibrationLineEvent(line, boardRoleLabel(role));
        if (event == nullptr)
            return;

        if (phase_ != RobotCalibrationPhase::Follower || role != currentBoard())
        {
            if (calibrationTextAfterFlashPrefix(event, KRABBY_FLASH_LITERAL("MOVE ")) == nullptr)
                return;
            CalibrationTextSink *out = nullptr;
            if (phase_ == RobotCalibrationPhase::Idle)
            {
                // No robot run to fail: stop just that board, leave the rest alone.
                CalibrationTextSink *link = hooks.followerLink(role);
                if (link != nullptr)
                {
                    link->printChar('H');
                    link->endLine();
                }
                out = &hooks.out();
                printRobotPrefix(*out);
                out->printFlash(KRABBY_FLASH_LITERAL("FAIL "));
                out->printText(boardRoleLabel(role));
                out->printFlash(KRABBY_FLASH_LITERAL(": "));
            }
            else
            {
                out = &beginRobotFailure(hooks, boardRoleLabel(role));
            }
            out->printFlash(KRABBY_FLASH_LITERAL("MOVE outside its turn"));
            out->endLine();
            return;
        }

        lastFollowerLineMs_ = hooks.nowMs();
        const char *rest = calibrationTextAfterFlashPrefix(event, KRABBY_FLASH_LITERAL("PASS_DONE "));
        if (rest != nullptr)
        {
            const char *passName = pass_ == CAL_PASS_SEEK
                ? KRABBY_FLASH_LITERAL("SEEK")
                : KRABBY_FLASH_LITERAL("SWEEP");
            if (calibrationTextAfterFlashPrefix(rest, passName) != nullptr)
            {
                ++board_;
                runStep(hooks);
            }
            return;
        }

        rest = calibrationTextAfterFlashPrefix(event, KRABBY_FLASH_LITERAL("FAIL "));
        if (rest == nullptr &&
            (calibrationTextAfterFlashPrefix(event, KRABBY_FLASH_LITERAL("ABORTED")) != nullptr ||
             calibrationTextAfterFlashPrefix(event, KRABBY_FLASH_LITERAL("BUSY")) != nullptr ||
             calibrationTextAfterFlashPrefix(event, KRABBY_FLASH_LITERAL("ERROR")) != nullptr))
        {
            rest = event;
        }
        if (rest == nullptr)
            return;

        CalibrationTextSink &out = beginRobotFailure(hooks, boardRoleLabel(role));
        for (uint8_t count = 0; count < CAL_ROBOT_REASON_MAX_CHARS && rest[count] != '\0'; ++count)
            out.printChar(rest[count]);
        out.endLine();
    }

    // Host command H/J/B/T while active: sends H to every follower link first,
    // aborts and stops the leader's own pass, then emits
    // "CAL ROBOT ABORTED <cause>". Returns false (and does nothing) when idle.
    bool abort(RobotCalibrationHooks &hooks, char cause)
    {
        if (phase_ == RobotCalibrationPhase::Idle)
            return false;
        sendHoldToFollowers(hooks);
        hooks.abortLocal(cause);
        hooks.stopLocalMotors();
        phase_ = RobotCalibrationPhase::Idle;
        CalibrationTextSink &out = hooks.out();
        printRobotPrefix(out);
        out.printFlash(KRABBY_FLASH_LITERAL("ABORTED "));
        out.printChar(cause);
        out.endLine();
        return true;
    }

    bool isActive() const { return phase_ != RobotCalibrationPhase::Idle; }
    RobotCalibrationPhase phase() const { return phase_; }
    // A "SENS <role> ..." line came from that follower since boot (or reset()).
    bool hasSensLineFrom(BoardRole role) const { return (sensSeenMask_ & sensBit(role)) != 0; }
    uint8_t passes() const { return passes_; }
    // Pass bit and board of the current (or last) step.
    uint8_t currentPass() const { return pass_; }
    BoardRole currentBoard() const { return robotCalibrationBoardRole(board_); }
    int pwm() const { return pwm_; }
    uint32_t lastFollowerLineMs() const { return lastFollowerLineMs_; }

private:
    static void printRobotPrefix(CalibrationTextSink &out)
    {
        out.printFlash(KRABBY_FLASH_LITERAL("CAL ROBOT "));
    }

    static void sendHoldToFollowers(RobotCalibrationHooks &hooks)
    {
        for (uint8_t board = 1; board < ROBOT_CALIBRATION_BOARD_COUNT; ++board)
        {
            CalibrationTextSink *link = hooks.followerLink(robotCalibrationBoardRole(board));
            if (link == nullptr)
                continue;
            link->printChar('H');
            link->endLine();
        }
    }

    // Halts the robot (H to followers, leader stopped, Idle) and prints
    // "CAL ROBOT FAIL <roleLabel>: "; the caller adds the reason and line end.
    CalibrationTextSink &beginRobotFailure(RobotCalibrationHooks &hooks, const char *roleLabel)
    {
        sendHoldToFollowers(hooks);
        hooks.stopLocalMotors();
        phase_ = RobotCalibrationPhase::Idle;
        CalibrationTextSink &out = hooks.out();
        printRobotPrefix(out);
        out.printFlash(KRABBY_FLASH_LITERAL("FAIL "));
        out.printText(roleLabel);
        out.printFlash(KRABBY_FLASH_LITERAL(": "));
        return out;
    }

    static uint8_t sensBit(BoardRole role) { return static_cast<uint8_t>(1U << role); }

    void noteSensLine(BoardRole role, const char *line)
    {
        const char *rest = calibrationTextAfterFlashPrefix(line, KRABBY_FLASH_LITERAL("SENS "));
        if (rest != nullptr)
            rest = calibrationTextAfterPrefix(rest, boardRoleLabel(role));
        if (rest != nullptr && *rest == ' ')
            sensSeenMask_ |= sensBit(role);
    }

    // On a quadrature leader, the first linked follower (LEFT, then RIGHT) that
    // has sent no SENS line since boot; ROLE_UNKNOWN when there is none, or the
    // leader has no quadrature halls (its own build prints no SENS lines).
    BoardRole followerWithoutSensLine(RobotCalibrationHooks &hooks) const
    {
        if (!hooks.hasQuadratureHall())
            return ROLE_UNKNOWN;
        for (uint8_t board = 1; board < ROBOT_CALIBRATION_BOARD_COUNT; ++board)
        {
            const BoardRole role = robotCalibrationBoardRole(board);
            if (hooks.followerLink(role) != nullptr && !hasSensLineFrom(role))
                return role;
        }
        return ROLE_UNKNOWN;
    }

    // A follower with no link while the other follower has one: the robot is
    // assembled but that board missed the role election. ROLE_UNKNOWN when both
    // links are there, or neither (a single board on the bench).
    static BoardRole partlyLinkedFollower(RobotCalibrationHooks &hooks)
    {
        const bool hasLeft = hooks.followerLink(ROLE_LEFT) != nullptr;
        const bool hasRight = hooks.followerLink(ROLE_RIGHT) != nullptr;
        if (hasLeft == hasRight)
            return ROLE_UNKNOWN;
        return hasLeft ? ROLE_RIGHT : ROLE_LEFT;
    }

    // Starts the step at (pass_, board_), skipping absent followers and moving
    // on to the next pass; emits "CAL ROBOT DONE" after the last one. A SWEEP
    // never starts while one follower has a link and the other has none: that
    // board was never curled, so the robot fails
    // "CAL ROBOT FAIL <role>: no link, no SWEEP until every board is linked".
    void runStep(RobotCalibrationHooks &hooks)
    {
        for (;;)
        {
            if (pass_ == CAL_PASS_SWEEP && board_ == 0)
            {
                const BoardRole missing = partlyLinkedFollower(hooks);
                if (missing != ROLE_UNKNOWN)
                {
                    CalibrationTextSink &out = beginRobotFailure(hooks, boardRoleLabel(missing));
                    out.printFlash(KRABBY_FLASH_LITERAL("no link, no SWEEP until every board is linked"));
                    out.endLine();
                    return;
                }
            }

            if (board_ >= ROBOT_CALIBRATION_BOARD_COUNT)
            {
                if (pass_ == CAL_PASS_SEEK && (passes_ & CAL_PASS_SWEEP) != 0)
                {
                    pass_ = CAL_PASS_SWEEP;
                    board_ = 0;
                    continue;
                }
                phase_ = RobotCalibrationPhase::Idle;
                CalibrationTextSink &out = hooks.out();
                printRobotPrefix(out);
                out.printFlash(KRABBY_FLASH_LITERAL("DONE"));
                out.endLine();
                return;
            }

            if (board_ == 0)
            {
                if (!hooks.startLocalPass(pass_, pwm_))
                {
                    CalibrationTextSink &out = beginRobotFailure(hooks, hooks.localRoleLabel());
                    out.printFlash(KRABBY_FLASH_LITERAL("could not start its pass"));
                    out.endLine();
                    return;
                }
                phase_ = RobotCalibrationPhase::Local;
                return;
            }

            const BoardRole role = currentBoard();
            CalibrationTextSink *link = hooks.followerLink(role);
            if (link == nullptr)
            {
                CalibrationTextSink &out = hooks.out();
                printRobotPrefix(out);
                out.printFlash(KRABBY_FLASH_LITERAL("SKIP "));
                out.printText(boardRoleLabel(role));
                out.printFlash(KRABBY_FLASH_LITERAL(" no link"));
                out.endLine();
                ++board_;
                continue;
            }

            link->printFlash(pass_ == CAL_PASS_SEEK
                ? KRABBY_FLASH_LITERAL("C SEEK ")
                : KRABBY_FLASH_LITERAL("C SWEEP "));
            link->printNumber(pwm_);
            link->endLine();
            lastFollowerLineMs_ = hooks.nowMs();
            phase_ = RobotCalibrationPhase::Follower;
            return;
        }
    }

    RobotCalibrationPhase phase_;
    uint8_t passes_;
    uint8_t pass_;
    uint8_t board_;
    uint8_t pwm_;
    uint8_t sensSeenMask_;  // bit (1 << role): a SENS line came from that follower
    uint32_t lastFollowerLineMs_;
};

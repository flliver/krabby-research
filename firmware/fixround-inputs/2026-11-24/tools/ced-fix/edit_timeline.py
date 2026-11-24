import sys

p = sys.argv[1]
s = open(p, newline='').read()
assert '\r\n' in s
s = s.replace('\r\n', '\n')


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# F3: the blind knee nudge bound, re-derived from the PWM 128 logs.
rep("""// A knee whose hall A is dead counts no edge, so only the 1 s cap stops its
// nudge. Re-derived for the knee nudge's PWM 128 cap (CAL_KNEE_NUDGE_MAX_PWM
// follows CAL_DEFAULT_PWM): after the fastest measured break-free (0.34 s) it
// ramps to 128 by 0.52 s and drives on to the cap at about 77 edges/s (the PWM
// 80 rate, 48 edges/s, scaled to 128; not measured on the robot), about 47
// edges plus a 1-edge coast (49 allows a tick), then springs out, so it may end
// up to 58 edges out before it curls back (and fails: no feedback). At PWM 80
// that was 36 / 45.
// A blind knee breaking free late (0.75 s) moves about 18 edges at 128.
constexpr double kSpecBlindKneeNudgeMaxEdges = 49.0;
""", """// A knee whose hall A is dead counts no edge, so only the 1 s cap stops its
// nudge. At the knee nudge's PWM 128 cap (CAL_KNEE_NUDGE_MAX_PWM follows
// CAL_DEFAULT_PWM), measured 2026-11-24: FLKL driven out of its curl end at 128
// had moved 72 edges 1.0 s after its drive started (a 0.64 s ramp; the
// firmware's is 0.52 s), the FR knees ran about 120-130 edges/s, and a knee
// coasted 3-9 edges after its stop. A blind knee that breaks free at once thus
// drives up to about 90 edges by the cap, 95 with its coast, and its spring-out
// (a curled knee's, up to 9) may add to that before it curls back (and fails:
// no feedback). At PWM 80 that was 36 / 45. Its hip is not proven curled: the
// knee nudge is the one move allowed before that.
constexpr double kSpecBlindKneeNudgeMaxEdges = 95.0;
""")
rep("""constexpr uint32_t kSpecSwitchStopMaxMs = 250;
constexpr uint32_t kSpecMechanicalStopMaxMs = 400;
constexpr uint32_t kSpecPressMaxMs = 1100;
""", """constexpr uint32_t kSpecSwitchStopMaxMs = 250;
constexpr uint32_t kSpecMechanicalStopMaxMs = 400;
constexpr uint32_t kSpecPressMaxMs = 1100;
// A drive that reached its end within 1.2 s of its start (it started 0-130
// edges from it: no running phase, or one learned in the end approach) stops
// at most this long after reaching it, by a signature or by the plain stall
// (no edge for 1 s), and the run goes on (never "stuck mid-stroke").
constexpr uint32_t kSpecEarlyEndStopMaxMs = 1150;
""")
rep("""    bool kneeTaper = false;      // knee slows to ~2 edges/s over its last 8 edges
""", """    bool kneeTaper = false;      // knee slows to ~2 edges/s over its last taperEdges edges
    // The taper's length in edges (kKneeTaperEdges; longer: a compliant end
    // approached over 0.3-0.5 s, its current rising, as the robot's at 120-150).
    double taperEdges = kKneeTaperEdges;
""")
rep("""        return kAtEndTolerance + (kneeTaper ? kKneeTaperEdges / edgesPerCount : 0.0);""",
    """        return kAtEndTolerance + (kneeTaper ? taperEdges / edgesPerCount : 0.0);""")
rep("""            if (previous > 0 && rod.kneeTaper && now - move.startMs >= 1200 &&
                (end - rod.position) * rod.edgesPerCount < kKneeTaperEdges)
                ++endChecks_[1];
            return;
        }
        if (reached - move.startMs < 1200) return;""", """            if (previous > 0 && rod.kneeTaper && now - move.startMs >= 1200 &&
                (end - rod.position) * rod.edgesPerCount < rod.taperEdges)
                ++endChecks_[1];
            return;
        }
        if (reached - move.startMs < 1200)
        {
            ++endChecks_[3];
            if (now - reached > kSpecEarlyEndStopMaxMs)
                violation(board, joint, "drove on past an end it reached within 1.2 s");
            return;
        }""")
rep("""            const double edgesLeft = (end - rod.position) * rod.edgesPerCount;
            if (edgesLeft < kKneeTaperEdges)
            {
                const double normal = velocity * rod.edgesPerCount * 1000.0;  // edges per s
                double rate = normal * edgesLeft / kKneeTaperEdges;""", """            const double edgesLeft = (end - rod.position) * rod.edgesPerCount;
            if (edgesLeft < rod.taperEdges)
            {
                const double normal = velocity * rod.edgesPerCount * 1000.0;  // edges per s
                double rate = normal * edgesLeft / rod.taperEdges;""")
rep("""            else if (rod.kneeTaper && isKnee(joint) && edgesLeft < kKneeTaperEdges)
                level = rod.isRun + (rod.isStall - rod.isRun) * (1.0 - edgesLeft / kKneeTaperEdges);""",
    """            else if (rod.kneeTaper && isKnee(joint) && edgesLeft < rod.taperEdges)
                level = rod.isRun + (rod.isStall - rod.isRun) * (1.0 - edgesLeft / rod.taperEdges);""")
rep("""    // End-detector checks made (drives that ran into a limit switch, into a
    // mechanical stop, and presses of a joint already at its end).
    int switchEndsChecked() const { return endChecks_[0]; }
    int stopEndsChecked() const { return endChecks_[1]; }
    int pressesChecked() const { return endChecks_[2]; }""", """    // End-detector checks made (drives that ran into a limit switch, into a
    // mechanical stop, presses of a joint already at its end, and drives that
    // reached their end within 1.2 s of their start).
    int switchEndsChecked() const { return endChecks_[0]; }
    int stopEndsChecked() const { return endChecks_[1]; }
    int pressesChecked() const { return endChecks_[2]; }
    int earlyEndsChecked() const { return endChecks_[3]; }""")
rep("""            pressAtStart_[board][joint] = isAtDriveEnd(board, joint, pwm);""",
    """            // At its end under the calibration PWM: a compliant knee parked
            // by a softer curl is not yet at the end this drive reaches.
            pressAtStart_[board][joint] =
                isAtDriveEnd(board, joint, requestedPwm > 0 ? sign * requestedPwm : pwm);""")
rep("""    int endChecks_[3] = {};""", """    int endChecks_[4] = {};""")
rep("""    // At a move's stop: a press of a joint already at its end lasts at most
    // kSpecPressMaxMs; a drive that ran (over 1.2 s before reaching its end, so
    // the end detector had armed) stops within kSpecSwitchStopMaxMs of a limit
    // switch and kSpecMechanicalStopMaxMs of a mechanical stop. A knee whose
    // drive stopped in its taper zone, short of the stop, stopped in time.""", """    // At a move's stop: a press of a joint already at its end lasts at most
    // kSpecPressMaxMs; a drive that ran (over 1.2 s before reaching its end, so
    // the end detector had armed) stops within kSpecSwitchStopMaxMs of a limit
    // switch and kSpecMechanicalStopMaxMs of a mechanical stop, and one that
    // reached its end sooner within kSpecEarlyEndStopMaxMs. A knee whose drive
    // stopped in its taper zone, short of the stop, stopped in time.""")
open(p, 'w', newline='').write(s.replace('\n', '\r\n'))
print('ok')

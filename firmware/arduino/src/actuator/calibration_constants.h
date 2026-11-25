#pragma once

#include <stdint.h>

#include "actuator_constants.h"

// Actuator auto-calibration tuning (C [SEEK|SWEEP] [pwm]). The per-board pass
// logic lives in leg_calibration.h and the leader's robot sequencing in
// robot_calibration.h. Every millisecond value is uint32_t: int is 16-bit on AVR.

// Calibration PWM magnitude: default and the accepted range for C <pwm>.
// 128 (the owner's choice, 2026-11-24): at 80 the FRKL knee could not break
// free of its curl end (nor at 100), FLKL and FRKL uncurls stalled short of
// their switch, and the FR knee's curl end showed only an 11% current rise. At
// 120 and 150 every end of the front legs showed its current signature.
static constexpr int CAL_DEFAULT_PWM = 128;
static constexpr int CAL_PWM_MIN = 40;
static constexpr int CAL_PWM_MAX = 150;

// Soft start: the magnitude climbs from 0 by this step every interval, never
// past the requested calibration PWM.
static constexpr int CAL_PWM_RAMP_STEP = 5;
static constexpr uint32_t CAL_PWM_RAMP_INTERVAL_MS = 20;

// Windowed end of travel (proven Pot joints, builds without quadrature halls,
// and the verification nudge's drive): after the ramp completes and the grace
// period passes, each window must advance the joint in the move's direction: a
// proven pot (smoothed reading, avgPot) by at least CAL_STALL_MIN_PROGRESS
// counts, a hall joint by at least CAL_HALL_STALL_MIN_EDGES edges. The first
// window that does not marks the end of travel. Hall joints on quadrature
// builds use the end detector below; its unarmed uncurl keeps the same
// CAL_START_GRACE_MS + CAL_STALL_WINDOW_MS break-free allowance.
static constexpr uint32_t CAL_START_GRACE_MS = 1500;
static constexpr uint32_t CAL_STALL_WINDOW_MS = 1000;
static constexpr int CAL_STALL_MIN_PROGRESS = 3;
// Motor stopped, sensors still updating, before the end reading is taken.
static constexpr uint32_t CAL_SETTLE_MS = 300;
// Hard per-move limit from move start: the move fails instead of driving on.
// The longest move measured at PWM 80 took 36 s (FLHL curling back from about
// 1000 edges, 2026-11-23), so 120 s leaves margin at the lowest PWM; moves at
// the default 128 are faster. It also ends a joint that creeps on forever
// (an edge at least every CAL_END_STUCK_STILL_MS keeps a drive alive).
static constexpr uint32_t CAL_MOVE_TIMEOUT_MS = 120000UL;
// Pot counts: an uncurl move that changes the pot by less than this fails the
// pass, and it is the travel at which a pot is told from a hall-B line.
static constexpr int CAL_MIN_TRAVEL_COUNTS = 30;
// A window with no motion cannot tell "already at the curl end" from "stalled",
// so a curl move only counts as reaching its curl end with evidence: it moved
// the pot at least CAL_MIN_TRAVEL_COUNTS, or it ended within this many counts of
// a known curl end (where this joint's curl end read earlier, or a measured
// stored limit on the curl side). A SWEEP move back to curl must also leave the
// uncurl end by
// CAL_MIN_TRAVEL_COUNTS and end within this many counts of the curl end read
// before the uncurl. Otherwise the pass fails before the partner joint moves.
static constexpr int CAL_CURL_END_TOLERANCE_COUNTS = 30;
// Longest follower reason the leader echoes into "CAL ROBOT FAIL <role>: ...",
// so that line stays under ~100 characters ("CAL ROBOT FAIL RIGHT: " is 22)
// even when a follower line arrives corrupted. Every reason the firmware itself
// prints fits (LegCalibration::printFailureReason).
static constexpr uint8_t CAL_ROBOT_REASON_MAX_CHARS = 78;
// Leader watchdog: longest silence from a calibrating follower's CAL lines.
static constexpr uint32_t CAL_FOLLOWER_SILENCE_MS = CAL_MOVE_TIMEOUT_MS + 15000UL;

// Curl direction, confirmed on the robot and identical on all six legs:
// +PWM extends a rod and -PWM retracts it. A hip curls by retracting, a knee
// by extending.
static constexpr int CAL_HIP_CURL_PWM_SIGN = -1;
static constexpr int CAL_KNEE_CURL_PWM_SIGN = +1;

// Hall encoder joints (quadrature builds). A joint is a hall when it counts at
// least CAL_HALL_DETECT_EDGES edges in the driven direction; the same number
// against the drive fails the move. Edges are x2 counts (both edges of A).
// The largest count change seen without motion is 6 (boot artifact 1, relaxation
// 1-3, coast at PWM 80 of 0-2, measured 2026-11-23), so 12 is twice it. Rest noise: 0 in ~301k frames.
static constexpr int CAL_HALL_DETECT_EDGES = 12;
// A curl that drove this many edges and then stalled is accepted as reaching its
// curl end (an obstacle looks the same after a boot).
static constexpr int CAL_HALL_CURL_EVIDENCE_EDGES = 12;
// Least progress per window: PWM 80 mid-stroke is 42-65 edges/s, PWM 40-50
// about 20/s (2026-11-23).
static constexpr int CAL_HALL_STALL_MIN_EDGES = 3;
// A curl may end this many edges short of a curl end proven this boot (one-sided:
// deeper is accepted). The knee end is compliant: a PWM 200 curl parks a knee
// about 17 edges deeper than a PWM 80 curl reaches, and the PWM 80 end itself
// varies 3-24 edges with the approach (FLKL, 2026-11-23).
static constexpr int CAL_HALL_CURL_END_TOLERANCE_EDGES = 40;
// Stroke bounds in edges: a knee's measured stroke must fall inside them, and the
// minimum is also the least a knee uncurl, or any SWEEP curl back, must travel.
// PROVISIONAL: one knee measured (FLKL, full stroke 776 edges with its hip
// curled, 2026-11-23): about 0.6 x and 2 x that stroke. Hall hips have no
// measured stroke: they sweep CAL_HIP_SWEEP_EDGES.
static constexpr int CAL_HALL_MIN_STROKE_EDGES = 450;
static constexpr int CAL_HALL_MAX_STROKE_EDGES = 1500;
// Fixed hall hip sweep: in SWEEP a hall hip uncurls from its proven curl end
// until it has made this many drive edges, then stops (it never runs to a
// stall), and its saved range is [0, CAL_HIP_SWEEP_EDGES], so T never drives it
// further. With its knee curled, a full hip uncurl presses the curled tibia
// against the bottom of the robot: FLHL stopped there at 941 edges (knee 24
// edges off its curl end) and 1003 edges (knee curled tight). PROVISIONAL:
// about 70% of that reach (the owner's "fixed 70% for now", 2026-11-23). Pot
// hips (builds without quadrature halls) still sweep to their end.
static constexpr int CAL_HIP_SWEEP_EDGES = 700;

// Verification nudge of a joint that shows no motion on its curl: out, then a
// full curl back. A hip nudge runs 24 edges or 2000 ms from nudge start (ramp
// included), whichever comes first. The knee nudge (CAL_OPT_KNEE_NUDGE, on by
// default) is bounded tighter. A pot nudge stops after CAL_POT_NUDGE_COUNTS.
// A knee with no hall feedback (hall A dead, cable off) is stopped only by the
// 1 s cap: a stored knee needs 0.36-0.77 s of drive to break free of its curl
// end (FLKL, 2026-11-23), so an earlier no-edge stop would also stop healthy
// knees, and a deadline at 0.9 s or later would save under 0.1 s of blind drive.
// At PWM 80 such a blind knee ended up to about 41 edges out. At the PWM 128
// nudge cap it is about twice that, measured 2026-11-24: FLKL driven out of its
// curl end at 128 had moved 72 edges 1.0 s after its drive started (with a
// 0.64 s ramp; the firmware ramps in 0.52 s), the FR knees ran about 120-130
// edges/s at 128, and a knee coasts 3-9 edges after its stop, so a blind knee
// that breaks free at once ends up to about 95 edges out (inside the 776-edge
// knee stroke) before it is curled back and fails NoFeedback. Its hip is not
// proven curled yet: the nudge is the one move allowed before that.
// A curled knee springs out 5-9 edges after its nudge stops: a 12-edge nudge
// ends about 18-21 edges out. The owner accepted that spring-out (2026-11-23).
// Those break-free times were measured at PWM 80 (2026-11-23); on 2026-11-24
// FRKL could not break free at 80 or 100, so the default is now 128 and a
// nudge below it may not break free in time: it then fails NoFeedback naming
// its PWM.
static constexpr int CAL_HIP_NUDGE_EDGES = 24;
static constexpr uint32_t CAL_HIP_NUDGE_MAX_MS = 2000;
static constexpr int CAL_KNEE_NUDGE_EDGES = 12;
static constexpr uint32_t CAL_KNEE_NUDGE_MAX_MS = 1000;
static constexpr int CAL_POT_NUDGE_COUNTS = 30;
// The knee nudge never ramps past this PWM, whatever the C pwm (less overshoot
// and coast past its 12 edges). Tied to the default: pinned at 80 it could not
// break FRKL free, nor at 100 (2026-11-24). The cost is the blind knee bound
// above (about 95 edges out at 128 against about 41 at 80); a lower cap (about
// 110) would shorten it, but a knee that cannot break free then fails
// NoFeedback. The owner's choice.
static constexpr int CAL_KNEE_NUDGE_MAX_PWM = CAL_DEFAULT_PWM;
// The curl back after a hall knee nudge is accepted when it stalls no more than
// CAL_HALL_CURL_END_TOLERANCE_EDGES short of where the knee rested before the
// nudge and moved at least this many edges toward curl: any motion beyond the
// 1-edge tolerance (CAL_POT_DETECT_MAX_EDGES), not CAL_HALL_DETECT_EDGES. A knee
// parked by a PWM 200 curl came back only 5 edges at PWM 80, stalling at the
// PWM 80 curl end; one parked at the PWM 80 end came back 14 (FLKL,
// 2026-11-23). A knee whose curl drive cannot move it at all is not accepted:
// it may be stuck mid-stroke (blocked toward curl), and accepting it would move
// its hip with the knee out. (A hall Return is a short drive of the end
// detector: unarmed, it ends Still, after at least as long as the nudge drove.)
// The cost, at PWM 80: a knee parked by a PWM 200 curl whose spring-out after
// the nudge is at the low end (5-6 edges) can come back 0-1 and fail "curled
// back only"; it is then at its calibration-speed curl end, so a second Curl
// only passes (at 128 that curl end is closer to the PWM 200 park). The hip
// return still needs CAL_HALL_DETECT_EDGES.
static constexpr int CAL_KNEE_RETURN_MIN_EDGES = 2;

// End detector (calibration_end_detector.h): hall joints on quadrature builds
// (every drive but the nudge out, while the joint is not proven Pot). Tuned on
// the 2026-11-24 front-leg traces (scripted jogs at PWM 80/120/150): a hip
// curl end and a knee uncurl end are the actuator's limit switch (the count
// stops dead and the current drops to the joint's switch-open level, 20-50%
// below running); the knee curl end is the leg's compliant stop (speed falls
// below 25% over 0.3-0.5 s while the current rises 30-80% at 120-150). Noise
// is +-3-5 counts frame to frame; running is only 2-8 counts above switch-open.
// Currents are tenths of an IS count (10 x avgIS), speeds edges per 200 ms.
static constexpr uint32_t CAL_END_SAMPLE_MS = 50;        // one sample: boxcar mean over it
static constexpr uint8_t CAL_END_RING = 24;              // samples kept (int16 edges, uint16 current)
static constexpr uint32_t CAL_END_LEARN_DELAY_MS = 200;  // after the ramp, before arming counts
static constexpr int CAL_END_RUN_MIN_EDGES = 4;          // per 200 ms (20 edges/s) is running
static constexpr uint8_t CAL_END_ARM_SAMPLES = 6;        // running samples in a row to arm (300 ms)
static constexpr int CAL_END_NORMAL_SPEED_PCT = 75;      // of vRef: the sample updates the references
static constexpr int CAL_END_IREF_STEP = 3;              // iRef moves at most this per sample
// Switch end (retract): the count still for 3 samples (150 ms); reference =
// median (and min) of the 20 samples (1 s) before the last forward edge, frozen
// per stop. A fast arrival (speed at that edge >= 50% of vRef) ends when the
// highest of the last 3 currents is <= 88% of it (hips drop 20-40%); a slow one
// (a joint crawling into its switch with its current raised: knee uncurls, and
// hips curling back from deep, at 0.50-0.62 of the reference) at <= 70%, below
// the reference minimum and <= 90% of iRef. A switch opens below the running
// level; a motor stalled mid-stroke draws more: FRHL curling at 128 from the
// body stalled 878 edges short of its switch (2026-11-24) with its current
// wobbling 17-31 counts over a running level of 16, and three samples at 17
// were 68% of the reference and below its minimum, but above 90% of iRef.
static constexpr uint8_t CAL_END_SWITCH_STILL_SAMPLES = 3;
static constexpr uint8_t CAL_END_SWITCH_REF_SAMPLES = 20;
static constexpr int CAL_END_ARRIVE_FAST_PCT = 50;
static constexpr int CAL_END_SWITCH_FAST_PCT = 88;
static constexpr int CAL_END_SWITCH_SLOW_PCT = 70;
static constexpr int CAL_END_SWITCH_SLOW_IREF_PCT = 90;
// Mechanical stop (extend): 3 samples in a row with the speed <= 25% of vRef
// and the median of the last 4 currents >= 125% of iRef.
static constexpr int CAL_END_STOP_SPEED_PCT = 25;
static constexpr int CAL_END_STOP_CURRENT_PCT = 125;
static constexpr uint8_t CAL_END_STOP_HOLD_SAMPLES = 3;
// Stuck mid-stroke (armed, no signature): fewer than CAL_HALL_STALL_MIN_EDGES
// edges over CAL_STALL_WINDOW_MS and no forward edge for this long (a knee
// crawling into its switch at 120 made an edge at least every ~0.3 s).
static constexpr uint32_t CAL_END_STUCK_STILL_MS = 400;
// Short curl drive: a curl that starts 50-130 edges from its end arms in the
// deceleration into it, so iRef and vRef are learned at the end and no later
// sample is normal running: its end never shows a signature against them. Such
// a stall ends Still (the plain stall) when the curl had fewer than this many
// normal-speed samples since arming (every 2026-11-24 curl end at PWM 120-150,
// replayed from each later start down to 14 edges, had at most 6; the FRKL
// curl at PWM 80 that ended stuck had 27, the slowly stalling FRHL above 7) ...
static constexpr uint8_t CAL_END_SHORT_NORMAL_SAMPLES = 12;
// ... and its last forward edge came fewer than this many samples after arming
// (at most 33 on those replays; the FRHL stall above ran 71). A retract whose
// median current is above iRef (a motor stalled mid-stroke) stays Stuck.
static constexpr uint8_t CAL_END_SHORT_RUN_SAMPLES = 40;
// Unarmed curl (a press at its end, a short return, a joint slower than
// CAL_END_RUN_MIN_EDGES per 200 ms): Still after fewer than
// CAL_HALL_STALL_MIN_EDGES edges in this window, not before the minimum drive
// time (a Return drives at least as long as its nudge did, and a Return whose
// nudge proved no motion, a blind one, CAL_STALL_WINDOW_MS longer). Pressing
// at an end thus takes about 1 s instead of about 3 s. Once the drive has made
// CAL_HALL_CURL_EVIDENCE_EDGES edges the window is CAL_STALL_WINDOW_MS: a slow
// joint (PWM 40-50 hips run 10-20 edges/s) pausing 0.5 s mid-stroke is not at
// its curl end, as it was not before the detector.
static constexpr uint32_t CAL_END_STILL_WINDOW_MS = 500;
static constexpr uint32_t CAL_END_STILL_MIN_DRIVE_MS = 1000;
// Switch-open signature (V1/R1, 2026-11-24 fix round). Once a limit switch
// opens the motor stops conducting and the current sits flat at the IS
// channel's offset: FL hip and knee 7-8 counts, FRHL 11, every switch end of
// the FL end test (clearance-FL-173941) and of fr-recover flat to within one
// count. The same channel reads that offset at the start of each drive, with
// EN high and the PWM still low (a running motor adds 2-9 counts, a stalled
// one more: FL at PWM 80 sat at 9-14 counts 229 edges short of its curl
// switch, FRHL at 128 at 17-31, 878 short). The floor is the lowest current of
// samples CAL_END_FLOOR_FIRST_SAMPLE (the IS has settled 150-200 ms after EN
// went high) through the ramp, at least through CAL_END_FLOOR_LAST_MIN_SAMPLE.
// switchOpen: the last CAL_END_OPEN_SAMPLES currents at most
// CAL_END_OPEN_MARGIN_TENTHS above the floor (FRHL's IS reaches its 11 only
// about 150 ms after EN goes high, so its floor reads up to 1 count low) and
// within CAL_END_OPEN_FLAT_TENTHS of each other.
static constexpr uint8_t CAL_END_FLOOR_FIRST_SAMPLE = 3;
static constexpr uint8_t CAL_END_FLOOR_LAST_MIN_SAMPLE = 4;
static constexpr uint8_t CAL_END_OPEN_SAMPLES = 6;
static constexpr uint16_t CAL_END_OPEN_MARGIN_TENTHS = 15;
static constexpr uint16_t CAL_END_OPEN_FLAT_TENTHS = 10;
// An unarmed stall of a switch-end drive is judged once the count has been
// still this many samples: the current takes about 3 samples to fall to the
// switch-open level after the count stops, and switchOpen then looks at the
// CAL_END_OPEN_SAMPLES after it.
static constexpr uint8_t CAL_END_OPEN_SETTLE_SAMPLES = 9;
// V3: on a switch-end drive Stuck needs CAL_END_SWITCH_STUCK_STILL_MS without
// a forward edge (CAL_END_STUCK_STILL_MS elsewhere), so the switch's drop and
// its switchOpen samples come first, and it waits on while the newest current
// is more than CAL_END_FALL_TENTHS below the higher of the two before it (a
// switch drop in progress: FLKL crawling into its switch at PWM 120 dropped
// 10, 8, 7 counts starting 0.41 s after its last edge, endcur-111808#10) or
// already sits within CAL_END_OPEN_MARGIN_TENTHS of the floor (the motor is
// not pressing: switchOpen is due), but no longer than
// CAL_END_STUCK_STILL_MAX_MS without a forward edge.
static constexpr uint32_t CAL_END_SWITCH_STUCK_STILL_MS = 600;
static constexpr uint16_t CAL_END_FALL_TENTHS = 5;
static constexpr uint32_t CAL_END_STUCK_STILL_MAX_MS = 1000;
// V3: the arrival speed of a switch stop is the fastest of the speeds at the
// last forward edge and this many samples before it (FLHL curling into its
// switch at 120 turned slow when its last edge came one sample late,
// endcur-111808#17).
static constexpr uint8_t CAL_END_ARRIVE_LOOKBACK_SAMPLES = 2;
// R2: a hip's extended end is a limit switch too (FLHL 3 of 3 on 2026-11-24:
// 11-12 counts falling to 7-8 at 1070, after a 25-50% rise over its last
// edges). On a hip extend the stop rule (then an obstacle) also needs the
// count still CAL_END_SWITCH_STOP_STILL_SAMPLES samples, so the switch can
// show its drop first, unless its streak reaches
// CAL_END_SWITCH_STOP_LONG_SAMPLES (an obstacle the hip still creeps into).
static constexpr uint8_t CAL_END_SWITCH_STOP_STILL_SAMPLES = 5;
static constexpr uint8_t CAL_END_SWITCH_STOP_LONG_SAMPLES = 8;

// Pot evidence on quadrature builds, read at rest (mean of direct reads). The
// analog pins also carry hall B, which rests at 0-19 or 1013-1020, so a pot has
// to read inside this band at rest (both snapshots to classify, one once proven).
static constexpr int CAL_POT_DETECT_MIN_RAW = 32;
static constexpr int CAL_POT_DETECT_MAX_RAW = 990;
// A pot never counts hall edges. The +-1 boot artifact is gone; 1 is tolerance.
static constexpr int CAL_POT_DETECT_MAX_EDGES = 1;
// Observed hall-B rest levels; the band above keeps a margin of 10 to them.
static constexpr int CAL_HALLB_REST_LOW_MAX = 19;
static constexpr int CAL_HALLB_REST_HIGH_MIN = 1013;

// Calibration option bits (C options, LegCalibration::start).
// Knee verification nudge before its hip is proven curled (12 edges, 1 s, then
// straight back): the one approved exception to "a knee moves only with its hip
// curled".
static constexpr uint8_t CAL_OPT_KNEE_NUDGE = 0x01;
static constexpr uint8_t CAL_OPT_ALL = CAL_OPT_KNEE_NUDGE;
// Shipped default, used by every C command run (the command cannot change it):
// the knee nudge, approved by the robot's owner on 2026-11-23. The strict path
// (no knee nudge) is options 0, passed explicitly to startCalibration.
static constexpr uint8_t CAL_DEFAULT_OPTIONS = CAL_OPT_KNEE_NUDGE;

// Pass selection bits: SEEK curls every leg, SWEEP measures each hip and knee.
static constexpr uint8_t CAL_PASS_SEEK = 0x01;
static constexpr uint8_t CAL_PASS_SWEEP = 0x02;
static constexpr uint8_t CAL_PASS_ALL = CAL_PASS_SEEK | CAL_PASS_SWEEP;

// True for a pass set and PWM that the calibration sequencers accept.
inline bool isValidCalibrationRequest(uint8_t passes, int pwm)
{
    return (passes & CAL_PASS_ALL) != 0 &&
           (passes & ~CAL_PASS_ALL) == 0 &&
           pwm >= CAL_PWM_MIN &&
           pwm <= CAL_PWM_MAX;
}

static_assert(CAL_PWM_MIN > 0 && CAL_PWM_MIN <= CAL_PWM_MAX,
              "calibration PWM range is empty");
static_assert(CAL_PWM_MAX <= ACTUATOR_PWM_MAXIMUM_MAGNITUDE,
              "calibration PWM exceeds the actuator PWM range");
static_assert(CAL_DEFAULT_PWM >= CAL_PWM_MIN && CAL_DEFAULT_PWM <= CAL_PWM_MAX,
              "default calibration PWM is outside the accepted range");
static_assert(CAL_PWM_RAMP_STEP > 0, "calibration ramp must climb");
static_assert(CAL_HIP_CURL_PWM_SIGN * CAL_HIP_CURL_PWM_SIGN == 1 &&
                  CAL_KNEE_CURL_PWM_SIGN * CAL_KNEE_CURL_PWM_SIGN == 1,
              "curl signs must be +1 or -1");
static_assert(CAL_CURL_END_TOLERANCE_COUNTS > 0 &&
                  CAL_CURL_END_TOLERANCE_COUNTS <= CAL_MIN_TRAVEL_COUNTS,
              "a curl end match must be tighter than the travel a sweep requires");
static_assert(CAL_FOLLOWER_SILENCE_MS > CAL_MOVE_TIMEOUT_MS + CAL_SETTLE_MS,
              "follower watchdog must outlast one calibration move");
static_assert(CAL_HALL_DETECT_EDGES >= 2 * (1 + 3 + 2),
              "hall detection must be twice the worst edges seen without motion");
static_assert(CAL_HALL_CURL_EVIDENCE_EDGES >= CAL_HALL_DETECT_EDGES,
              "curl evidence must be at least the detection threshold");
static_assert(CAL_HIP_NUDGE_EDGES >= 2 * CAL_HALL_DETECT_EDGES,
              "a hip nudge must leave room for a return of the detection threshold");
static_assert(CAL_KNEE_NUDGE_EDGES >= CAL_HALL_DETECT_EDGES,
              "a knee nudge must be able to reach the detection threshold");
static_assert(CAL_START_GRACE_MS + CAL_STALL_WINDOW_MS >= CAL_HIP_NUDGE_MAX_MS,
              "a windowed (pot) return curl must drive at least as long as the nudge");
static_assert(CAL_HIP_NUDGE_MAX_MS + 1000UL < 0xFFFFUL &&
                  CAL_END_STILL_MIN_DRIVE_MS >= CAL_KNEE_NUDGE_MAX_MS,
              "a nudge's drive time must fit uint16 and bound the unarmed return");
static_assert(CAL_HIP_NUDGE_MAX_MS >= CAL_KNEE_NUDGE_MAX_MS,
              "the knee nudge must not outlast the hip nudge");
static_assert(CAL_KNEE_NUDGE_MAX_PWM >= CAL_PWM_MIN && CAL_KNEE_NUDGE_MAX_PWM <= CAL_PWM_MAX,
              "the knee nudge PWM cap must be a valid calibration PWM");
static_assert(CAL_KNEE_RETURN_MIN_EDGES > CAL_POT_DETECT_MAX_EDGES &&
                  CAL_KNEE_RETURN_MIN_EDGES < CAL_HALL_DETECT_EDGES,
              "the knee return must show motion beyond the 1-edge tolerance, less than evidence");
static_assert(CAL_POT_DETECT_MIN_RAW >= CAL_HALLB_REST_LOW_MAX + 10 &&
                  CAL_POT_DETECT_MAX_RAW + 10 <= CAL_HALLB_REST_HIGH_MIN,
              "the pot band must keep a margin of 10 to the hall-B rest levels");
static_assert(CAL_HALL_MIN_STROKE_EDGES > CAL_HIP_NUDGE_EDGES + CAL_HALL_CURL_END_TOLERANCE_EDGES,
              "the minimum stroke must exceed a nudge plus the curl-end tolerance");
static_assert(CAL_HALL_MIN_STROKE_EDGES > ACTUATOR_ENC_SWITCH_INSET_EDGES + ACTUATOR_ENC_STOP_INSET_EDGES,
              "the stroke must be wider than the T insets");
static_assert(CAL_END_SAMPLE_MS > 0 && CAL_STALL_WINDOW_MS % CAL_END_SAMPLE_MS == 0 &&
                  CAL_END_STILL_WINDOW_MS % CAL_END_SAMPLE_MS == 0 &&
                  CAL_END_LEARN_DELAY_MS % CAL_END_SAMPLE_MS == 0 &&
                  (CAL_START_GRACE_MS + CAL_STALL_WINDOW_MS) % CAL_END_SAMPLE_MS == 0,
              "end detector windows must be whole samples");
static_assert(CAL_END_RING >= CAL_END_SWITCH_REF_SAMPLES + CAL_END_SWITCH_STILL_SAMPLES + 1 &&
                  CAL_END_RING > CAL_STALL_WINDOW_MS / CAL_END_SAMPLE_MS &&
                  CAL_END_RING > CAL_END_STILL_WINDOW_MS / CAL_END_SAMPLE_MS &&
                  CAL_END_RING >= CAL_END_ARM_SAMPLES + 4 + 1 &&
                  CAL_END_RING > CAL_END_SWITCH_STILL_SAMPLES + 4 &&
                  CAL_END_RING <= 255,
              "the end detector ring must hold every window it looks back over");
static_assert(CAL_END_SWITCH_STILL_SAMPLES >= 1 && CAL_END_STOP_HOLD_SAMPLES >= 1 &&
                  CAL_END_ARM_SAMPLES >= 1 && CAL_END_RUN_MIN_EDGES > 0 &&
                  CAL_END_SWITCH_REF_SAMPLES >= CAL_END_ARM_SAMPLES &&
                  CAL_END_SWITCH_REF_SAMPLES >= 4,
              "end detector counts must be positive and fit the median buffer");
static_assert(CAL_END_SWITCH_SLOW_PCT < CAL_END_SWITCH_FAST_PCT && CAL_END_SWITCH_FAST_PCT < 100 &&
                  CAL_END_STOP_CURRENT_PCT > 100 && CAL_END_STOP_SPEED_PCT < CAL_END_NORMAL_SPEED_PCT,
              "end signatures: a switch drops the current, a stop raises it and slows the joint");
static_assert(CAL_END_STUCK_STILL_MS < CAL_STALL_WINDOW_MS,
              "stuck needs less stillness than the stall window");
static_assert(CAL_END_SWITCH_SLOW_IREF_PCT < 100 && CAL_END_SHORT_NORMAL_SAMPLES > 0 &&
                  CAL_END_SHORT_RUN_SAMPLES > 0,
              "a slow switch opens below the running level; the short-curl bounds must be positive");
static_assert(CAL_HIP_NUDGE_MAX_MS + CAL_STALL_WINDOW_MS < 0xFFFFUL,
              "a blind Return's minimum drive time must fit uint16");
static_assert(CAL_END_OPEN_SAMPLES >= 2 && CAL_END_OPEN_SAMPLES < CAL_END_RING &&
                  CAL_END_OPEN_SETTLE_SAMPLES >= CAL_END_SWITCH_STILL_SAMPLES + CAL_END_OPEN_SAMPLES &&
                  CAL_END_FLOOR_FIRST_SAMPLE <= CAL_END_FLOOR_LAST_MIN_SAMPLE &&
                  CAL_END_FLOOR_LAST_MIN_SAMPLE < CAL_STALL_WINDOW_MS / CAL_END_SAMPLE_MS,
              "the switch-open test must fit the ring and follow the floor");
static_assert(CAL_END_SWITCH_STUCK_STILL_MS >= CAL_END_STUCK_STILL_MS &&
                  CAL_END_SWITCH_STUCK_STILL_MS / CAL_END_SAMPLE_MS > CAL_END_OPEN_SETTLE_SAMPLES &&
                  CAL_END_STUCK_STILL_MAX_MS > CAL_END_SWITCH_STUCK_STILL_MS &&
                  CAL_END_STUCK_STILL_MAX_MS <= CAL_STALL_WINDOW_MS,
              "the falling-current wait must be bounded inside the stall window");
static_assert(CAL_END_SWITCH_STILL_SAMPLES + CAL_END_ARRIVE_LOOKBACK_SAMPLES + 4 < CAL_END_RING,
              "the arrival lookback must fit the ring");
static_assert(CAL_END_SWITCH_STOP_STILL_SAMPLES > CAL_END_SWITCH_STILL_SAMPLES &&
                  CAL_END_SWITCH_STOP_LONG_SAMPLES > CAL_END_STOP_HOLD_SAMPLES,
              "a hip extend's stop must leave its switch time to show");
static_assert(CAL_HALL_MIN_STROKE_EDGES < CAL_HALL_MAX_STROKE_EDGES,
              "the stroke bounds are empty");
static_assert(CAL_HALL_MAX_STROKE_EDGES + ACTUATOR_ENC_RANGE_MARGIN_EDGES <=
                      ACTUATOR_HALL_POSITION_LIMIT &&
                  ACTUATOR_HALL_POSITION_LIMIT < 32767,
              "the largest stroke and range margin must fit the int16 hall frame");
static_assert(CAL_HIP_SWEEP_EDGES >= CAL_HALL_MIN_STROKE_EDGES &&
                  CAL_HIP_SWEEP_EDGES <= CAL_HALL_MAX_STROKE_EDGES,
              "the hip sweep must leave room for the curl back's minimum travel and fit the stroke bounds");
static_assert(CAL_HIP_SWEEP_EDGES > CAL_HIP_NUDGE_EDGES + CAL_HALL_CURL_END_TOLERANCE_EDGES,
              "the hip sweep must exceed a nudge plus the curl-end tolerance");
static_assert((CAL_DEFAULT_OPTIONS & ~CAL_OPT_ALL) == 0,
              "the default options contain an unknown bit");

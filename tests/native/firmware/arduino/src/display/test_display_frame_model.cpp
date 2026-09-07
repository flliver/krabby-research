#include <stdint.h>

#include "src/display/display_frame_model.h"
#include "unity.h"

namespace
{
const char *VALID_LEFT =
    "LEFT ; RLHY 0.1 0 0 1 1 0 30 0; RLHL nan 0 0 0 0 0 0 0;"
    " RLKL 0.2 0 0 1 1 40 0 0; MLHY 0.3 0 0 0 0 0 0 0;"
    " MLHL 0.4 0 0 0 0 0 0 0; MLKL 0.5 0 0 0 0 0 0 0";
const char *VALID_LEFT_CURRENT =
    "LEFT ; RLHY 0.1 0 0 1 1 0 30 0 1; RLHL nan 0 0 0 0 0 0 0 1;"
    " RLKL 0.2 0 0 1 1 40 0 0 1; MLHY 0.3 0 0 0 0 0 0 0 1;"
    " MLHL 0.4 0 0 0 0 0 0 0 1; MLKL 0.5 0 0 0 0 0 0 0 1";
}

void setUp() {}
void tearDown() {}

static void setControllerActuatorIds(
    ActuatorStatus (&actuators)[CONTROLLER_ACTUATOR_COUNT],
    ActuatorId firstActuator,
    ActuatorId secondActuator)
{
    for (uint8_t offset = 0; offset < 3; ++offset)
    {
        actuators[offset].actuatorId =
            static_cast<ActuatorId>(firstActuator + offset);
        actuators[offset + 3].actuatorId =
            static_cast<ActuatorId>(secondActuator + offset);
    }
}

static bool parseControllerTelemetry(
    ControllerFreshnessTracker &tracker,
    ActuatorStatus (&latest)[ActuatorId::ActuatorCount],
    const char *line,
    BoardRole expectedRole,
    uint32_t nowMilliseconds)
{
    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT];
    if (!parseActuatorStatus(line, expectedRole, actuators))
        return false;
    tracker = ControllerFreshnessTracker::seenAt(nowMilliseconds);
    for (const ActuatorStatus &actuator : actuators)
        latest[actuator.actuatorId] = actuator;
    return true;
}

static ActuatorGlyph updateAndGetActuatorGlyph(
    ActuatorConnection connectionState,
    int pwm,
    int moveThreshold)
{
    ActuatorStatus actuator;
    actuator.connectionState = connectionState;
    actuator.commandedPwm = static_cast<int16_t>(pwm);
    return selectActuatorGlyph(actuator, moveThreshold);
}

static void test_nine_field_segment_reads_as_unverified()
{
    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT];
    TEST_ASSERT_TRUE(parseActuatorStatus(
        VALID_LEFT, ROLE_LEFT, actuators));
    TEST_ASSERT_EQUAL_INT((int)ActuatorId::RLHY, (int)actuators[0].actuatorId);
    TEST_ASSERT_EQUAL_INT((int)ActuatorId::MLKL, (int)actuators[5].actuatorId);
    for (size_t i = 0; i < CONTROLLER_ACTUATOR_COUNT; ++i)
        TEST_ASSERT_EQUAL_INT(
            (int)(i == 1
                ? ActuatorConnection::Disconnected
                : ActuatorConnection::Unknown),
            (int)actuators[i].connectionState);
}

static void test_actuator_identity_makes_segment_order_irrelevant()
{
    const char *line =
        "LEFT ; MLKL 0 0 0 0 0 0 0 0 1; RLHY 0 0 0 0 0 0 20 0 1;"
        " MLHY 0 0 0 0 0 0 0 0 1; RLKL 0 0 0 0 0 0 0 0 1;"
        " MLHL 0 0 0 0 0 0 0 0 1; RLHL 0 0 0 0 0 0 0 0 1";
    ControllerFreshnessTracker tracker;
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    TEST_ASSERT_TRUE(parseControllerTelemetry(
        tracker, latest, line, ROLE_LEFT, 1000));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Extend,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLHY], 20));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Hold,
        (int)selectActuatorGlyph(
            latest[ActuatorId::MLKL], 20));
}

static void test_invalid_actuator_identity_rejects_the_whole_line()
{
    const char *unknownName =
        "LEFT ; XZQW 0 0 0 0 0 0 0 0; RLHL 0 0 0 0 0 0 0 0;"
        " RLKL 0 0 0 0 0 0 0 0; MLHY 0 0 0 0 0 0 0 0;"
        " MLHL 0 0 0 0 0 0 0 0; MLKL 0 0 0 0 0 0 0 0";
    const char *shortName =
        "LEFT ; RLH 0 0 0 0 0 0 0 0; RLHL 0 0 0 0 0 0 0 0;"
        " RLKL 0 0 0 0 0 0 0 0; MLHY 0 0 0 0 0 0 0 0;"
        " MLHL 0 0 0 0 0 0 0 0; MLKL 0 0 0 0 0 0 0 0";
    const char *duplicateAndMissing =
        "LEFT ; RLHY 0 0 0 0 0 0 0 0; RLHL 0 0 0 0 0 0 0 0;"
        " RLKL 0 0 0 0 0 0 0 0; MLHY 0 0 0 0 0 0 0 0;"
        " MLHL 0 0 0 0 0 0 0 0; RLHY 0 0 0 0 0 0 0 0";
    const char *wrongController =
        "LEFT ; FLHY 0 0 0 0 0 0 0 0; RLHL 0 0 0 0 0 0 0 0;"
        " RLKL 0 0 0 0 0 0 0 0; MLHY 0 0 0 0 0 0 0 0;"
        " MLHL 0 0 0 0 0 0 0 0; MLKL 0 0 0 0 0 0 0 0";

    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT];
    actuators[0].actuatorId = ActuatorId::FRKL;
    TEST_ASSERT_FALSE(parseActuatorStatus(
        unknownName, ROLE_LEFT, actuators));
    TEST_ASSERT_FALSE(parseActuatorStatus(
        shortName, ROLE_LEFT, actuators));
    TEST_ASSERT_FALSE(parseActuatorStatus(
        duplicateAndMissing, ROLE_LEFT, actuators));
    TEST_ASSERT_FALSE(parseActuatorStatus(
        wrongController, ROLE_LEFT, actuators));
    TEST_ASSERT_EQUAL_INT((int)ActuatorId::FRKL, (int)actuators[0].actuatorId);
}

static void test_tenth_field_carries_composed_connection_state()
{
    const char *line =
        "LEFT ; RLHY 0.1 0 0 1 1 0 30 0 0; RLHL 0.2 0 0 0 0 0 0 0 1;"
        " RLKL 0.2 0 0 1 1 40 0 0 2; MLHY 0.3 0 0 0 0 0 0 0 1;"
        " MLHL 0.4 0 0 0 0 0 0 0 0; MLKL 0.5 0 0 0 0 0 0 0 2";
    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT];
    TEST_ASSERT_TRUE(parseActuatorStatus(
        line, ROLE_LEFT, actuators));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorConnection::Unknown,
        (int)actuators[0].connectionState);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorConnection::Connected,
        (int)actuators[1].connectionState);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorConnection::Unknown,
        (int)actuators[4].connectionState);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorConnection::Disconnected,
        (int)actuators[5].connectionState);
}

static void test_multi_character_connection_state_field_is_rejected()
{
    const char *line =
        "LEFT ; RLHY 0.1 0 0 1 1 0 30 0 10; RLHL 0.2 0 0 0 0 0 0 0 1;"
        " RLKL 0.2 0 0 1 1 40 0 0 0; MLHY 0.3 0 0 0 0 0 0 0 1;"
        " MLHL 0.4 0 0 0 0 0 0 0 0; MLKL 0.5 0 0 0 0 0 0 0 1";
    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT];
    TEST_ASSERT_FALSE(parseActuatorStatus(
        line, ROLE_LEFT, actuators));
}

static void test_eleven_field_segment_is_rejected()
{
    const char *line =
        "LEFT ; RLHY 0.1 0 0 1 1 0 30 0 1 9; RLHL 0.2 0 0 0 0 0 0 0 1;"
        " RLKL 0.2 0 0 1 1 40 0 0 0; MLHY 0.3 0 0 0 0 0 0 0 1;"
        " MLHL 0.4 0 0 0 0 0 0 0 0; MLKL 0.5 0 0 0 0 0 0 0 1";
    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT];
    TEST_ASSERT_FALSE(parseActuatorStatus(
        line, ROLE_LEFT, actuators));
}

static void test_unverified_follower_channel_renders_unverified()
{
    const char *line =
        "LEFT ; RLHY 0.1 0 0 0 0 0 0 0 0; RLHL 0.2 0 0 0 0 0 0 0 1;"
        " RLKL 0.2 0 0 0 0 0 0 0 0; MLHY 0.3 0 0 0 0 0 0 0 1;"
        " MLHL 0.4 0 0 0 0 0 0 0 0; MLKL 0.5 0 0 0 0 0 0 0 1";
    ControllerFreshnessTracker tracker;
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    TEST_ASSERT_TRUE(parseControllerTelemetry(
        tracker, latest, line, ROLE_LEFT, 1000));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Unverified,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLHY], 20));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Hold,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLHL], 20));
}

static void test_glyph_boundaries_and_disconnection()
{
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Disconnected,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Disconnected, 255, 20));
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Hold,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Connected, 19, 20));
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Extend,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Connected, 20, 20));
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Retract,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Connected, -20, 20));
}

static void test_unknown_connection_is_distinct_from_hold()
{
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Unverified,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Unknown, 0, 20));
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Unverified,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Unknown, 19, 20));

    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Extend,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Unknown, 20, 20));
    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Retract,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Unknown, -20, 20));

    TEST_ASSERT_EQUAL_INT((int)ActuatorGlyph::Disconnected,
        (int)updateAndGetActuatorGlyph(
            ActuatorConnection::Disconnected, 0, 20));
}

static void test_controller_telemetry_updates_freshness_and_glyphs_transactionally()
{
    ControllerFreshnessTracker tracker;
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    TEST_ASSERT_TRUE(parseControllerTelemetry(
        tracker, latest, VALID_LEFT_CURRENT, ROLE_LEFT, 1000));
    TEST_ASSERT_TRUE(tracker.isFresh(
        1000, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Extend,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLHY], 20));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Disconnected,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLHL], 20));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Retract,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLKL], 20));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Hold,
        (int)selectActuatorGlyph(
            latest[ActuatorId::MLHY], 20));

    TEST_ASSERT_FALSE(parseControllerTelemetry(
        tracker, latest, "LEFT ; RLHY 0 0", ROLE_LEFT, 2000));
    TEST_ASSERT_TRUE(tracker.isFresh(
        1499, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
    TEST_ASSERT_FALSE(tracker.isFresh(
        1500, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Extend,
        (int)selectActuatorGlyph(
            latest[ActuatorId::RLHY], 20));
}

static void test_invalid_pwm_and_conflicting_directions_are_rejected()
{
    ControllerFreshnessTracker tracker;
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    const char *outOfRange =
        "LEFT ; RLHY 0 0 0 0 0 0 256 0; RLHL 0 0 0 0 0 0 0 0;"
        " RLKL 0 0 0 0 0 0 0 0; MLHY 0 0 0 0 0 0 0 0;"
        " MLHL 0 0 0 0 0 0 0 0; MLKL 0 0 0 0 0 0 0 0";
    const char *bothDirections =
        "LEFT ; RLHY 0 0 0 0 0 1 1 0; RLHL 0 0 0 0 0 0 0 0;"
        " RLKL 0 0 0 0 0 0 0 0; MLHY 0 0 0 0 0 0 0 0;"
        " MLHL 0 0 0 0 0 0 0 0; MLKL 0 0 0 0 0 0 0 0";
    TEST_ASSERT_FALSE(parseControllerTelemetry(
        tracker, latest, outOfRange, ROLE_LEFT, 1));
    TEST_ASSERT_FALSE(parseControllerTelemetry(
        tracker, latest, bothDirections, ROLE_LEFT, 1));
    TEST_ASSERT_FALSE(tracker.isFresh(
        1, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
}

static void test_freshness_boundary_and_rollover()
{
    ControllerFreshnessTracker tracker;
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    TEST_ASSERT_FALSE(tracker.isFresh(
        0, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
    TEST_ASSERT_TRUE(parseControllerTelemetry(
        tracker, latest, VALID_LEFT, ROLE_LEFT,
        UINT32_MAX - 100));
    TEST_ASSERT_TRUE(tracker.isFresh(
        398, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
    TEST_ASSERT_FALSE(tracker.isFresh(
        399, CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS));
}

static void test_controller_freshness_snapshot_disconnect_scan_and_frame_equality()
{
    ActuatorStatus actuators[CONTROLLER_ACTUATOR_COUNT] = {};
    setControllerActuatorIds(actuators, ActuatorId::RLHY, ActuatorId::MLHY);
    for (size_t actuator = 0; actuator < CONTROLLER_ACTUATOR_COUNT; ++actuator)
        actuators[actuator].connectionState = ActuatorConnection::Connected;
    actuators[1].commandedPwm = 20;
    actuators[2].commandedPwm = -20;
    actuators[3].connectionState = ActuatorConnection::Disconnected;
    actuators[5].commandedPwm = 20;
    ControllerFreshnessTracker trackers[BOARD_ROLE_COUNT];
    trackers[ROLE_LEFT] = ControllerFreshnessTracker::seenAt(0);
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    for (const ActuatorStatus &actuator : actuators)
        latest[actuator.actuatorId] = actuator;
    DisplayFrame frame = buildDisplayFrame(
        ROLE_UNKNOWN,
        trackers,
        latest,
        ImuMeasurement(), 0, 20);
    TEST_ASSERT_TRUE(frame.controllers[ROLE_LEFT]);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Hold,
        (int)frame.actuators[ActuatorId::RLHY]);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorGlyph::Disconnected,
        (int)frame.actuators[ActuatorId::MLHY]);
    TEST_ASSERT_TRUE(hasDisconnectedActuator(frame));

    DisplayFrame sameFrame = frame;
    TEST_ASSERT_TRUE(displayFramesEqual(frame, sameFrame));
    sameFrame.controllers[ROLE_RIGHT] = true;
    TEST_ASSERT_FALSE(displayFramesEqual(frame, sameFrame));
    sameFrame.controllers[ROLE_RIGHT] = false;
    frame.pitch = Degrees(1.0f);
    TEST_ASSERT_FALSE(displayFramesEqual(frame, sameFrame));

    DisplayFrame batteryChanged = sameFrame;
    batteryChanged.batteryLevel[1] = 0.5f;
    TEST_ASSERT_FALSE(displayFramesEqual(sameFrame, batteryChanged));
}

static ImuMeasurement accelerationSample(float ax, float ay, float az)
{
    ImuMeasurement measurement{true};
    measurement.acceleration[0] = MetersPerSecondSquared(ax);
    measurement.acceleration[1] = MetersPerSecondSquared(ay);
    measurement.acceleration[2] = MetersPerSecondSquared(az);
    return measurement;
}

static void setControllerGlyphs(
    ActuatorStatus (&actuators)[ActuatorId::ActuatorCount],
    BoardRole boardRole,
    ActuatorGlyph glyph)
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        if (getBoardRoleForActuator(actuatorId) == boardRole)
        {
            ActuatorStatus &actuator = actuators[actuatorId];
            actuator.actuatorId = actuatorId;
            actuator.connectionState = glyph == ActuatorGlyph::Disconnected
                ? ActuatorConnection::Disconnected
                : glyph == ActuatorGlyph::Unverified
                    ? ActuatorConnection::Unknown
                    : ActuatorConnection::Connected;
            actuator.commandedPwm = glyph == ActuatorGlyph::Extend
                ? 20
                : glyph == ActuatorGlyph::Retract ? -20 : 0;
        }
}

static void test_frame_rounds_tilt_to_whole_degrees()
{
    ControllerFreshnessTracker trackers[BOARD_ROLE_COUNT];
    ActuatorStatus latest[ActuatorId::ActuatorCount];

    // Presentation rounds the 5.71-degree IMU result.
    const DisplayFrame positive = buildDisplayFrame(
        ROLE_FRONT, trackers, latest,
        accelerationSample(0, 0.1f, 1.0f), 0, 20);
    const DisplayFrame negative = buildDisplayFrame(
        ROLE_FRONT, trackers, latest,
        accelerationSample(0, -0.1f, 1.0f), 0, 20);

    TEST_ASSERT_EQUAL_FLOAT(6.0f, positive.roll.value());
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, negative.roll.value());
}

static void test_every_role_has_a_label()
{
    TEST_ASSERT_EQUAL_STRING(
        "FRONT", boardRoleLabel(ROLE_FRONT));
    TEST_ASSERT_EQUAL_STRING(
        "LEFT", boardRoleLabel(ROLE_LEFT));
    TEST_ASSERT_EQUAL_STRING(
        "RIGHT", boardRoleLabel(ROLE_RIGHT));
    TEST_ASSERT_EQUAL_STRING(
        "UNKWN", boardRoleLabel(ROLE_UNKNOWN));
}

static void test_disconnect_check_ignores_stale_states()
{
    ControllerFreshnessTracker trackers[BOARD_ROLE_COUNT];
    trackers[ROLE_FRONT] = ControllerFreshnessTracker::seenAt(0);
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    setControllerGlyphs(
        latest, ROLE_FRONT, ActuatorGlyph::Hold);

    DisplayFrame frame = buildDisplayFrame(
        ROLE_FRONT, trackers, latest, ImuMeasurement(), 0, 20);
    TEST_ASSERT_FALSE(hasDisconnectedActuator(frame));

    latest[ActuatorId::FLHY].connectionState =
        ActuatorConnection::Disconnected;
    frame = buildDisplayFrame(
        ROLE_FRONT, trackers, latest, ImuMeasurement(), 0, 20);
    TEST_ASSERT_TRUE(hasDisconnectedActuator(frame));

    frame = buildDisplayFrame(
        ROLE_FRONT, trackers, latest, ImuMeasurement(),
        CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS, 20);
    TEST_ASSERT_FALSE(hasDisconnectedActuator(frame));
}

static void test_frame_groups_controllers_by_peer_freshness()
{
    const uint32_t now = 1000;
    ControllerFreshnessTracker trackers[BOARD_ROLE_COUNT];
    trackers[ROLE_FRONT] = ControllerFreshnessTracker::seenAt(now - 100);
    trackers[ROLE_LEFT] = ControllerFreshnessTracker::seenAt(now - 100);
    trackers[ROLE_RIGHT] = ControllerFreshnessTracker::seenAt(now - 900);
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    setControllerGlyphs(latest, ROLE_FRONT, ActuatorGlyph::Hold);
    setControllerGlyphs(latest, ROLE_LEFT, ActuatorGlyph::Extend);
    setControllerGlyphs(latest, ROLE_RIGHT, ActuatorGlyph::Retract);

    const DisplayFrame frame = buildDisplayFrame(
        ROLE_FRONT, trackers, latest,
        accelerationSample(0, 0, METERS_PER_SECOND_SQUARED_PER_G), now, 20);

    TEST_ASSERT_EQUAL_INT((int)ROLE_FRONT, (int)frame.role);
    TEST_ASSERT_TRUE(frame.controllers[ROLE_FRONT]);
    TEST_ASSERT_TRUE(frame.controllers[ROLE_LEFT]);
    TEST_ASSERT_FALSE(frame.controllers[ROLE_RIGHT]);

    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        const BoardRole boardRole = getBoardRoleForActuator(actuatorId);
        const ActuatorGlyph expected = boardRole == ROLE_FRONT
            ? ActuatorGlyph::Hold
            : boardRole == ROLE_LEFT
                ? ActuatorGlyph::Extend
                : ActuatorGlyph::Disconnected;
        TEST_ASSERT_EQUAL_INT(
            (int)expected,
            (int)frame.actuators[actuatorId]);
    }
}

static void test_frame_masks_front_actuators_when_front_is_missing()
{
    ControllerFreshnessTracker trackers[BOARD_ROLE_COUNT];
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    setControllerGlyphs(
        latest, ROLE_FRONT, ActuatorGlyph::Hold);

    const DisplayFrame frame = buildDisplayFrame(
        ROLE_FRONT, trackers, latest, ImuMeasurement(), 0, 20);

    TEST_ASSERT_FALSE(frame.controllers[ROLE_FRONT]);
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        if (getBoardRoleForActuator(actuatorId) != ROLE_FRONT)
            continue;
        TEST_ASSERT_EQUAL_INT(
            (int)ActuatorGlyph::Disconnected,
            (int)frame.actuators[actuatorId]);
    }
}

static void test_frame_keeps_default_tilt_when_the_sample_is_invalid()
{
    ControllerFreshnessTracker trackers[BOARD_ROLE_COUNT];
    for (BoardRole role : ALL_BOARD_ROLES)
        trackers[role] = ControllerFreshnessTracker::seenAt(0);
    ActuatorStatus latest[ActuatorId::ActuatorCount];
    for (BoardRole role : ALL_BOARD_ROLES)
        setControllerGlyphs(latest, role, ActuatorGlyph::Hold);

    ImuMeasurement failed =
        accelerationSample(0, METERS_PER_SECOND_SQUARED_PER_G, 0);
    failed.isValid = false;
    TEST_ASSERT_FALSE(failed.didSucceed());

    const DisplayFrame frame = buildDisplayFrame(
        ROLE_LEFT, trackers, latest, failed, 0, 20);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, frame.roll.value());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, frame.pitch.value());
}

static void test_battery_voltage_conversion_and_missing_readings()
{
    DisplayFrame frame;
    TEST_ASSERT_EQUAL_INT(BATTERY_DECIVOLTS_NO_SIGNAL, frame.batteryDecivolts[0]);
    TEST_ASSERT_EQUAL_INT(BATTERY_DECIVOLTS_NO_SIGNAL, frame.batteryDecivolts[1]);
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, frame.packVoltage.value());
    const Volts midpoint[2] = {Volts(12.7f), Volts(13.35f)};
    setBatteryVoltages(frame, midpoint);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.5f, frame.batteryLevel[0]);
    TEST_ASSERT_EQUAL_INT(127, frame.batteryDecivolts[0]);
    TEST_ASSERT_EQUAL_INT(134, frame.batteryDecivolts[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 26.05f, frame.packVoltage.value());

    const Volts clamped[2] = {Volts(10.0f), Volts(14.6f)};
    setBatteryVoltages(frame, clamped);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, frame.batteryLevel[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, frame.batteryLevel[1]);
    TEST_ASSERT_EQUAL_INT(100, frame.batteryDecivolts[0]);
    TEST_ASSERT_EQUAL_INT(146, frame.batteryDecivolts[1]);

    const float invalid[] = {-1.0f, NAN, INFINITY, 100.0f};
    for (float value : invalid)
        for (int battery = 0; battery < 2; ++battery)
        {
            Volts readings[2] = {Volts(13.3f), Volts(13.3f)};
            readings[battery] = Volts(value);
            setBatteryVoltages(frame, readings);
            TEST_ASSERT_EQUAL_FLOAT(0.0f, frame.batteryLevel[battery]);
            TEST_ASSERT_EQUAL_INT(BATTERY_DECIVOLTS_NO_SIGNAL, frame.batteryDecivolts[battery]);
            TEST_ASSERT_EQUAL_INT(133, frame.batteryDecivolts[1 - battery]);
            TEST_ASSERT_EQUAL_FLOAT(-1.0f, frame.packVoltage.value());
        }
    DisplayFrame same = frame;
    TEST_ASSERT_TRUE(displayFramesEqual(frame, same));
    ++same.batteryDecivolts[0];
    TEST_ASSERT_FALSE(displayFramesEqual(frame, same));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_battery_voltage_conversion_and_missing_readings);
    RUN_TEST(test_nine_field_segment_reads_as_unverified);
    RUN_TEST(test_actuator_identity_makes_segment_order_irrelevant);
    RUN_TEST(test_invalid_actuator_identity_rejects_the_whole_line);
    RUN_TEST(test_tenth_field_carries_composed_connection_state);
    RUN_TEST(test_multi_character_connection_state_field_is_rejected);
    RUN_TEST(test_eleven_field_segment_is_rejected);
    RUN_TEST(test_unverified_follower_channel_renders_unverified);
    RUN_TEST(test_glyph_boundaries_and_disconnection);
    RUN_TEST(test_unknown_connection_is_distinct_from_hold);
    RUN_TEST(test_controller_telemetry_updates_freshness_and_glyphs_transactionally);
    RUN_TEST(test_invalid_pwm_and_conflicting_directions_are_rejected);
    RUN_TEST(test_freshness_boundary_and_rollover);
    RUN_TEST(test_controller_freshness_snapshot_disconnect_scan_and_frame_equality);
    RUN_TEST(test_frame_rounds_tilt_to_whole_degrees);
    RUN_TEST(test_every_role_has_a_label);
    RUN_TEST(test_disconnect_check_ignores_stale_states);
    RUN_TEST(test_frame_groups_controllers_by_peer_freshness);
    RUN_TEST(test_frame_masks_front_actuators_when_front_is_missing);
    RUN_TEST(test_frame_keeps_default_tilt_when_the_sample_is_invalid);
    return UNITY_END();
}

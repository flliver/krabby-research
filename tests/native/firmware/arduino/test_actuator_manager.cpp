#include "unity.h"
#include "actuator_manager.h"
#include <algorithm>
#include <string>
#include <vector>

namespace fakeArduino
{
uint32_t now;
int modes[32], digital[32], pwm[32], analog[32];
std::deque<int> readings[32];
std::vector<Event> events;
}
Print Serial;
FakeEEPROM EEPROM;
// Hall stub: signed counts per slot and the build's quadrature flag.
int32_t g_fakeHallCount[6];
bool g_fakeQuadrature = false;
int32_t hallHwGetEdgeCount(uint8_t slot) { return g_fakeHallCount[slot]; }
bool hallHwHasQuadrature() { return g_fakeQuadrature; }

using namespace fakeArduino;

void setUp()
{
    now = 0;
    std::fill_n(modes, 32, INPUT);
    std::fill_n(digital, 32, LOW);
    std::fill_n(pwm, 32, 0);
    std::fill_n(analog, 32, 500);
    for (auto &queue : readings) queue.clear();
    events.clear();
    Serial.output.clear();
    EEPROM = FakeEEPROM();
    std::fill_n(g_fakeHallCount, 6, 0);
    g_fakeQuadrature = false;
}
void tearDown() {}

static LinearActuator makeActuator(int8_t slot = -1)
{
    return LinearActuator("FLHY", 0, 1, 2, 3, 4, slot);
}

static bool contains(const std::string &text, const char *needle)
{
    return text.find(needle) != std::string::npos;
}

static size_t occurrences(const std::string &text, const char *needle)
{
    size_t found = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1))
        ++found;
    return found;
}

static void assertKind(JointSensor expected, const LinearActuator &actuator)
{
    TEST_ASSERT_EQUAL_INT(static_cast<int>(expected), static_cast<int>(actuator.sensor));
}

// A Hall joint homed at its current count with a known stroke [lo, hi].
static void makeReadyHall(LinearActuator &actuator, int lo, int hi)
{
    actuator.setSensor(JointSensor::Hall);
    actuator.setHome();
    actuator.minStop = lo;
    actuator.maxStop = hi;
    actuator.encFlags |= ACTUATOR_FLAG_LIMITS_VALID;
}

static void test_slot_config_ctor_sets_pins_name_and_control_config()
{
    const LinearActuator::SlotConfig slot{10, 11, 12, 13, 14, 3};
    // Wide position-error deadband: mid-travel target must not command PWM if cfg applied.
    const LinearActuator::ControlConfig cfg(5, 10, 20, 200, 2.0f);
    LinearActuator actuator("RLHY", slot, cfg);

    TEST_ASSERT_EQUAL_STRING("RLHY", actuator.getName());
    TEST_ASSERT_EQUAL_INT(10, actuator.pinPwmR);
    TEST_ASSERT_EQUAL_INT(11, actuator.pinPwmL);
    TEST_ASSERT_EQUAL_INT(12, actuator.pinEn);
    TEST_ASSERT_EQUAL_INT(13, actuator.pinIS);
    TEST_ASSERT_EQUAL_INT(14, actuator.pinPot);
    TEST_ASSERT_EQUAL_INT(3, actuator.hallSlot);

    analog[14] = 500;
    actuator.init();
    TEST_ASSERT_TRUE(actuator.setTarget(0.6f)); // raw target ~614; error ~114 < 200 deadband
    now = 100;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(0, actuator.currentPwm);
    TEST_ASSERT_EQUAL_INT(0, pwm[10]);
    TEST_ASSERT_EQUAL_INT(0, pwm[11]);
    TEST_ASSERT_EQUAL_INT(LOW, digital[12]);
}

static void assertDrive(int extend, int retract, int enabled)
{
    TEST_ASSERT_EQUAL_INT(extend, pwm[0]);
    TEST_ASSERT_EQUAL_INT(retract, pwm[1]);
    TEST_ASSERT_EQUAL_INT(enabled, digital[2]);
}

static void test_init_resets_hardware_filters_and_evidence()
{
    auto actuator = makeActuator();
    actuator.manualDrive(100);
    actuator.setTarget(1);
    analog[3] = 12;
    analog[4] = 600;
    actuator.init();
    assertDrive(0, 0, LOW);
    TEST_ASSERT_EQUAL_INT(OUTPUT, modes[0]);
    TEST_ASSERT_EQUAL_INT(OUTPUT, modes[1]);
    TEST_ASSERT_EQUAL_INT(OUTPUT, modes[2]);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[3]);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[4]);
    TEST_ASSERT_FALSE(actuator.hasTarget);
    TEST_ASSERT_EQUAL_FLOAT(600, actuator.avgPot);
    TEST_ASSERT_EQUAL_FLOAT(12, actuator.avgIS);
    TEST_ASSERT_TRUE(actuator.potTracker.isValid());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorConnection::Unknown),
        static_cast<int>(actuator.getConnectionState()));
    TEST_ASSERT_EQUAL_STRING("FLHY", actuator.getName());
    assertKind(JointSensor::Pot, actuator);
    TEST_ASSERT_EQUAL_INT(1023, actuator.maxStop);
}

// On a quadrature build every joint starts Unknown: no limits, hall B pull-up kept.
static void test_init_on_quadrature_build_starts_unknown_with_pullup()
{
    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    actuator.encFlags = ACTUATOR_FLAG_HOMED;
    actuator.hallHome = 9;
    actuator.init();
    assertKind(JointSensor::Unknown, actuator);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
    TEST_ASSERT_EQUAL_INT(0, actuator.minStop);
    TEST_ASSERT_EQUAL_INT(0, actuator.maxStop);
    TEST_ASSERT_EQUAL_UINT8(0, actuator.encFlags);
    TEST_ASSERT_EQUAL_INT(0, actuator.hallHome);
}

static void test_probe_averages_in_order_and_restores_input()
{
    auto actuator = makeActuator();
    actuator.avgPot = 321;
    readings[4] = {498, 502, 499, 501, 508, 512, 509, 512};
    TEST_ASSERT_EQUAL_FLOAT(10.25f, actuator.readPotentiometerProbeRise());
    TEST_ASSERT_EQUAL_UINT(11, events.size());
    for (unsigned int i = 0; i < 4; ++i)
    {
        TEST_ASSERT_EQUAL_CHAR('r', events[i].operation);
        TEST_ASSERT_EQUAL_INT(4, events[i].pin);
        TEST_ASSERT_EQUAL_CHAR('r', events[i + 6].operation);
        TEST_ASSERT_EQUAL_INT(4, events[i + 6].pin);
    }
    TEST_ASSERT_EQUAL_CHAR('m', events[4].operation);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, events[4].value);
    TEST_ASSERT_EQUAL_CHAR('d', events[5].operation);
    TEST_ASSERT_EQUAL_INT(POT_PROBE_SETTLE_US, events[5].value);
    TEST_ASSERT_EQUAL_CHAR('m', events[10].operation);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[4]);
    TEST_ASSERT_EQUAL_FLOAT(321, actuator.avgPot);
    readings[4] = {510, 510, 510, 510, 500, 500, 500, 499};
    TEST_ASSERT_EQUAL_FLOAT(-10.25f, actuator.readPotentiometerProbeRise());
}

// Four direct reads, truncated mean; no pin mode change.
static void test_pot_snapshot_is_the_mean_of_four_direct_reads()
{
    auto actuator = makeActuator();
    readings[4] = {100, 101, 102, 104};
    TEST_ASSERT_EQUAL_INT(101, actuator.readPotSnapshot());
    TEST_ASSERT_EQUAL_UINT(4, events.size());
    for (const auto &event : events)
        TEST_ASSERT_EQUAL_CHAR('r', event.operation);
}

static void test_sensor_update_probe_timing_and_filter_isolation()
{
    auto actuator = makeActuator();
    actuator.init();
    events.clear();
    now = POT_PROBE_INTERVAL_MS - 1;
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(2, events.size());
    now++;
    readings[4] = {504, 498, 502, 498, 502, 520, 520, 520, 520};
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_FLOAT(500.6f, actuator.avgPot);
    TEST_ASSERT_TRUE(actuator.potTracker.isPositionOpen());
    TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));
    TEST_ASSERT_TRUE(readings[4].empty());
    events.clear();
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(2, events.size());
    actuator.manualDrive(100);
    now += POT_PROBE_INTERVAL_MS;
    events.clear();
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(2, events.size());
    actuator.stopMotor();
    actuator.updateSensors();
    TEST_ASSERT_FALSE(actuator.potTracker.isPositionOpen());
    TEST_ASSERT_FALSE(std::isnan(actuator.getPos()));
}

// The pot validity check and its pull-up probe run only for a proven pot.
static void test_probe_and_tracker_run_only_for_pot_joints()
{
    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    analog[4] = 1016;  // hall B resting high
    actuator.init();
    now = POT_PROBE_INTERVAL_MS * 3;
    events.clear();
    for (int i = 0; i < 5; ++i) actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(10, events.size());  // two reads per update, no probe
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
    TEST_ASSERT_TRUE(actuator.potTracker.isValid());  // not updated
    TEST_ASSERT_FALSE(actuator.isPositionReadable());

    actuator.setSensor(JointSensor::Pot);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[4]);
    events.clear();
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(13, events.size());  // probe ran
    TEST_ASSERT_EQUAL_INT(INPUT, modes[4]);
    for (int i = 0; i < 3; ++i) actuator.updateSensors();
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());  // 1016 is past the pot band
}

static void test_sensor_rail_jitter_debounce_and_recovery()
{
    auto actuator = makeActuator();
    actuator.init();
    for (int raw : {507, 500, 507, 500})
    {
        analog[4] = raw;
        actuator.updateSensors();
        TEST_ASSERT_TRUE(actuator.potTracker.isValid());
    }
    for (int raw : {600, 400, 600})
    {
        analog[4] = raw;
        actuator.updateSensors();
    }
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());
    actuator.manualDrive(100);
    analog[4] = 700;
    actuator.updateSensors();
    TEST_ASSERT_TRUE(actuator.potTracker.isValid());
    analog[4] = 0;
    for (int i = 0; i < 40; ++i) actuator.updateSensors();
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());
    analog[4] = 1023;
    for (int i = 0; i < 40; ++i) actuator.updateSensors();
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());
    analog[4] = 500;
    actuator.updateSensors();
    TEST_ASSERT_TRUE(actuator.potTracker.isValid());
}

static void test_probe_fractional_boundary_at_different_positions()
{
    for (int baseline : {100, 500, 990, 1013})
    {
        auto actuator = makeActuator();
        analog[4] = baseline;
        actuator.init();
        now = POT_PROBE_INTERVAL_MS;
        readings[4] = {baseline, baseline, baseline, baseline, baseline,
            baseline + 9, baseline + 10, baseline + 10, baseline + 10};
        actuator.updateSensors();
        TEST_ASSERT_FALSE(actuator.potTracker.isPositionOpen());
        now += POT_PROBE_INTERVAL_MS;
        readings[4] = {baseline, baseline, baseline, baseline, baseline,
            baseline + 10, baseline + 10, baseline + 10, baseline + 10};
        actuator.updateSensors();
        TEST_ASSERT_TRUE(actuator.potTracker.isPositionOpen());
        TEST_ASSERT_EQUAL_FLOAT(baseline, actuator.avgPot);
        TEST_ASSERT_EQUAL_INT(INPUT, modes[4]);
    }
}

static void test_delayed_probe_runs_once_and_rail_check_remains_independent()
{
    auto actuator = makeActuator();
    analog[4] = 1014;
    actuator.init();
    now = POT_PROBE_INTERVAL_MS * 10;
    events.clear();
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(13, events.size());
    TEST_ASSERT_FALSE(actuator.potTracker.isPositionOpen());
    events.clear();
    actuator.updateSensors();
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_UINT(4, events.size());
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());
    TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));
}

static void test_high_baselines_fail_rail_check_despite_saturated_probe()
{
    for (int baseline = 1014; baseline <= 1023; ++baseline)
    {
        auto actuator = makeActuator();
        analog[4] = baseline;
        actuator.init();
        now = POT_PROBE_INTERVAL_MS;
        readings[4] = {baseline, baseline, baseline, baseline, baseline,
            1023, 1023, 1023, 1023};
        actuator.updateSensors();
        TEST_ASSERT_FALSE(actuator.potTracker.isPositionOpen());
        TEST_ASSERT_TRUE(actuator.potTracker.isValid());
        actuator.updateSensors();
        TEST_ASSERT_TRUE(actuator.potTracker.isValid());
        actuator.updateSensors();
        TEST_ASSERT_FALSE(actuator.potTracker.isValid());
        TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));
    }
}

static void test_telemetry_keeps_valid_position_when_current_is_absent()
{
    auto actuator = makeActuator();
    analog[3] = 0;
    actuator.init();
    actuator.maxStop = 1000;
    actuator.manualDrive(100);
    for (int i = 0; i < 3; ++i) actuator.updateSensors();
    Print out;
    actuator.printTelemetry(out);
    TEST_ASSERT_EQUAL_STRING("FLHY 0.500 500 0 1 1 0 100 0 2", out.output.c_str());
    analog[4] = 0;
    for (int i = 0; i < 40; ++i) actuator.updateSensors();
    out.output.clear();
    actuator.printTelemetry(out);
    TEST_ASSERT_NOT_NULL(std::strstr(out.output.c_str(), "FLHY nan "));
}

static void test_filtered_current_qualifies_and_preserves_fraction()
{
    auto actuator = makeActuator();
    analog[3] = 0;
    actuator.init();
    actuator.manualDrive(255);
    analog[3] = 2;
    for (int i = 0; i < 3; ++i) actuator.updateSensors();
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.542f, actuator.avgIS);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorConnection::Disconnected),
        static_cast<int>(actuator.getConnectionState()));
    for (int i = 0; i < 6; ++i) actuator.updateSensors();
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorConnection::Connected),
        static_cast<int>(actuator.getConnectionState()));
    actuator.stopMotor();
    actuator.updateSensors();
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorConnection::Connected),
        static_cast<int>(actuator.getConnectionState()));
}

static void test_position_target_and_manual_control()
{
    auto actuator = makeActuator();
    actuator.minStop = 100;
    actuator.maxStop = 900;
    actuator.avgPot = 500.9f;
    TEST_ASSERT_EQUAL_INT(500, actuator.getRawPos());
    TEST_ASSERT_EQUAL_FLOAT(0.5f, actuator.getPos());
    TEST_ASSERT_TRUE(actuator.setTarget(-1));
    TEST_ASSERT_EQUAL_INT(100, actuator.currentTarget);
    TEST_ASSERT_TRUE(actuator.setTarget(2));
    TEST_ASSERT_EQUAL_INT(900, actuator.currentTarget);
    TEST_ASSERT_TRUE(actuator.setTarget(0.25f));
    TEST_ASSERT_EQUAL_INT(300, actuator.currentTarget);
    TEST_ASSERT_TRUE(actuator.hasTarget);
    actuator.manualDrive(999);
    TEST_ASSERT_FALSE(actuator.hasTarget);  // a jog is never overridden by a stale T
    assertDrive(255, 0, HIGH);
    actuator.manualDrive(-999);
    assertDrive(0, 255, HIGH);
    TEST_ASSERT_EQUAL_INT(-255, actuator.getStatus().commandedPwm);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorId::FLHY),
        static_cast<int>(actuator.getStatus().actuatorId));
    actuator.manualDrive(1);
    assertDrive(1, 0, HIGH);
    TEST_ASSERT_TRUE(actuator.setTarget(0.5f));
    actuator.manualDrive(0);
    assertDrive(0, 0, LOW);
    TEST_ASSERT_FALSE(actuator.hasTarget);
    TEST_ASSERT_TRUE(actuator.setTarget(0.5f));
    actuator.stopMotor();
    TEST_ASSERT_FALSE(actuator.hasTarget);
    actuator.maxStop = actuator.minStop;
    TEST_ASSERT_EQUAL_FLOAT(0.5f, actuator.getPos());
}

static void test_update_ramps_both_directions_and_respects_deadbands()
{
    auto actuator = makeActuator();
    actuator.init();
    actuator.setControlConfig(LinearActuator::ControlConfig(20, 10, 10, 5, 1));
    actuator.setTarget(1);
    now = 9;
    actuator.update();
    assertDrive(0, 0, LOW);
    now = 10;
    actuator.update();
    assertDrive(20, 0, HIGH);
    actuator.currentTarget = 525;
    now = 20;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(25, actuator.currentPwm);
    now = 30;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(25, actuator.currentPwm);
    actuator.currentTarget = 515;
    now = 40;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(15, actuator.currentPwm);
    actuator.currentTarget = 503;
    now = 50;
    actuator.update();
    assertDrive(0, 0, LOW);
    actuator.setTarget(0);
    now = 60;
    actuator.update();
    assertDrive(0, 20, HIGH);
    actuator.stopMotor();
    actuator.manualDrive(70);
    now = 70;
    actuator.update();
    assertDrive(70, 0, HIGH);
}

static void test_telemetry_fields_and_hall_bounds()
{
    for (int i = 0; i < 6; ++i) g_fakeHallCount[i] = 100 + i;
    for (int slot : {-1, 0, 5, 6})
    {
        auto actuator = makeActuator(slot);
        actuator.init();
        actuator.minStop = 0;
        actuator.maxStop = 1000;
        actuator.manualDrive(-23);
        Print out;
        actuator.printTelemetry(out);
        const std::string expected = "FLHY 0.500 500 500 1 1 23 0 " +
            std::to_string(slot >= 0 && slot < 6 ? 100 + slot : 0) + " 0";
        TEST_ASSERT_EQUAL_STRING(expected.c_str(), out.output.c_str());
        actuator.manualDrive(24);
        out.output.clear();
        actuator.printTelemetry(out);
        TEST_ASSERT_NOT_NULL(std::strstr(out.output.c_str(), " 1 1 0 24 "));
    }
}

// Unknown has no position even with a valid pot tracker; Hall needs a home from
// this boot and a stroke. The frame is x = -(count - home), clamped to +-30000.
static void test_position_per_sensor_kind_and_homing_state()
{
    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    actuator.init();
    g_fakeHallCount[0] = -250;
    TEST_ASSERT_TRUE(actuator.potTracker.isValid());
    TEST_ASSERT_FALSE(actuator.isPositionReadable());
    TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));
    TEST_ASSERT_EQUAL_INT(250, actuator.getRawPos());
    TEST_ASSERT_EQUAL_CHAR('?', actuator.sensorStateCode());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorConnection::Disconnected),
        static_cast<int>(actuator.getConnectionState()));
    Print out;
    actuator.printTelemetry(out);
    TEST_ASSERT_EQUAL_STRING("FLHY nan 500 500 0 0 0 0 -250 2", out.output.c_str());

    actuator.setSensor(JointSensor::Hall);
    TEST_ASSERT_EQUAL_CHAR('e', actuator.sensorStateCode());
    TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));
    actuator.setHome();
    TEST_ASSERT_EQUAL_INT(-250, actuator.hallHome);
    TEST_ASSERT_EQUAL_INT(0, actuator.getRawPos());
    TEST_ASSERT_EQUAL_CHAR('s', actuator.sensorStateCode());
    TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));  // no stroke yet

    actuator.maxStop = 1000;
    actuator.encFlags |= ACTUATOR_FLAG_LIMITS_VALID;
    TEST_ASSERT_EQUAL_CHAR('E', actuator.sensorStateCode());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, actuator.getPos());
    g_fakeHallCount[0] = -750;  // extended 500 edges
    TEST_ASSERT_EQUAL_FLOAT(0.5f, actuator.getPos());
    g_fakeHallCount[0] = -40000;
    TEST_ASSERT_EQUAL_INT(ACTUATOR_HALL_POSITION_LIMIT, actuator.hallPosition());
    g_fakeHallCount[0] = 40000;
    TEST_ASSERT_EQUAL_INT(-ACTUATOR_HALL_POSITION_LIMIT, actuator.hallPosition());

    actuator.clearHome();
    TEST_ASSERT_EQUAL_CHAR('x', actuator.sensorStateCode());
    TEST_ASSERT_EQUAL_UINT8(ACTUATOR_FLAG_HOME_LOST | ACTUATOR_FLAG_LIMITS_VALID, actuator.encFlags);
    actuator.clearHome();  // nothing more to lose
    TEST_ASSERT_EQUAL_UINT8(ACTUATOR_FLAG_HOME_LOST | ACTUATOR_FLAG_LIMITS_VALID, actuator.encFlags);
    TEST_ASSERT_TRUE(std::isnan(actuator.getPos()));
    actuator.setHome();
    TEST_ASSERT_EQUAL_UINT8(ACTUATOR_FLAG_HOMED | ACTUATOR_FLAG_LIMITS_VALID, actuator.encFlags);

    actuator.setSensor(JointSensor::Pot);
    TEST_ASSERT_EQUAL_CHAR('P', actuator.sensorStateCode());
    actuator.avgPot = 300.7f;
    TEST_ASSERT_EQUAL_INT(300, actuator.getRawPos());
    TEST_ASSERT_TRUE(actuator.isPositionReadable());
}

// setSensor: forced Pot without quadrature halls; pin mode on every call with
// them; a change resets the encoder state and limits and stops a target.
static void test_set_sensor_pin_modes_and_resets()
{
    auto plain = makeActuator(0);
    plain.init();
    events.clear();
    plain.minStop = 100;
    plain.setSensor(JointSensor::Hall);
    assertKind(JointSensor::Pot, plain);
    TEST_ASSERT_TRUE(events.empty());
    TEST_ASSERT_EQUAL_INT(100, plain.minStop);  // no change, no reset

    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    actuator.init();
    actuator.setSensor(JointSensor::Pot);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[4]);
    TEST_ASSERT_EQUAL_INT(0, actuator.minStop);
    TEST_ASSERT_EQUAL_INT(1023, actuator.maxStop);
    // Rail readings make the pot invalid; a later return to Pot reseeds the tracker.
    analog[4] = 0;
    for (int i = 0; i < 3; ++i) actuator.updateSensors();
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());
    actuator.setSensor(JointSensor::Hall);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
    TEST_ASSERT_EQUAL_INT(0, actuator.maxStop);
    actuator.setHome();
    events.clear();
    actuator.setSensor(JointSensor::Hall);  // same kind: pin mode only
    TEST_ASSERT_EQUAL_UINT(1, events.size());
    TEST_ASSERT_TRUE(actuator.encFlags & ACTUATOR_FLAG_HOMED);
    actuator.avgPot = 400;
    actuator.setSensor(JointSensor::Pot);
    TEST_ASSERT_TRUE(actuator.potTracker.isValid());
    TEST_ASSERT_EQUAL_UINT8(0, actuator.encFlags);
    TEST_ASSERT_EQUAL_INT(0, actuator.hallHome);

    TEST_ASSERT_TRUE(actuator.setTarget(1));
    now = 10;
    actuator.update();
    TEST_ASSERT_NOT_EQUAL(0, actuator.currentPwm);
    actuator.setSensor(JointSensor::Unknown);
    TEST_ASSERT_FALSE(actuator.hasTarget);
    TEST_ASSERT_EQUAL_INT(0, actuator.currentPwm);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
}

static void test_set_target_refuses_nan_and_unreadable_and_insets_encoder_targets()
{
    auto pot = makeActuator();
    pot.init();
    TEST_ASSERT_FALSE(pot.setTarget(NAN));
    TEST_ASSERT_FALSE(pot.hasTarget);
    TEST_ASSERT_TRUE(pot.setTarget(1));
    now = 10;
    pot.update();
    TEST_ASSERT_NOT_EQUAL(0, pot.currentPwm);
    TEST_ASSERT_FALSE(pot.setTarget(NAN));  // a refusal stops the active target
    TEST_ASSERT_FALSE(pot.hasTarget);
    TEST_ASSERT_EQUAL_INT(0, pot.currentPwm);

    g_fakeQuadrature = true;
    auto hip = makeActuator(0);
    hip.init();
    TEST_ASSERT_FALSE(hip.setTarget(0.5f));  // Unknown
    hip.setSensor(JointSensor::Hall);
    TEST_ASSERT_FALSE(hip.setTarget(0.5f));  // unhomed
    makeReadyHall(hip, 0, 1000);
    now = 1234;
    // Hall T stays 8 edges inside minStop (the retracted end: an actuator
    // limit switch) and, inside maxStop, 10 edges for a hip (its extend
    // switch) or 25 for a knee (a compliant leg stop).
    TEST_ASSERT_EQUAL_INT(8, ACTUATOR_ENC_SWITCH_INSET_EDGES);
    TEST_ASSERT_EQUAL_INT(25, ACTUATOR_ENC_STOP_INSET_EDGES);
    TEST_ASSERT_EQUAL_INT(10, ACTUATOR_ENC_HIP_EXTEND_INSET_EDGES);
    TEST_ASSERT_TRUE(hip.setTarget(0));
    TEST_ASSERT_EQUAL_INT(8, hip.currentTarget);
    TEST_ASSERT_EQUAL_UINT16(1234, hip.guardMs);
    TEST_ASSERT_TRUE(hip.setTarget(1));
    TEST_ASSERT_EQUAL_INT(1000 - 10, hip.currentTarget);
    TEST_ASSERT_TRUE(hip.setTarget(0.5f));
    TEST_ASSERT_EQUAL_INT(500, hip.currentTarget);
    // The guard window restarts only for a target that reverses the drive: one
    // re-sent in the same direction (a held key, a streaming host) keeps it.
    g_fakeHallCount[0] = -40;
    now = 2000;
    TEST_ASSERT_TRUE(hip.setTarget(0.5f));
    TEST_ASSERT_EQUAL_INT(0, hip.guardPos);
    TEST_ASSERT_EQUAL_UINT16(1234, hip.guardMs);
    TEST_ASSERT_TRUE(hip.setTarget(0));
    TEST_ASSERT_EQUAL_INT(40, hip.guardPos);
    TEST_ASSERT_EQUAL_UINT16(2000, hip.guardMs);
    hip.clearHome();
    TEST_ASSERT_FALSE(hip.setTarget(0.5f));
    TEST_ASSERT_FALSE(hip.hasTarget);

    auto knee = makeActuator(1);
    knee.init();
    makeReadyHall(knee, -1000, 0);
    TEST_ASSERT_TRUE(knee.setTarget(1));
    TEST_ASSERT_EQUAL_INT(-25, knee.currentTarget);  // the leg stop at the knee's curl end
    TEST_ASSERT_TRUE(knee.setTarget(0));
    TEST_ASSERT_EQUAL_INT(-1000 + 8, knee.currentTarget);  // the knee's uncurl limit switch
    TEST_ASSERT_TRUE(knee.setTarget(0.98f));
    TEST_ASSERT_EQUAL_INT(-25, knee.currentTarget);
    TEST_ASSERT_TRUE(knee.setTarget(0.5f));
    TEST_ASSERT_EQUAL_INT(-500, knee.currentTarget);

    // The saved hall hip range [0, CAL_HIP_STROKE_EDGES = 1069]: T covers
    // [8, 1059], the extend switch left as a hardware backstop. The knee of
    // the same stroke keeps its own bounds, [-S + 8, -25].
    TEST_ASSERT_EQUAL_INT(1069, CAL_HIP_STROKE_EDGES);
    auto strokeHip = makeActuator(0);
    strokeHip.init();
    makeReadyHall(strokeHip, 0, CAL_HIP_STROKE_EDGES);
    TEST_ASSERT_TRUE(strokeHip.isHallHipRange());
    TEST_ASSERT_TRUE(strokeHip.setTarget(0));
    TEST_ASSERT_EQUAL_INT(8, strokeHip.currentTarget);
    TEST_ASSERT_TRUE(strokeHip.setTarget(1));
    TEST_ASSERT_EQUAL_INT(1059, strokeHip.currentTarget);
    TEST_ASSERT_TRUE(strokeHip.setTarget(2.0f));  // beyond 1: still clamped
    TEST_ASSERT_EQUAL_INT(1059, strokeHip.currentTarget);
    TEST_ASSERT_TRUE(strokeHip.setTarget(-1.0f));
    TEST_ASSERT_EQUAL_INT(8, strokeHip.currentTarget);

    auto strokeKnee = makeActuator(1);
    strokeKnee.init();
    makeReadyHall(strokeKnee, -785, 0);
    TEST_ASSERT_FALSE(strokeKnee.isHallHipRange());
    TEST_ASSERT_TRUE(strokeKnee.setTarget(0));
    TEST_ASSERT_EQUAL_INT(-777, strokeKnee.currentTarget);
    TEST_ASSERT_TRUE(strokeKnee.setTarget(1));
    TEST_ASSERT_EQUAL_INT(-25, strokeKnee.currentTarget);

    // Pot targets have no inset.
    g_fakeQuadrature = false;
    auto potJoint = makeActuator(0);
    potJoint.init();
    TEST_ASSERT_TRUE(potJoint.setTarget(0));
    TEST_ASSERT_EQUAL_INT(potJoint.minStop, potJoint.currentTarget);
    TEST_ASSERT_TRUE(potJoint.setTarget(1));
    TEST_ASSERT_EQUAL_INT(potJoint.maxStop, potJoint.currentTarget);
}

// update() stops a target whose position becomes unreadable.
static void test_update_stops_target_when_position_becomes_unreadable()
{
    auto actuator = makeActuator();
    actuator.init();
    TEST_ASSERT_TRUE(actuator.setTarget(1));
    now = 10;
    actuator.update();
    TEST_ASSERT_NOT_EQUAL(0, actuator.currentPwm);
    analog[4] = 0;
    for (int i = 0; i < 3; ++i)
    {
        now += 10;
        actuator.update();
    }
    TEST_ASSERT_FALSE(actuator.potTracker.isValid());
    TEST_ASSERT_FALSE(actuator.hasTarget);
    assertDrive(0, 0, LOW);
}

// Encoder T: Kp 1 per edge, 12-edge deadband, PWM cap 150, same sign as a pot.
static void test_encoder_control_sign_deadband_and_cap()
{
    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    actuator.init();
    actuator.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    makeReadyHall(actuator, 0, 1000);
    TEST_ASSERT_TRUE(actuator.setTarget(0.5f));
    now = 10;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(ACTUATOR_ENC_MAX_PWM, actuator.currentPwm);
    assertDrive(ACTUATOR_ENC_MAX_PWM, 0, HIGH);
    g_fakeHallCount[0] = -470;  // x = 470: error 30
    now = 20;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(30, actuator.currentPwm);
    g_fakeHallCount[0] = -489;  // error 11: inside the deadband
    now = 30;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(0, actuator.currentPwm);
    g_fakeHallCount[0] = -900;  // past the target: retract
    now = 40;
    actuator.update();
    TEST_ASSERT_EQUAL_INT(-ACTUATOR_ENC_MAX_PWM, actuator.currentPwm);
    assertDrive(0, ACTUATOR_ENC_MAX_PWM, HIGH);
}

// Driven at PWM 60 or more without 3 edges or counts of progress for 1 s, a
// target is stopped and an ERR STALL line is pending; an encoder loses its home.
static void test_stall_guard_stops_hall_and_pot_targets()
{
    g_fakeQuadrature = true;
    auto hall = makeActuator(0);
    hall.init();
    hall.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    makeReadyHall(hall, 0, 1000);
    now = 65000;  // the window crosses the 16-bit wrap
    TEST_ASSERT_TRUE(hall.setTarget(0.5f));
    now += 10;
    hall.update();
    TEST_ASSERT_EQUAL_INT(ACTUATOR_ENC_MAX_PWM, hall.currentPwm);
    now += 900;
    g_fakeHallCount[0] = -3;  // progress: the window restarts
    hall.update();
    now += 999;
    hall.update();
    TEST_ASSERT_TRUE(hall.hasTarget);
    now += 1;
    hall.update();
    TEST_ASSERT_FALSE(hall.hasTarget);
    TEST_ASSERT_EQUAL_INT(0, hall.currentPwm);
    TEST_ASSERT_TRUE(hall.encFlags & ACTUATOR_FLAG_ERR_STALL);
    TEST_ASSERT_EQUAL_CHAR('x', hall.sensorStateCode());

    g_fakeQuadrature = false;
    auto pot = makeActuator();
    pot.init();
    pot.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    now = 0;
    TEST_ASSERT_TRUE(pot.setTarget(1));
    for (int i = 0; i < 100; ++i)
    {
        now += 10;
        pot.update();
    }
    TEST_ASSERT_FALSE(pot.hasTarget);
    TEST_ASSERT_TRUE(pot.encFlags & ACTUATOR_FLAG_ERR_STALL);
    TEST_ASSERT_EQUAL_CHAR('P', pot.sensorStateCode());

    // Below PWM 60 (above the drive deadband) a joint that does not move is
    // stopped after the 3 s window, quietly: no ERR line, an encoder keeps its home.
    auto gentle = makeActuator();
    gentle.init();
    gentle.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    TEST_ASSERT_TRUE(gentle.setTarget(0.5f));
    gentle.currentTarget = 525;  // error 25: PWM 50
    for (int i = 0; i < 299; ++i)
    {
        now += 10;
        gentle.update();
    }
    TEST_ASSERT_TRUE(gentle.hasTarget);
    TEST_ASSERT_EQUAL_INT(50, gentle.currentPwm);
    now += 10;
    gentle.update();
    TEST_ASSERT_FALSE(gentle.hasTarget);
    TEST_ASSERT_EQUAL_INT(0, gentle.currentPwm);
    TEST_ASSERT_EQUAL_UINT8(0, gentle.encFlags);

    g_fakeQuadrature = true;
    auto slow = makeActuator(0);
    slow.init();
    slow.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    makeReadyHall(slow, 0, 1000);
    g_fakeHallCount[0] = -460;  // x = 460
    TEST_ASSERT_TRUE(slow.setTarget(0.5f));  // error 40: PWM 40
    for (int i = 0; i < 300; ++i)
    {
        now += 10;
        slow.update();
    }
    TEST_ASSERT_FALSE(slow.hasTarget);
    TEST_ASSERT_EQUAL_UINT8(0, slow.encFlags & ACTUATOR_FLAG_ERR_STALL);
    TEST_ASSERT_EQUAL_CHAR('E', slow.sensorStateCode());

    // Within the drive deadband nothing is driven and nothing trips.
    g_fakeQuadrature = false;
    auto settled = makeActuator();
    settled.init();
    settled.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    TEST_ASSERT_TRUE(settled.setTarget(0.5f));
    settled.currentTarget = 520;  // error 20 at Kp 2: PWM 40; at Kp 0.5 below 20
    settled.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 0.5f));
    for (int i = 0; i < 400; ++i)
    {
        now += 10;
        settled.update();
    }
    TEST_ASSERT_TRUE(settled.hasTarget);
    TEST_ASSERT_EQUAL_INT(10, settled.currentPwm);
}

// A T re-sent while the joint is blocked (a held menu key, a streaming host)
// does not restart the stall window: the guard still trips 1 s after the drive
// stopped making progress.
static void test_stall_guard_is_not_reset_by_a_repeated_target()
{
    g_fakeQuadrature = true;
    auto hall = makeActuator(0);
    hall.init();
    hall.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    makeReadyHall(hall, 0, 1000);
    now = 100;
    TEST_ASSERT_TRUE(hall.setTarget(0.5f));
    uint32_t stoppedAt = 0;
    for (int i = 1; i <= 1000 && stoppedAt == 0; ++i)
    {
        now += 10;
        if (i % 30 == 0) hall.setTarget(0.5f);  // every 300 ms, count frozen
        hall.update();
        if (!hall.hasTarget) stoppedAt = now;
    }
    TEST_ASSERT_EQUAL_UINT32(1100, stoppedAt);
    TEST_ASSERT_TRUE(hall.encFlags & ACTUATOR_FLAG_ERR_STALL);
    TEST_ASSERT_EQUAL_CHAR('x', hall.sensorStateCode());
}

// A guard trip latches against a T re-sent in the same direction (a held menu
// key 0 re-sends every 0.3 s, a host may stream T): the blocked joint stays
// stopped, with one ERR line at most, until a T reverses, a jog or a stop (H).
static void test_stall_guard_trip_stays_stopped_while_the_same_target_is_resent()
{
    // A hall joint quiet-stopped below PWM 60 keeps its home, so a re-sent T is
    // accepted; it must not drive into the obstacle again.
    g_fakeQuadrature = true;
    auto slow = makeActuator(0);
    slow.init();
    slow.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    makeReadyHall(slow, 0, 1000);
    g_fakeHallCount[0] = -460;  // x = 460
    now = 0;
    TEST_ASSERT_TRUE(slow.setTarget(0.5f));  // error 40: PWM 40
    int i = 0;
    for (; i < 300 && slow.hasTarget; ++i)
    {
        now += 10;
        slow.update();
    }
    TEST_ASSERT_FALSE(slow.hasTarget);
    uint32_t drivenMs = 0;
    for (int tick = 1; tick <= 1000; ++tick)  // 10 s, T every 300 ms
    {
        now += 10;
        if (tick % 30 == 0) TEST_ASSERT_TRUE(slow.setTarget(0.5f));
        slow.update();
        if (slow.currentPwm != 0) drivenMs += 10;
    }
    TEST_ASSERT_EQUAL_UINT32(0, drivenMs);
    TEST_ASSERT_FALSE(slow.hasTarget);
    TEST_ASSERT_EQUAL_CHAR('E', slow.sensorStateCode());
    // A T the other way is obeyed, and releases the latch.
    TEST_ASSERT_TRUE(slow.setTarget(0.0f));
    TEST_ASSERT_TRUE(slow.hasTarget);
    TEST_ASSERT_EQUAL_INT8(0, slow.stallLatchSign);
    // A jog releases it too: the same T then drives again.
    slow.stallLatchSign = 1;
    slow.manualDrive(0);
    TEST_ASSERT_TRUE(slow.setTarget(0.5f));
    TEST_ASSERT_TRUE(slow.hasTarget);
    // And a stop (H): stopMotor.
    slow.stallLatchSign = 1;
    slow.stopMotor();
    TEST_ASSERT_TRUE(slow.setTarget(0.5f));
    TEST_ASSERT_TRUE(slow.hasTarget);

    // A pot joint stopped by ERR STALL at PWM 60 or more is not re-armed every
    // second by a re-sent T: one ERR STALL, then it stays stopped.
    g_fakeQuadrature = false;
    auto pot = makeActuator();
    pot.init();
    pot.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    now = 0;
    TEST_ASSERT_TRUE(pot.setTarget(1));  // analog 500, target 1023: PWM up
    int trips = 0;
    drivenMs = 0;
    bool stopped = false;
    for (int tick = 1; tick <= 1000; ++tick)
    {
        now += 10;
        if (tick % 30 == 0) TEST_ASSERT_TRUE(pot.setTarget(1));
        pot.update();
        if (pot.encFlags & ACTUATOR_FLAG_ERR_STALL)
        {
            ++trips;
            pot.encFlags = static_cast<uint8_t>(pot.encFlags & ~ACTUATOR_FLAG_ERR_STALL);
        }
        if (!pot.hasTarget) stopped = true;
        if (stopped && pot.currentPwm != 0) drivenMs += 10;
    }
    TEST_ASSERT_EQUAL_INT(1, trips);
    TEST_ASSERT_EQUAL_UINT32(0, drivenMs);
    TEST_ASSERT_EQUAL_INT8(1, pot.stallLatchSign);
    TEST_ASSERT_TRUE(pot.setTarget(0));  // the other way
    TEST_ASSERT_TRUE(pot.hasTarget);
}

// A quiet stop (below PWM 60) holds only its own target: a T within the 12-edge
// deadband of it stays stopped, but a farther T the same way re-arms the guard
// and drives at the PWM its own error gives (F2-1: the normal end of a short T
// step must not freeze the joint in that direction). The ERR STALL guard still
// protects that new drive.
static void test_quiet_stop_holds_only_its_own_target()
{
    g_fakeQuadrature = true;
    auto hip = makeActuator(0);
    hip.init();
    hip.setControlConfig(LinearActuator::ControlConfig(255, 10, 20, 10, 2.0f));
    makeReadyHall(hip, 0, 1000);
    g_fakeHallCount[0] = -460;  // x = 460
    now = 0;
    TEST_ASSERT_TRUE(hip.setTarget(0.5f));  // target 500, error 40: PWM 40
    TEST_ASSERT_EQUAL_INT(500, hip.currentTarget);
    for (int i = 0; i < 400 && hip.hasTarget; ++i)
    {
        now += 10;
        hip.update();
    }
    TEST_ASSERT_FALSE(hip.hasTarget);
    TEST_ASSERT_EQUAL_UINT8(0, hip.encFlags & ACTUATOR_FLAG_ERR_STALL);
    TEST_ASSERT_EQUAL_INT8(1, hip.stallLatchSign);

    // Re-sent, or a few edges off (inside the deadband): still stopped.
    TEST_ASSERT_TRUE(hip.setTarget(0.5f));
    TEST_ASSERT_FALSE(hip.hasTarget);
    TEST_ASSERT_TRUE(hip.setTarget(0.51f));
    TEST_ASSERT_FALSE(hip.hasTarget);
    TEST_ASSERT_EQUAL_INT8(1, hip.stallLatchSign);

    // A farther target the same way drives.
    TEST_ASSERT_TRUE(hip.setTarget(0.95f));  // target 950, error 490: PWM 150
    TEST_ASSERT_TRUE(hip.hasTarget);
    TEST_ASSERT_EQUAL_INT(950, hip.currentTarget);
    TEST_ASSERT_EQUAL_INT8(0, hip.stallLatchSign);
    uint32_t drivenMs = 0;
    for (int i = 0; i < 300 && hip.hasTarget; ++i)
    {
        now += 10;
        hip.update();
        if (hip.currentPwm != 0) drivenMs += 10;
    }
    TEST_ASSERT_TRUE(drivenMs > 0);
    TEST_ASSERT_TRUE(drivenMs <= 1100);  // blocked: ERR STALL after 1 s
    TEST_ASSERT_FALSE(hip.hasTarget);
    TEST_ASSERT_TRUE(hip.encFlags & ACTUATOR_FLAG_ERR_STALL);
    g_fakeHallCount[0] = 0;
}

// A hall hip's saved range is its whole stroke [0, 1069]: like a knee's it keeps
// its home out to the range plus the 60-edge margin (it can reach 941-1003
// edges with its knee curled, inside the stroke) and loses it beyond, whatever
// its stroke. T keeps 10 edges inside the extend switch.
static void test_range_guard_uses_a_hips_stroke_plus_margin()
{
    g_fakeQuadrature = true;
    auto hip = makeActuator(0);
    hip.init();
    makeReadyHall(hip, 0, CAL_HIP_STROKE_EDGES);
    g_fakeHallCount[0] = -1003;  // x = 1003: the body contact, inside the stroke
    hip.update();
    TEST_ASSERT_EQUAL_CHAR('E', hip.sensorStateCode());
    TEST_ASSERT_EQUAL_UINT8(0, hip.encFlags & ACTUATOR_FLAG_ERR_HOME);
    TEST_ASSERT_TRUE(hip.setTarget(1.0f));
    TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES - ACTUATOR_ENC_HIP_EXTEND_INSET_EDGES, hip.currentTarget);
    hip.stopMotor();
    g_fakeHallCount[0] = -(CAL_HIP_STROKE_EDGES + ACTUATOR_ENC_RANGE_MARGIN_EDGES);  // x = 1129
    hip.update();
    TEST_ASSERT_EQUAL_CHAR('E', hip.sensorStateCode());
    TEST_ASSERT_EQUAL_UINT8(0, hip.encFlags & ACTUATOR_FLAG_ERR_HOME);
    g_fakeHallCount[0] -= 1;  // x = 1130: a wrong home
    hip.update();
    TEST_ASSERT_EQUAL_CHAR('x', hip.sensorStateCode());
    TEST_ASSERT_TRUE(hip.encFlags & ACTUATOR_FLAG_ERR_HOME);

    // Any other range keeps its own bound: [0, 701] trips at 762 (there is no
    // longer a fixed-sweep range with a wider bound).
    g_fakeHallCount[0] = 0;
    auto other = makeActuator(0);
    other.init();
    makeReadyHall(other, 0, 701);
    g_fakeHallCount[0] = -(701 + ACTUATOR_ENC_RANGE_MARGIN_EDGES);
    other.update();
    TEST_ASSERT_EQUAL_CHAR('E', other.sensorStateCode());
    g_fakeHallCount[0] -= 1;
    other.update();
    TEST_ASSERT_EQUAL_CHAR('x', other.sensorStateCode());
}

// A homed encoder more than 60 edges outside its stroke clears its home.
static void test_range_guard_clears_a_wrong_home()
{
    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    actuator.init();
    makeReadyHall(actuator, 0, 1000);
    g_fakeHallCount[0] = -1060;
    actuator.update();
    TEST_ASSERT_EQUAL_CHAR('E', actuator.sensorStateCode());
    g_fakeHallCount[0] = 60;
    actuator.update();
    TEST_ASSERT_EQUAL_CHAR('E', actuator.sensorStateCode());
    TEST_ASSERT_TRUE(actuator.setTarget(0.5f));
    g_fakeHallCount[0] = 61;  // x = -61
    actuator.update();
    TEST_ASSERT_EQUAL_CHAR('x', actuator.sensorStateCode());
    TEST_ASSERT_TRUE(actuator.encFlags & ACTUATOR_FLAG_ERR_HOME);
    TEST_ASSERT_FALSE(actuator.hasTarget);

    makeReadyHall(actuator, 0, 1000);  // homed again at count 61
    g_fakeHallCount[0] = 61 - 1061;    // x = 1061, no target
    actuator.update();
    TEST_ASSERT_EQUAL_CHAR('x', actuator.sensorStateCode());
}

// A Pot joint whose hall counts 12 edges was swapped for an encoder.
static void test_pot_joint_that_counts_edges_is_demoted()
{
    g_fakeQuadrature = true;
    auto actuator = makeActuator(0);
    actuator.init();
    actuator.setSensor(JointSensor::Pot);
    actuator.minStop = 100;
    actuator.maxStop = 900;
    actuator.encFlags |= ACTUATOR_FLAG_LIMITS_VALID;
    TEST_ASSERT_TRUE(actuator.setTarget(1));
    g_fakeHallCount[0] = 11;
    now = 10;
    actuator.update();
    assertKind(JointSensor::Pot, actuator);
    TEST_ASSERT_TRUE(actuator.hasTarget);
    g_fakeHallCount[0] = -12;
    now = 20;
    actuator.update();
    assertKind(JointSensor::Unknown, actuator);
    TEST_ASSERT_FALSE(actuator.hasTarget);
    TEST_ASSERT_EQUAL_INT(0, actuator.maxStop);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
    g_fakeHallCount[0] = 12;
    actuator.setSensor(JointSensor::Pot);
    actuator.update();
    assertKind(JointSensor::Unknown, actuator);

    // Without quadrature halls the counts are never evidence.
    g_fakeQuadrature = false;
    auto plain = makeActuator(0);
    plain.init();
    g_fakeHallCount[0] = 500;
    plain.update();
    assertKind(JointSensor::Pot, plain);
}

static void test_manager_dispatch_hold_and_telemetry()
{
    auto first = makeActuator();
    LinearActuator second("FRHY", 5, 6, 7, 8, 9);
    LinearActuator *acts[] = {&first, &second};
    ActuatorManager manager(acts, 2);
    manager.initAll();
    manager.handleJog("FRHY", -100);
    manager.handleJog("missing", 200);
    TEST_ASSERT_EQUAL_INT(-100, second.currentPwm);
    TEST_ASSERT_EQUAL_INT(0, first.currentPwm);
    Command commands[] = {{"missing", 1}, {"FRHY", 0.5f}, {"FLHY", 1}};
    manager.applyCommands(commands, 3);
    TEST_ASSERT_EQUAL_INT(511, second.currentTarget);
    TEST_ASSERT_EQUAL_INT(1023, first.currentTarget);
    now = 10;
    manager.updateAll();
    TEST_ASSERT_EQUAL_INT(5, first.currentPwm);
    manager.holdAll();
    TEST_ASSERT_FALSE(first.hasTarget);
    TEST_ASSERT_FALSE(second.hasTarget);
    TEST_ASSERT_EQUAL_INT(0, second.currentPwm);
    Print out;
    manager.printTelemetry(out);
    TEST_ASSERT_NOT_NULL(std::strstr(out.output.c_str(), ";FRHY"));
    TEST_ASSERT_NULL(std::strchr(out.output.c_str(), '\n'));
    TEST_ASSERT_TRUE(Serial.output.empty());  // no SENS without quadrature halls
    ActuatorManager empty(nullptr, 0);
    empty.initAll();
    empty.handleJog("FLHY", 100);
    empty.applyCommands(nullptr, 0);
    empty.updateAll();
    empty.holdAll();
    out.output.clear();
    empty.printTelemetry(out);
    TEST_ASSERT_TRUE(out.output.empty());
}

// "SENS <ROLE> <NAME>:<c> ..." on quadrature builds: first loop, on any change,
// and every 2 s; never without quadrature halls or without an output.
static void test_sensor_state_line_on_change_and_period()
{
    auto plainFirst = makeActuator(0);
    LinearActuator *plainActs[] = {&plainFirst};
    ActuatorManager plain(plainActs, 1);
    Print plainOut;
    plain.setCalibrationOutput(&plainOut, ROLE_LEFT);
    plain.initAll();
    plain.updateAll();
    TEST_ASSERT_TRUE(plainOut.output.empty());

    g_fakeQuadrature = true;
    auto first = makeActuator(0);
    LinearActuator second("FLHL", 5, 6, 7, 8, 9, 1);
    LinearActuator *acts[] = {&first, &second};
    ActuatorManager manager(acts, 2);
    Print out;
    manager.setCalibrationOutput(&out, ROLE_LEFT);
    manager.initAll();
    now = 100;
    manager.updateAll();
    TEST_ASSERT_EQUAL_STRING("SENS LEFT FLHY:? FLHL:?\n", out.output.c_str());
    out.output.clear();
    now = 2099;
    manager.updateAll();
    TEST_ASSERT_TRUE(out.output.empty());
    second.setSensor(JointSensor::Hall);
    manager.updateAll();
    TEST_ASSERT_EQUAL_STRING("SENS LEFT FLHY:? FLHL:e\n", out.output.c_str());
    out.output.clear();
    now = 4098;
    manager.updateAll();
    TEST_ASSERT_TRUE(out.output.empty());
    now = 4099;
    manager.updateAll();
    TEST_ASSERT_EQUAL_STRING("SENS LEFT FLHY:? FLHL:e\n", out.output.c_str());

    out.output.clear();
    manager.setCalibrationOutput(nullptr, ROLE_LEFT);
    first.setSensor(JointSensor::Pot);
    manager.updateAll();
    TEST_ASSERT_TRUE(out.output.empty());
    manager.setCalibrationOutput(&out, ROLE_LEFT);
    manager.updateAll();
    TEST_ASSERT_EQUAL_STRING("SENS LEFT FLHY:P FLHL:e\n", out.output.c_str());
}

// Guard trips print once through the calibration output; the pending bits clear.
static void test_pending_errors_print_once()
{
    g_fakeQuadrature = true;
    LinearActuator hip("FLHL", 0, 1, 2, 3, 4, 0);
    LinearActuator pot("FLKL", 5, 6, 7, 8, 9, 1);
    LinearActuator *acts[] = {&hip, &pot};
    ActuatorManager manager(acts, 2);
    Print out;
    manager.setCalibrationOutput(&out, ROLE_FRONT);
    manager.initAll();
    makeReadyHall(hip, 0, 1000);
    g_fakeHallCount[0] = 61;  // x = -61: outside the stroke
    pot.encFlags |= ACTUATOR_FLAG_ERR_STALL;
    manager.updateAll();
    TEST_ASSERT_TRUE(contains(out.output, "ERR HOME FLHL lost (x=-61)\n"));
    TEST_ASSERT_TRUE(contains(out.output, "ERR STALL FLKL\n"));
    TEST_ASSERT_EQUAL_UINT8(0, pot.encFlags);
    hip.encFlags |= ACTUATOR_FLAG_ERR_STALL | ACTUATOR_FLAG_ERR_HOME;
    out.output.clear();
    manager.updateAll();
    TEST_ASSERT_TRUE(contains(out.output, "ERR STALL FLHL\nERR HOME FLHL lost (x=-61)\n"));
    out.output.clear();
    manager.updateAll();
    TEST_ASSERT_FALSE(contains(out.output, "ERR"));

    // Without an output the bits still clear.
    manager.setCalibrationOutput(nullptr, ROLE_FRONT);
    pot.encFlags |= ACTUATOR_FLAG_ERR_STALL;
    manager.updateAll();
    TEST_ASSERT_EQUAL_UINT8(0, pot.encFlags);
}

// T to joints without a position: refused, one ERR line per second per board.
static void test_refused_targets_are_reported_and_rate_limited()
{
    g_fakeQuadrature = true;
    LinearActuator yaw("FLHY", 0, 1, 2, 3, 4, 0);
    LinearActuator hip("FLHL", 5, 6, 7, 8, 9, 1);
    LinearActuator knee("FLKL", 10, 11, 12, 13, 14, 2);
    LinearActuator *acts[] = {&yaw, &hip, &knee};
    ActuatorManager manager(acts, 3);
    Print out;
    manager.setCalibrationOutput(&out, ROLE_RIGHT);
    manager.initAll();
    makeReadyHall(hip, 0, 1000);
    Command commands[] = {{"FLKL", 0.5f}, {"FLHL", 0.5f}, {"FLHY", 0.5f}, {"FLKL", 0.2f}, {"nope", 1}};
    now = 5;
    manager.applyCommands(commands, 5);
    TEST_ASSERT_EQUAL_STRING("ERR T refused FLHY FLKL (no position)\n", out.output.c_str());
    TEST_ASSERT_TRUE(hip.hasTarget);
    TEST_ASSERT_FALSE(yaw.hasTarget);
    out.output.clear();
    now = 1004;
    manager.applyCommands(commands, 5);
    TEST_ASSERT_TRUE(out.output.empty());
    now = 1005;
    manager.applyCommands(commands + 1, 1);  // accepted only: no line
    TEST_ASSERT_TRUE(out.output.empty());
    manager.applyCommands(commands + 2, 1);
    TEST_ASSERT_EQUAL_STRING("ERR T refused FLHY (no position)\n", out.output.c_str());
    out.output.clear();
    manager.setCalibrationOutput(nullptr, ROLE_RIGHT);
    now = 5000;
    manager.applyCommands(commands, 5);
    TEST_ASSERT_TRUE(out.output.empty());
}

// The refused set is a bit per actuator index; joints past 32 are refused unnamed.
static void test_refused_targets_beyond_thirty_two_actuators()
{
    g_fakeQuadrature = true;
    std::vector<std::string> names;
    for (int i = 0; i < 33; ++i) names.push_back("J" + std::to_string(i));
    std::vector<LinearActuator> storage;
    storage.reserve(33);
    for (int i = 0; i < 33; ++i) storage.emplace_back(names[i].c_str(), 0, 1, 2, 3, 4);
    std::vector<LinearActuator *> acts;
    for (auto &actuator : storage) acts.push_back(&actuator);
    ActuatorManager manager(acts.data(), acts.size());
    Print out;
    manager.setCalibrationOutput(&out, ROLE_FRONT);
    manager.initAll();
    Command commands[] = {{"J32", 0.5f}, {"J31", 0.5f}};
    manager.applyCommands(commands, 2);
    TEST_ASSERT_EQUAL_STRING("ERR T refused J31 (no position)\n", out.output.c_str());
    TEST_ASSERT_FALSE(storage[32].hasTarget);
}

// Rod travel limits of the calibration rig's plant (pot counts).
constexpr int kRigLow = 100;
constexpr int kRigHigh = 900;
// Hall rig: hips travel 0..kHallStroke edges, knees 0..kHallKneeStroke (FL's
// CAL_KNEE_STROKE_EDGES: the table-safe SWEEP's knee travel window is 745-825);
// count = kHallCountOffset - travel (extend counts down). Hall B rests high on
// the pot pin.
constexpr int kHallStroke = 1000;
constexpr int kHallKneeStroke = CAL_KNEE_STROKE_EDGES;
inline int hallStrokeOf(int joint) { return joint % 3 == 2 ? kHallKneeStroke : kHallStroke; }
constexpr int kHallCountOffset = 37;
constexpr int kHallBRest = 1016;

// Six actuators on distinct pins: slot j uses pins 5j..5j+4 (PWM_R, PWM_L, EN, IS, POT);
// with halls, hall slot j.
struct CalibrationRig
{
    const char *names[6] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    const bool hall;
    LinearActuator storage[6];
    LinearActuator *acts[6];
    ActuatorManager manager{acts, 6};
    Print output;
    int travel[6] = {500, 500, 500, 500, 500, 500};
    int maxEnabled = 0;
    int maxMagnitude = 0;
    bool yawDriven = false;

    explicit CalibrationRig(bool withHalls = false)
        : hall(withHalls),
          storage{
              LinearActuator(names[0], 0, 1, 2, 3, 4, withHalls ? 0 : -1),
              LinearActuator(names[1], 5, 6, 7, 8, 9, withHalls ? 1 : -1),
              LinearActuator(names[2], 10, 11, 12, 13, 14, withHalls ? 2 : -1),
              LinearActuator(names[3], 15, 16, 17, 18, 19, withHalls ? 3 : -1),
              LinearActuator(names[4], 20, 21, 22, 23, 24, withHalls ? 4 : -1),
              LinearActuator(names[5], 25, 26, 27, 28, 29, withHalls ? 5 : -1)}
    {
        for (int i = 0; i < 6; ++i)
        {
            acts[i] = &storage[i];
            analog[5 * i + 4] = hall ? kHallBRest : 500;
            if (hall) g_fakeHallCount[i] = kHallCountOffset - travel[i];
        }
        manager.initAll();
        manager.setCalibrationOutput(&output, ROLE_FRONT);
    }

    int signedPwm(int joint) const
    {
        if (digital[5 * joint + 2] != HIGH) return 0;
        return pwm[5 * joint] - pwm[5 * joint + 1];
    }

    // One 10 ms loop: the plant moves, then the firmware runs. A pot moves 2
    // counts per tick while driven; a hall rod moves PWM / 40 edges per tick,
    // and its IS pin reads 12 while it runs, 8 driven back into its retracted
    // end (the limit switch opens) and 18 pressed into its extended end (a
    // mechanical stop); 0 undriven.
    void tick()
    {
        now += 10;
        for (int joint = 0; joint < 6; ++joint)
        {
            const int drive = signedPwm(joint);
            if (hall)
            {
                travel[joint] = std::max(0, std::min(hallStrokeOf(joint), travel[joint] + drive / 40));
                g_fakeHallCount[joint] = kHallCountOffset - travel[joint];
                analog[5 * joint + 3] = drive == 0                          ? 0
                    : drive < 0 && travel[joint] == 0                       ? 8
                    : drive > 0 && travel[joint] == hallStrokeOf(joint)     ? 18
                                                                            : 12;
                continue;
            }
            int &pot = analog[5 * joint + 4];
            if (drive > 0) pot = std::min(kRigHigh, pot + 2);
            if (drive < 0) pot = std::max(kRigLow, pot - 2);
        }
        manager.updateAll();
        int enabled = 0;
        for (int joint = 0; joint < 6; ++joint)
        {
            const int drive = signedPwm(joint);
            if (drive == 0) continue;
            ++enabled;
            maxMagnitude = std::max(maxMagnitude, std::abs(drive));
            if (joint == 0 || joint == 3) yawDriven = true;
        }
        maxEnabled = std::max(maxEnabled, enabled);
    }

    void runCalibration()
    {
        for (int i = 0; i < 40000 && manager.isCalibrating(); ++i) tick();
    }
};

static void test_calibration_sweeps_hips_and_knees_one_at_a_time_and_saves()
{
    CalibrationRig rig;
    rig.acts[0]->minStop = 11;
    rig.acts[0]->maxStop = 999;
    rig.acts[1]->setTarget(1);
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_TRUE(rig.manager.isCalibrating());
    TEST_ASSERT_FALSE(rig.acts[1]->hasTarget);
    TEST_ASSERT_FALSE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM));
    rig.runCalibration();

    TEST_ASSERT_FALSE(rig.manager.isCalibrating());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Completed),
        static_cast<int>(rig.manager.calibrationOutcome()));
    TEST_ASSERT_EQUAL_INT(1, rig.maxEnabled);
    TEST_ASSERT_EQUAL_INT(CAL_DEFAULT_PWM, rig.maxMagnitude);
    TEST_ASSERT_FALSE(rig.yawDriven);
    TEST_ASSERT_EQUAL_INT(11, rig.acts[0]->minStop);
    TEST_ASSERT_EQUAL_INT(999, rig.acts[0]->maxStop);
    TEST_ASSERT_EQUAL_INT(0, rig.acts[3]->minStop);
    TEST_ASSERT_EQUAL_INT(1023, rig.acts[3]->maxStop);
    for (int joint : {1, 2, 4, 5})
    {
        TEST_ASSERT_INT_WITHIN(1, kRigLow, rig.acts[joint]->minStop);
        TEST_ASSERT_INT_WITHIN(1, kRigHigh, rig.acts[joint]->maxStop);
        TEST_ASSERT_TRUE(rig.acts[joint]->encFlags & ACTUATOR_FLAG_LIMITS_VALID);
    }
    for (auto *actuator : rig.acts)
    {
        TEST_ASSERT_EQUAL_INT(0, actuator->currentPwm);
        TEST_ASSERT_FALSE(actuator->hasTarget);
    }
    TEST_ASSERT_EQUAL_UINT(3, EEPROM.writes);  // magic cleared, record, magic
    TEST_ASSERT_NOT_NULL(std::strstr(rig.output.output.c_str(), "CAL FRONT PASS SEEK pwm=128\n"));
    TEST_ASSERT_NOT_NULL(std::strstr(rig.output.output.c_str(), "CAL FRONT PASS_DONE SWEEP saved\n"));
    TEST_ASSERT_FALSE(contains(rig.output.output, "SENS "));

    // Persisted: a reload restores the measured hips/knees and the untouched yaws.
    for (auto *actuator : rig.acts)
    {
        actuator->minStop = 0;
        actuator->maxStop = 1023;
    }
    rig.manager.loadCalibration();
    TEST_ASSERT_EQUAL_INT(11, rig.acts[0]->minStop);
    TEST_ASSERT_INT_WITHIN(1, kRigHigh, rig.acts[5]->maxStop);
}

// All-hall board: SEEK homes every hip and knee, SWEEP saves canonical strokes
// and kinds, yaws stay Unknown, then T works on a ready hip and is refused on a yaw.
static void test_hall_board_calibrates_homes_and_accepts_targets()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    for (int joint = 0; joint < 6; ++joint)
        TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[5 * joint + 4]);
    rig.tick();
    TEST_ASSERT_TRUE(contains(rig.output.output, "SENS FRONT FLHY:? FLHL:? FLKL:? FRHY:? FRHL:? FRKL:?\n"));
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM));
    rig.runCalibration();

    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Completed),
        static_cast<int>(rig.manager.calibrationOutcome()));
    TEST_ASSERT_FALSE(rig.yawDriven);
    TEST_ASSERT_EQUAL_INT(1, rig.maxEnabled);
    for (int joint : {0, 3})
    {
        assertKind(JointSensor::Unknown, *rig.acts[joint]);
        TEST_ASSERT_EQUAL_INT(0, rig.acts[joint]->maxStop);
    }
    for (int joint : {1, 4})
    {
        // A hall hip saves its derived stroke, not what its table-safe sweep reached.
        assertKind(JointSensor::Hall, *rig.acts[joint]);
        TEST_ASSERT_EQUAL_INT(0, rig.acts[joint]->minStop);
        TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, rig.acts[joint]->maxStop);
        TEST_ASSERT_EQUAL_INT(kHallCountOffset, rig.acts[joint]->hallHome);
        TEST_ASSERT_EQUAL_CHAR('E', rig.acts[joint]->sensorStateCode());
    }
    for (int joint : {2, 5})
    {
        assertKind(JointSensor::Hall, *rig.acts[joint]);
        TEST_ASSERT_EQUAL_INT(-kHallKneeStroke, rig.acts[joint]->minStop);
        TEST_ASSERT_EQUAL_INT(0, rig.acts[joint]->maxStop);
        TEST_ASSERT_EQUAL_INT(kHallCountOffset - kHallKneeStroke, rig.acts[joint]->hallHome);
        TEST_ASSERT_EQUAL_FLOAT(1.0f, rig.acts[joint]->getPos());
    }
    TEST_ASSERT_TRUE(contains(rig.output.output, "CAL FRONT HOME FLKL count=-748\n"));
    TEST_ASSERT_TRUE(contains(rig.output.output, "SENS FRONT FLHY:? FLHL:E FLKL:E FRHY:? FRHL:E FRKL:E\n"));
    // Saved: v2 with the kinds (hips and knees Hall, yaws Unknown).
    TEST_ASSERT_EQUAL_UINT(3, EEPROM.writes);  // magic cleared, record, magic
    ActuatorManager::CalData saved{};
    EEPROM.get(EEPROM_ACTUATOR_CAL_ADDR, saved);
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::CAL_DATA_MAGIC, saved.magic);
    TEST_ASSERT_EQUAL_HEX16(0x0A28, saved.sensorBits);
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::calDataCheck(saved), saved.check);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, saved.maxVals[1]);
    TEST_ASSERT_EQUAL_INT(-kHallKneeStroke, saved.minVals[2]);
    TEST_ASSERT_TRUE(contains(rig.output.output, "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)\n"));

    // One leg's moves, in the order the table-safe schedule runs them: SEEK is
    // knee curl, hip curl, knee curl; SWEEP is knee curl, hip curl, knee curl
    // again, knee uncurl (its travel judged, the KNEE line), knee park (521 from
    // its switch: -264 at a 785 travel, flagged as FL's values), hip sweep (to 840 edges, by
    // count), hip curl (which proves the home again and saves the derived
    // stroke), knee curl (which saves the knee's).
    static const char *const kFlLegOrder[] = {
        "CAL FRONT PASS SEEK pwm=128\n",
        "CAL FRONT MOVE FLKL curl\n",
        "CAL FRONT MOVE FLHL curl\n",
        "CAL FRONT MOVE FLKL curl\n",
        "CAL FRONT MOVE FRKL curl\n",
        "CAL FRONT PASS_DONE SEEK\n",
        "CAL FRONT PASS SWEEP pwm=128\n",
        "CAL FRONT MOVE FLKL curl\n",
        "CAL FRONT MOVE FLHL curl\n",
        "CAL FRONT MOVE FLKL curl\n",
        "CAL FRONT MOVE FLKL uncurl\n",
        "CAL FRONT KNEE FLKL home=-748 sw=37 travel=785 park=-264 pwm=128 (FL 785)\n",
        "CAL FRONT MOVE FLKL curl park\n",
        "CAL FRONT PARK FLKL hall=-263 fromSwitch=522 target=-264 (FL values)\n",
        "CAL FRONT MOVE FLHL uncurl sweep\n",
        "CAL FRONT END FLHL uncurl hall=840 travel=840\n",
        "CAL FRONT KNEE FLKL sweep fromSwitch=522..522\n",
        "CAL FRONT MOVE FLHL curl\n",
        "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)\n",
        "CAL FRONT MOVE FLKL curl\n",
        "CAL FRONT JOINT FLKL min=-785 max=0 edges\n",
        "CAL FRONT MOVE FRKL curl\n",
    };
    size_t cursor = 0;
    for (const char *line : kFlLegOrder)
    {
        const size_t at = rig.output.output.find(line, cursor);
        TEST_ASSERT_TRUE_MESSAGE(at != std::string::npos,
                                 (std::string("missing in order: ") + line + rig.output.output).c_str());
        cursor = at + 1;
    }
    // FLKL curls: two in SEEK, then three in SWEEP (before the hip, before the
    // uncurl, after the hip's curl back); the park is its own move line.
    TEST_ASSERT_EQUAL_UINT(5, occurrences(rig.output.output, "MOVE FLKL curl\n"));
    TEST_ASSERT_EQUAL_UINT(1, occurrences(rig.output.output, "MOVE FLKL uncurl\n"));
    TEST_ASSERT_EQUAL_UINT(2, occurrences(rig.output.output, " curl park\n"));
    TEST_ASSERT_EQUAL_UINT(2, occurrences(rig.output.output, " uncurl sweep\n"));
    TEST_ASSERT_FALSE(contains(rig.output.output, "WARN"));
    TEST_ASSERT_FALSE(contains(rig.output.output, "FAIL"));

    rig.output.output.clear();
    Command commands[] = {{"FLHL", 0.5f}, {"FLHY", 0.5f}};
    rig.manager.applyCommands(commands, 2);
    TEST_ASSERT_TRUE(contains(rig.output.output, "ERR T refused FLHY (no position)\n"));
    for (int i = 0; i < 1000; ++i) rig.tick();
    TEST_ASSERT_INT_WITHIN(60, CAL_HIP_STROKE_EDGES / 2, rig.acts[1]->getRawPos());
    TEST_ASSERT_FALSE(contains(rig.output.output, "ERR STALL"));
    TEST_ASSERT_EQUAL_INT(0, rig.signedPwm(0));
}

// The T guards never run during calibration: a swapped pot is only demoted once
// calibration has ended.
static void test_guards_do_not_run_while_calibrating()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    rig.acts[0]->setSensor(JointSensor::Pot);
    g_fakeHallCount[0] = 50;
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    for (int i = 0; i < 50; ++i) rig.tick();
    g_fakeHallCount[0] = 50;
    TEST_ASSERT_TRUE(rig.manager.isCalibrating());
    assertKind(JointSensor::Pot, *rig.acts[0]);
    rig.manager.holdAll();
    rig.manager.updateAll();
    assertKind(JointSensor::Unknown, *rig.acts[0]);
}

static void test_sensors_refresh_every_loop_while_calibrating()
{
    CalibrationRig rig;
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    analog[5 * 0 + 4] = 700;  // a yaw the sequencer never drives
    for (int i = 0; i < 40; ++i) rig.tick();
    TEST_ASSERT_TRUE(rig.manager.isCalibrating());
    TEST_ASSERT_GREATER_THAN_INT(650, rig.acts[0]->getRawPos());
}

static void test_hold_aborts_calibration_and_motors_stay_off()
{
    CalibrationRig rig;
    TEST_ASSERT_FALSE(rig.manager.abortCalibration('H'));
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM));
    for (int i = 0; i < 100; ++i) rig.tick();
    TEST_ASSERT_NOT_EQUAL(0, rig.signedPwm(2));
    rig.manager.holdAll();
    TEST_ASSERT_FALSE(rig.manager.isCalibrating());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Aborted),
        static_cast<int>(rig.manager.calibrationOutcome()));
    for (int i = 0; i < 500; ++i) rig.tick();
    for (int joint = 0; joint < 6; ++joint)
        TEST_ASSERT_EQUAL_INT(0, rig.signedPwm(joint));
    TEST_ASSERT_EQUAL_UINT(0, EEPROM.writes);
    TEST_ASSERT_NOT_NULL(std::strstr(rig.output.output.c_str(), "CAL FRONT ABORTED H\n"));
    Print reason;
    PrintCalibrationSink reasonSink(&reason);
    rig.manager.printCalibrationFailure(reasonSink);
    TEST_ASSERT_EQUAL_STRING("ABORTED H", reason.output.c_str());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Aborted),
        static_cast<int>(rig.manager.calibrationSequencer().outcome()));
    rig.manager.resetCalibration();
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Idle),
        static_cast<int>(rig.manager.calibrationOutcome()));
}

static void test_calibration_requires_six_actuators_and_an_output()
{
    auto first = makeActuator();
    LinearActuator *acts[] = {&first};
    ActuatorManager small(acts, 1);
    TEST_ASSERT_FALSE(small.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM));
    CalibrationRig rig;
    rig.manager.setCalibrationOutput(nullptr, ROLE_FRONT);
    TEST_ASSERT_FALSE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM));
    rig.manager.setCalibrationOutput(&rig.output, ROLE_LEFT);
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, rig.manager.calibrationOutputRole());
    TEST_ASSERT_FALSE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_PWM_MAX + 1));
    TEST_ASSERT_FALSE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_PWM_MIN - 1));  // 109: below the floor
    TEST_ASSERT_FALSE(rig.manager.startCalibration(CAL_PASS_ALL, 80));               // an old PWM
    TEST_ASSERT_FALSE(rig.manager.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM, 0x80));
    TEST_ASSERT_EQUAL_INT(110, CAL_PWM_MIN);
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_SWEEP, CAL_PWM_MIN));
    TEST_ASSERT_EQUAL_UINT8(CAL_DEFAULT_OPTIONS, rig.manager.calibrationSequencer().options());
    TEST_ASSERT_NOT_NULL(std::strstr(rig.output.output.c_str(), "CAL LEFT PASS SWEEP pwm=110\n"));
    rig.manager.holdAll();
    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_EQUAL_UINT8(CAL_OPT_KNEE_NUDGE, rig.manager.calibrationSequencer().options());
}

// A C run's first move is a knee curl. When the leg's hall hip had its home
// cleared by a guard while it reads far out, the knee must not move: the run is
// refused with "FLHL out at <x> edges: jog it to its curl end first" and no
// MOVE line. A hip that reads 560 edges or less is not "out".
// travelHomed: the rod's travel (edges) when its home was set; travelNow: where
// it reads afterwards; hi: the hip's stroke (a small one lets the range guard
// trip within the rig's 1000-edge rod).
static void runWithLostHipHome(int travelHomed, int travelNow, int hi, bool expectRefused)
{
    setUp();
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    rig.travel[1] = travelHomed;
    g_fakeHallCount[1] = kHallCountOffset - travelHomed;
    makeReadyHall(*rig.acts[1], 0, hi);
    rig.travel[1] = travelNow;
    g_fakeHallCount[1] = kHallCountOffset - travelNow;
    rig.manager.updateAll();  // the range guard clears the home
    const std::string label = "home@" + std::to_string(travelHomed) + " now@" + std::to_string(travelNow);
    TEST_ASSERT_TRUE_MESSAGE(rig.acts[1]->encFlags & ACTUATOR_FLAG_HOME_LOST, label.c_str());
    const int x = travelNow - travelHomed;

    TEST_ASSERT_TRUE(rig.manager.startCalibration(CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    for (int i = 0; i < 40; ++i) rig.tick();
    const std::string failLine = "CAL FRONT FAIL FLHL out at " + std::to_string(x) +
        " edges: jog it to its curl end first\n";
    if (expectRefused)
    {
        TEST_ASSERT_FALSE_MESSAGE(rig.manager.isCalibrating(), label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(LegCalibrationOutcome::Failed),
            static_cast<int>(rig.manager.calibrationOutcome()), label.c_str());
        TEST_ASSERT_TRUE_MESSAGE(contains(rig.output.output, failLine.c_str()), (label + "\n" + rig.output.output).c_str());
        TEST_ASSERT_FALSE_MESSAGE(contains(rig.output.output, "MOVE"), (label + "\n" + rig.output.output).c_str());
        for (int joint = 0; joint < 6; ++joint) TEST_ASSERT_EQUAL_INT(0, rig.signedPwm(joint));
        // The refusal moved nothing: no home was cleared by it, no loss added.
        TEST_ASSERT_TRUE(rig.acts[1]->encFlags & ACTUATOR_FLAG_HOME_LOST);
    }
    else
    {
        TEST_ASSERT_FALSE_MESSAGE(contains(rig.output.output, "jog it to its curl end first"), (label + "\n" + rig.output.output).c_str());
        TEST_ASSERT_TRUE_MESSAGE(contains(rig.output.output, "CAL FRONT MOVE FLKL curl\n"), (label + "\n" + rig.output.output).c_str());
    }
}

static void test_calibration_refuses_a_knee_move_while_its_hip_reads_out_with_a_lost_home()
{
    // Past the range by more than its 60-edge margin, far out: refused.
    runWithLostHipHome(0, 800, 700, true);     // x = 800 > 560
    runWithLostHipHome(0, 1000, 700, true);
    // The boundary: 560 is not out (the guard still trips on a [0, 400] hip), 561 is.
    runWithLostHipHome(0, 560, 400, false);
    runWithLostHipHome(0, 561, 400, true);
    // Lost behind its home (reads negative): not out, the knee curl goes ahead.
    runWithLostHipHome(100, 0, 700, false);    // x = -100 < -60
}

// The plant's isHomeLost: false before any home, false while homed, true after a
// guard (the range guard or the T stall guard) cleared the home, false again
// once the joint is re-homed.
static void test_plant_home_lost_follows_the_guards()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    ActuatorManager::CalibrationPlant plant(rig.manager);
    for (int joint = 0; joint < 6; ++joint) TEST_ASSERT_FALSE(plant.isHomeLost(joint));

    // Range guard on hip 1: homed (no loss), then read 61 edges past its range.
    rig.acts[1]->setSensor(JointSensor::Hall);
    TEST_ASSERT_FALSE(plant.isHomeLost(1));  // a hall joint with no home yet
    makeReadyHall(*rig.acts[1], 0, CAL_HIP_STROKE_EDGES);
    TEST_ASSERT_FALSE(plant.isHomeLost(1));
    rig.manager.updateAll();
    TEST_ASSERT_FALSE(plant.isHomeLost(1));
    g_fakeHallCount[1] = rig.acts[1]->hallHome - (CAL_HIP_STROKE_EDGES + ACTUATOR_ENC_RANGE_MARGIN_EDGES + 1);
    rig.manager.updateAll();
    TEST_ASSERT_FALSE(plant.isHomed(1));
    TEST_ASSERT_TRUE(plant.isHomeLost(1));
    plant.setHome(1);
    TEST_ASSERT_FALSE(plant.isHomeLost(1));

    // T stall guard on hip 4: driven at full PWM for a second with no edge.
    makeReadyHall(*rig.acts[4], 0, CAL_HIP_STROKE_EDGES);
    TEST_ASSERT_FALSE(plant.isHomeLost(4));
    TEST_ASSERT_TRUE(rig.acts[4]->setTarget(0.5f));
    for (int i = 0; i < 150 && rig.acts[4]->hasTarget; ++i)
    {
        now += 10;
        rig.manager.updateAll();
    }
    TEST_ASSERT_TRUE(contains(rig.output.output, "ERR STALL"));
    TEST_ASSERT_FALSE(plant.isHomed(4));
    TEST_ASSERT_TRUE(plant.isHomeLost(4));
    // The other joints were never homed, so none of them reads as lost.
    for (int joint : {0, 2, 3, 5}) TEST_ASSERT_FALSE(plant.isHomeLost(joint));
}

// Every plant method reaches the right actuator state.
static void test_calibration_plant_forwards_to_the_actuators()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    ActuatorManager::CalibrationPlant plant(rig.manager);
    TEST_ASSERT_TRUE(rig.acts[1]->setTarget(0.5f) == false);
    plant.drive(1, 70);
    TEST_ASSERT_EQUAL_INT(70, rig.signedPwm(1));
    plant.stopAll();
    TEST_ASSERT_EQUAL_INT(0, rig.signedPwm(1));
    rig.acts[1]->avgPot = 612.7f;
    TEST_ASSERT_EQUAL_INT(612, plant.rawPot(1));
    readings[9] = {100, 101, 102, 103};
    TEST_ASSERT_EQUAL_INT(101, plant.potSnapshot(1));
    TEST_ASSERT_FALSE(plant.isPotOpen(1));
    TEST_ASSERT_TRUE(plant.hasQuadratureHall());
    g_fakeHallCount[1] = -40;
    TEST_ASSERT_EQUAL_INT(-40, plant.hallCount(1));
    TEST_ASSERT_EQUAL_INT(40, plant.position(1));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Unknown), static_cast<int>(plant.sensor(1)));
    plant.setSensor(1, JointSensor::Hall);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Hall), static_cast<int>(plant.sensor(1)));
    TEST_ASSERT_FALSE(plant.isHomed(1));
    plant.setHome(1);
    TEST_ASSERT_TRUE(plant.isHomed(1));
    TEST_ASSERT_EQUAL_INT(0, plant.position(1));
    TEST_ASSERT_FALSE(plant.isHomeLost(1));  // a live home is not a lost one
    plant.clearHome(1);
    TEST_ASSERT_FALSE(plant.isHomed(1));
    TEST_ASSERT_TRUE(plant.isHomeLost(1));
    TEST_ASSERT_EQUAL_CHAR('x', rig.acts[1]->sensorStateCode());

    // The end detector's current: round(10 * avgIS), clamped.
    rig.acts[2]->avgIS = 12.34f;
    TEST_ASSERT_EQUAL_UINT16(123, plant.motorCurrentTenths(2));
    rig.acts[2]->avgIS = 7.96f;
    TEST_ASSERT_EQUAL_UINT16(80, plant.motorCurrentTenths(2));
    rig.acts[2]->avgIS = -1.0f;
    TEST_ASSERT_EQUAL_UINT16(0, plant.motorCurrentTenths(2));
    rig.acts[2]->avgIS = 7000.0f;
    TEST_ASSERT_EQUAL_UINT16(0xFFFF, plant.motorCurrentTenths(2));

    // Current evidence stays latched after the motor stops.
    TEST_ASSERT_FALSE(plant.isCurrentAbsent(2));
    analog[13] = 0;
    rig.acts[2]->avgIS = 0;
    plant.drive(2, 100);
    for (int i = 0; i < 3; ++i) rig.acts[2]->updateSensors();
    plant.stopAll();
    rig.acts[2]->updateSensors();
    TEST_ASSERT_TRUE(plant.isCurrentAbsent(2));

    int limit = -1;
    TEST_ASSERT_FALSE(plant.measuredLimit(1, true, limit));  // Hall
    plant.setSensor(4, JointSensor::Pot);
    TEST_ASSERT_FALSE(plant.measuredLimit(4, true, limit));  // defaults, not measured
    TEST_ASSERT_EQUAL_INT(-1, limit);
    plant.storeLimits(4, 120, 880);
    TEST_ASSERT_TRUE(plant.measuredLimit(4, true, limit));
    TEST_ASSERT_EQUAL_INT(880, limit);
    TEST_ASSERT_TRUE(plant.measuredLimit(4, false, limit));
    TEST_ASSERT_EQUAL_INT(120, limit);
    TEST_ASSERT_EQUAL_INT(0, plant.savedStroke(4));  // Pot
    TEST_ASSERT_EQUAL_INT(0, plant.savedStroke(1));  // Hall without a stroke
    plant.storeLimits(1, 0, 800);
    TEST_ASSERT_EQUAL_INT(800, plant.savedStroke(1));
    TEST_ASSERT_TRUE(rig.acts[1]->encFlags & ACTUATOR_FLAG_LIMITS_VALID);

    now = 4321;
    TEST_ASSERT_EQUAL_UINT32(4321, plant.nowMs());
    TEST_ASSERT_EQUAL_STRING("FLKL", plant.jointName(2));
    TEST_ASSERT_EQUAL_STRING("FRONT", plant.roleLabel());
    rig.output.output.clear();
    plant.out().printText("hello");
    TEST_ASSERT_EQUAL_STRING("hello", rig.output.output.c_str());
    plant.saveLimits();
    TEST_ASSERT_EQUAL_UINT(3, EEPROM.writes);  // magic cleared, record, magic
}

static void test_save_and_load_round_trip()
{
    CalibrationRig rig;
    rig.manager.loadCalibration();
    TEST_ASSERT_EQUAL_INT(1023, rig.acts[0]->maxStop);
    for (int i = 0; i < 6; ++i)
    {
        rig.acts[i]->minStop = 100 + i;
        rig.acts[i]->maxStop = 900 - i;
    }
    rig.manager.saveCalibration();
    TEST_ASSERT_EQUAL_UINT(3, EEPROM.writes);  // magic cleared, record, magic
    TEST_ASSERT_TRUE(contains(rig.output.output, "Limits saved\n"));
    for (auto *actuator : rig.acts)
    {
        actuator->minStop = 0;
        actuator->maxStop = 1023;
    }
    rig.manager.loadCalibration();
    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_INT(100 + i, rig.acts[i]->minStop);
        TEST_ASSERT_EQUAL_INT(900 - i, rig.acts[i]->maxStop);
    }
}

// --- EEPROM record v2 loading ----------------------------------------------

struct StoredJoint
{
    JointSensor kind;
    int lo;
    int hi;
};

static void writeRecord(uint16_t magic, const StoredJoint (&joints)[6], uint16_t extraBits = 0, int checkDelta = 0)
{
    ActuatorManager::CalData data{};
    uint16_t bits = extraBits;
    for (int i = 0; i < 6; ++i)
    {
        data.minVals[i] = static_cast<int16_t>(joints[i].lo);
        data.maxVals[i] = static_cast<int16_t>(joints[i].hi);
        bits = static_cast<uint16_t>(bits | (static_cast<uint16_t>(joints[i].kind) << (2 * i)));
    }
    data.magic = magic;
    data.sensorBits = bits;
    data.check = static_cast<uint16_t>(ActuatorManager::calDataCheck(data) + checkDelta);
    EEPROM.put(EEPROM_ACTUATOR_CAL_ADDR, data);
}

// As a quadrature build saves it: yaws Unknown, a hall hip's derived stroke, knees
// at the shortest and longest plausible strokes, a pot hip on joint 4.
static const StoredJoint kGoodRecord[6] = {
    {JointSensor::Unknown, 0, 0},
    {JointSensor::Hall, 0, CAL_HIP_STROKE_EDGES},
    {JointSensor::Hall, -CAL_HALL_MIN_STROKE_EDGES, 0},
    {JointSensor::Unknown, 0, 0},
    {JointSensor::Pot, 50, 950},
    {JointSensor::Hall, -CAL_HALL_MAX_STROKE_EDGES, 0},
};

static void assertJoint(const LinearActuator &actuator, JointSensor kind, int lo, int hi, bool limitsValid)
{
    assertKind(kind, actuator);
    TEST_ASSERT_EQUAL_INT(lo, actuator.minStop);
    TEST_ASSERT_EQUAL_INT(hi, actuator.maxStop);
    TEST_ASSERT_EQUAL(limitsValid, (actuator.encFlags & ACTUATOR_FLAG_LIMITS_VALID) != 0);
}

static void test_load_v2_record_applies_kinds_limits_and_pin_modes()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, kGoodRecord);
    rig.manager.loadCalibration();
    assertJoint(*rig.acts[0], JointSensor::Unknown, 0, 0, false);
    assertJoint(*rig.acts[1], JointSensor::Hall, 0, 1069, true);
    assertJoint(*rig.acts[2], JointSensor::Hall, -450, 0, true);
    assertJoint(*rig.acts[3], JointSensor::Unknown, 0, 0, false);
    assertJoint(*rig.acts[4], JointSensor::Pot, 50, 950, true);
    assertJoint(*rig.acts[5], JointSensor::Hall, -1500, 0, true);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[19]);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[24]);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[29]);
    TEST_ASSERT_EQUAL_CHAR('e', rig.acts[1]->sensorStateCode());  // homes are never stored
    TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration v2 loaded\n"));
    TEST_ASSERT_FALSE(contains(rig.output.output, "invalid"));
    // The hall hip's 1069 is derived, not found at its switch: loading says so
    // once (the hip only; the knees carry measured strokes).
    TEST_ASSERT_EQUAL_UINT(1, occurrences(rig.output.output, "extend derived"));
    TEST_ASSERT_TRUE(contains(rig.output.output,
        "Calibration joint FLHL extend derived (1069 edges): not switch-verified\n"));

    // Bits value 3 reads as Unknown; a stored Unknown on a pot build is a pot with defaults.
    StoredJoint record[6];
    std::copy(kGoodRecord, kGoodRecord + 6, record);
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, record, 0x3 << 4);  // joint 2: 2 | 3 = 3
    rig.manager.loadCalibration();
    assertJoint(*rig.acts[2], JointSensor::Unknown, 0, 0, false);
    assertJoint(*rig.acts[1], JointSensor::Hall, 0, 1069, true);

    g_fakeQuadrature = false;
    CalibrationRig pots;
    record[1] = {JointSensor::Pot, 200, 800};
    record[2] = {JointSensor::Pot, 210, 810};
    record[5] = {JointSensor::Pot, 220, 820};
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, record);
    pots.manager.loadCalibration();
    assertJoint(*pots.acts[0], JointSensor::Pot, 0, 1023, false);
    assertJoint(*pots.acts[1], JointSensor::Pot, 200, 800, true);
    assertJoint(*pots.acts[5], JointSensor::Pot, 220, 820, true);
}

static void test_load_rejects_corrupt_v2_records()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, kGoodRecord, 0, 1);  // check mismatch
    rig.manager.loadCalibration();
    for (auto *actuator : rig.acts)
        assertJoint(*actuator, JointSensor::Unknown, 0, 0, false);
    TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration record corrupt: defaults\n"));

    rig.output.output.clear();
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, kGoodRecord, 0x1000);  // reserved bit
    rig.manager.loadCalibration();
    assertJoint(*rig.acts[1], JointSensor::Unknown, 0, 0, false);
    TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration record corrupt: defaults\n"));
}

// Each joint is validated on its own; an invalid one gets the defaults and a line.
static void test_load_validates_each_joint()
{
    struct Case
    {
        bool quadrature;
        uint8_t joint;
        StoredJoint stored;
    };
    const Case cases[] = {
        {true, 4, {JointSensor::Pot, -1, 900}},
        {true, 4, {JointSensor::Pot, 500, 500}},
        {true, 4, {JointSensor::Pot, 100, 1024}},
        {true, 0, {JointSensor::Hall, 0, 1069}},       // a yaw is never Hall
        {true, 1, {JointSensor::Hall, 5, 1074}},       // hip curl end not at 0
        {true, 2, {JointSensor::Hall, -900, 5}},       // knee curl end not at 0
        // A hall hip loads only as [0, N] with N within 20 edges of its derived
        // stroke 1069: the old fixed-sweep record [0, 700] (and anything else
        // short of 1049 or past 1089) is invalid and needs a new C.
        {true, 1, {JointSensor::Hall, 0, 700}},
        {true, 1, {JointSensor::Hall, 0, 1048}},
        {true, 1, {JointSensor::Hall, 0, 1090}},
        {true, 1, {JointSensor::Hall, 0, 941}},
        {true, 1, {JointSensor::Hall, 0, CAL_HALL_MAX_STROKE_EDGES}},
        {true, 2, {JointSensor::Hall, -CAL_HALL_MIN_STROKE_EDGES + 1, 0}},
        {true, 2, {JointSensor::Hall, -CAL_HALL_MAX_STROKE_EDGES - 1, 0}},
        {false, 1, {JointSensor::Hall, 0, 1069}},      // no quadrature halls
    };
    for (const Case &c : cases)
    {
        setUp();
        g_fakeQuadrature = c.quadrature;
        CalibrationRig rig(c.quadrature);
        StoredJoint record[6];
        std::copy(kGoodRecord, kGoodRecord + 6, record);
        if (!c.quadrature)
            for (StoredJoint &joint : record) joint = {JointSensor::Pot, 100, 900};
        record[c.joint] = c.stored;
        writeRecord(ActuatorManager::CAL_DATA_MAGIC, record);
        rig.manager.loadCalibration();
        const JointSensor fallback = c.quadrature ? JointSensor::Unknown : JointSensor::Pot;
        assertJoint(*rig.acts[c.joint], fallback, 0, c.quadrature ? 0 : 1023, false);
        const std::string line = std::string("Calibration joint ") + rig.names[c.joint] + " invalid: defaults\n";
        TEST_ASSERT_TRUE(contains(rig.output.output, line.c_str()));
        TEST_ASSERT_EQUAL_UINT(1, occurrences(rig.output.output, "invalid"));
        TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration v2 loaded\n"));
        const uint8_t other = c.joint == 4 ? 1 : 4;
        TEST_ASSERT_TRUE(rig.acts[other]->encFlags & ACTUATOR_FLAG_LIMITS_VALID);
    }
}

// The accepted hall hip strokes are 1049..1089 (1069 plus or minus 20); only
// exactly 1069 prints "extend derived", a hollow-stand sweep's measured switch
// within the tolerance does not. Outside the window the joint is invalid.
static void test_load_hall_hip_window_and_derived_line()
{
    struct Case
    {
        int hi;
        bool valid;
        bool derivedLine;
    };
    const Case cases[] = {
        {1069, true, true},
        {1049, true, false},
        {1089, true, false},
        {1048, false, false},
        {1090, false, false},
        {700, false, false},
        {0, false, false},
    };
    for (const Case &c : cases)
    {
        setUp();
        g_fakeQuadrature = true;
        CalibrationRig rig(true);
        StoredJoint record[6];
        std::copy(kGoodRecord, kGoodRecord + 6, record);
        record[1] = {JointSensor::Hall, 0, c.hi};
        record[4] = {JointSensor::Hall, 0, c.hi};  // the right hip too
        writeRecord(ActuatorManager::CAL_DATA_MAGIC, record);
        rig.manager.loadCalibration();
        const std::string out = "hi=" + std::to_string(c.hi) + "\n" + rig.output.output;
        for (int hip : {1, 4})
        {
            if (c.valid)
                assertJoint(*rig.acts[hip], JointSensor::Hall, 0, c.hi, true);
            else
                assertJoint(*rig.acts[hip], JointSensor::Unknown, 0, 0, false);
            const std::string invalid = std::string("Calibration joint ") + rig.names[hip] + " invalid: defaults\n";
            const std::string derived = std::string("Calibration joint ") + rig.names[hip] +
                " extend derived (1069 edges): not switch-verified\n";
            TEST_ASSERT_EQUAL_INT_MESSAGE(c.valid ? 0 : 1, contains(rig.output.output, invalid.c_str()) ? 1 : 0, out.c_str());
            TEST_ASSERT_EQUAL_INT_MESSAGE(c.derivedLine ? 1 : 0, contains(rig.output.output, derived.c_str()) ? 1 : 0, out.c_str());
        }
        // The knees never print it.
        TEST_ASSERT_EQUAL_UINT(c.derivedLine ? 2 : 0, occurrences(rig.output.output, "extend derived"));
    }
}

// A v2 record saved by a build without quadrature (every joint Pot, the yaws
// too; hips and knees at hall-B levels) is ignored on a quadrature build: its
// Pot kinds would remove the hall-B pull-ups and give a yaw a position. Without
// quadrature the same record loads.
static void test_load_ignores_a_record_from_a_build_without_halls()
{
    const StoredJoint rev2[6] = {
        {JointSensor::Pot, 0, 1023}, {JointSensor::Pot, 12, 1016}, {JointSensor::Pot, 12, 1016},
        {JointSensor::Pot, 0, 1023}, {JointSensor::Pot, 12, 1016}, {JointSensor::Pot, 12, 1016}};
    g_fakeQuadrature = true;
    {
        CalibrationRig rig(true);
        writeRecord(ActuatorManager::CAL_DATA_MAGIC, rev2);
        rig.manager.loadCalibration();
        for (auto *actuator : rig.acts)
            assertJoint(*actuator, JointSensor::Unknown, 0, 0, false);
        for (int joint = 0; joint < 6; ++joint)
            TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[5 * joint + 4]);
        TEST_ASSERT_EQUAL_STRING("Calibration from a build without halls ignored: run C\n",
            rig.output.output.c_str());
        TEST_ASSERT_FALSE(rig.acts[0]->setTarget(0.5f));  // the yaw has no position
    }
    // One Pot yaw is enough; a Pot hip alone (a real pot on a Rev 3 slot) is not.
    StoredJoint record[6];
    std::copy(kGoodRecord, kGoodRecord + 6, record);
    record[3] = {JointSensor::Pot, 100, 900};
    {
        CalibrationRig rig(true);
        writeRecord(ActuatorManager::CAL_DATA_MAGIC, record);
        rig.manager.loadCalibration();
        assertJoint(*rig.acts[1], JointSensor::Unknown, 0, 0, false);
        assertJoint(*rig.acts[4], JointSensor::Unknown, 0, 0, false);
        TEST_ASSERT_TRUE(contains(rig.output.output, "without halls ignored"));
    }
    g_fakeQuadrature = false;
    CalibrationRig pots;
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, rev2);
    pots.manager.loadCalibration();
    assertJoint(*pots.acts[0], JointSensor::Pot, 0, 1023, true);
    assertJoint(*pots.acts[1], JointSensor::Pot, 12, 1016, true);
    TEST_ASSERT_TRUE(contains(pots.output.output, "Calibration v2 loaded\n"));
}

// A v1 record may hold hall B rail values: ignored with quadrature halls,
// migrated as pots (each joint validated) without them.
static void test_load_v1_record_per_build()
{
    const StoredJoint v1[6] = {
        {JointSensor::Unknown, 5, 1018}, {JointSensor::Unknown, 0, 1023},
        {JointSensor::Unknown, 300, 200}, {JointSensor::Unknown, 100, 900},
        {JointSensor::Unknown, 0, 0}, {JointSensor::Unknown, 1016, 1016}};
    g_fakeQuadrature = true;
    {
        CalibrationRig rig(true);
        writeRecord(ActuatorManager::CAL_DATA_MAGIC_V1, v1);
        rig.manager.loadCalibration();
        for (auto *actuator : rig.acts)
            assertJoint(*actuator, JointSensor::Unknown, 0, 0, false);
        TEST_ASSERT_EQUAL_STRING("Calibration v1 ignored: run C\n", rig.output.output.c_str());
    }
    g_fakeQuadrature = false;
    CalibrationRig rig;
    writeRecord(ActuatorManager::CAL_DATA_MAGIC_V1, v1);
    rig.manager.loadCalibration();
    assertJoint(*rig.acts[0], JointSensor::Pot, 5, 1018, true);
    assertJoint(*rig.acts[1], JointSensor::Pot, 0, 1023, true);
    assertJoint(*rig.acts[2], JointSensor::Pot, 0, 1023, false);
    assertJoint(*rig.acts[3], JointSensor::Pot, 100, 900, true);
    assertJoint(*rig.acts[4], JointSensor::Pot, 0, 1023, false);
    TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration joint FLKL invalid: defaults\n"));
    TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration joint FRHL invalid: defaults\n"));
    TEST_ASSERT_TRUE(contains(rig.output.output, "Calibration v1 loaded\n"));
}

// No usable record: the joints keep what init() set; only the pin modes are
// reapplied.
static void test_load_without_record_keeps_defaults()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    rig.acts[3]->setSensor(JointSensor::Pot);
    modes[19] = OUTPUT;
    modes[4] = OUTPUT;
    rig.manager.loadCalibration();
    assertJoint(*rig.acts[0], JointSensor::Unknown, 0, 0, false);
    assertJoint(*rig.acts[3], JointSensor::Pot, 0, 1023, false);
    TEST_ASSERT_EQUAL_INT(INPUT_PULLUP, modes[4]);
    TEST_ASSERT_EQUAL_INT(INPUT, modes[19]);
    TEST_ASSERT_EQUAL_STRING("No calibration record: defaults\n", rig.output.output.c_str());

    // Without an output nothing is printed; a seventh actuator keeps its state.
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, kGoodRecord);
    LinearActuator extra("XTRA", 30, 31, 0, 1, 2, 0);
    LinearActuator *seven[7];
    for (int i = 0; i < 6; ++i) seven[i] = rig.acts[i];
    seven[6] = &extra;
    ActuatorManager wide(seven, 7);
    wide.setCalibrationOutput(nullptr, ROLE_FRONT);
    extra.init();
    extra.setSensor(JointSensor::Pot);
    rig.output.output.clear();
    wide.loadCalibration();
    assertJoint(extra, JointSensor::Pot, 0, 1023, false);
    assertJoint(*rig.acts[1], JointSensor::Hall, 0, 1069, true);
    StoredJoint bad[6];
    std::copy(kGoodRecord, kGoodRecord + 6, bad);
    bad[1] = {JointSensor::Hall, 1, 1069};
    writeRecord(ActuatorManager::CAL_DATA_MAGIC, bad);
    wide.loadCalibration();
    assertJoint(*rig.acts[1], JointSensor::Unknown, 0, 0, false);
    wide.saveCalibration();
    TEST_ASSERT_TRUE(Serial.output.empty());
    TEST_ASSERT_TRUE(rig.output.output.empty());
}

// A save cut short by a power loss after any number of changed bytes never
// leaves a record that loads with old and new values mixed: the magic is
// cleared first and written last. The old and new records here differ in four
// limits whose byte changes cancel in pairs (-60 +60 +5 -5), so a body written
// over a still-valid magic and check would load half old, half new.
static void test_save_cut_short_never_loads_a_mixed_record()
{
    const int oldMax[6] = {900, 900, 900, 900, 900, 900};
    const int newMax[6] = {840, 960, 905, 895, 900, 900};
    int changed = -1;
    for (int budget = 0; budget < 64 && changed < 0; ++budget)
    {
        setUp();
        CalibrationRig rig;
        for (int i = 0; i < 6; ++i)
        {
            rig.acts[i]->minStop = 100;
            rig.acts[i]->maxStop = oldMax[i];
        }
        rig.manager.saveCalibration();
        for (int i = 0; i < 6; ++i) rig.acts[i]->maxStop = newMax[i];
        EEPROM.writeBudget = budget;
        rig.manager.saveCalibration();
        const bool complete = EEPROM.droppedBytes == 0;
        EEPROM.writeBudget = -1;
        CalibrationRig next;  // the next boot
        next.manager.loadCalibration();
        bool allOld = true;
        bool allNew = true;
        bool defaults = true;
        for (int i = 0; i < 6; ++i)
        {
            allOld = allOld && next.acts[i]->maxStop == oldMax[i];
            allNew = allNew && next.acts[i]->maxStop == newMax[i];
            defaults = defaults && next.acts[i]->maxStop == 1023 && next.acts[i]->minStop == 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(allOld || allNew || defaults, std::to_string(budget).c_str());
        if (budget > 0 && budget < 8)
            TEST_ASSERT_TRUE_MESSAGE(defaults, std::to_string(budget).c_str());
        if (complete) changed = budget;
    }
    // Two magic bytes cleared, four limit bytes, two magic bytes restored.
    TEST_ASSERT_EQUAL_INT(8, changed);
}

// The save writes v2 for the first six actuators only.
static void test_save_writes_v2_with_kinds_and_check()
{
    g_fakeQuadrature = true;
    CalibrationRig rig(true);
    rig.acts[1]->setSensor(JointSensor::Hall);
    rig.acts[1]->maxStop = 900;
    rig.acts[3]->setSensor(JointSensor::Pot);
    rig.manager.saveCalibration();
    ActuatorManager::CalData saved{};
    EEPROM.get(EEPROM_ACTUATOR_CAL_ADDR, saved);
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::CAL_DATA_MAGIC, saved.magic);
    TEST_ASSERT_EQUAL_HEX16(0x0048, saved.sensorBits);  // joint 1 Hall, joint 3 Pot
    TEST_ASSERT_EQUAL_HEX16(ActuatorManager::calDataCheck(saved), saved.check);
    TEST_ASSERT_EQUAL_INT(900, saved.maxVals[1]);
    TEST_ASSERT_EQUAL_INT(1023, saved.maxVals[3]);

    auto lone = makeActuator();
    LinearActuator *one[] = {&lone};
    ActuatorManager small(one, 1);
    small.setCalibrationOutput(nullptr, ROLE_FRONT);
    small.initAll();
    lone.setSensor(JointSensor::Pot);
    lone.minStop = 10;
    lone.maxStop = 20;
    small.saveCalibration();
    EEPROM.get(EEPROM_ACTUATOR_CAL_ADDR, saved);
    TEST_ASSERT_EQUAL_HEX16(0x0001, saved.sensorBits);
    TEST_ASSERT_EQUAL_INT(20, saved.maxVals[0]);
    TEST_ASSERT_EQUAL_INT(0, saved.maxVals[1]);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_slot_config_ctor_sets_pins_name_and_control_config);
    RUN_TEST(test_init_resets_hardware_filters_and_evidence);
    RUN_TEST(test_init_on_quadrature_build_starts_unknown_with_pullup);
    RUN_TEST(test_probe_averages_in_order_and_restores_input);
    RUN_TEST(test_pot_snapshot_is_the_mean_of_four_direct_reads);
    RUN_TEST(test_sensor_update_probe_timing_and_filter_isolation);
    RUN_TEST(test_probe_and_tracker_run_only_for_pot_joints);
    RUN_TEST(test_sensor_rail_jitter_debounce_and_recovery);
    RUN_TEST(test_probe_fractional_boundary_at_different_positions);
    RUN_TEST(test_delayed_probe_runs_once_and_rail_check_remains_independent);
    RUN_TEST(test_high_baselines_fail_rail_check_despite_saturated_probe);
    RUN_TEST(test_telemetry_keeps_valid_position_when_current_is_absent);
    RUN_TEST(test_filtered_current_qualifies_and_preserves_fraction);
    RUN_TEST(test_position_target_and_manual_control);
    RUN_TEST(test_update_ramps_both_directions_and_respects_deadbands);
    RUN_TEST(test_telemetry_fields_and_hall_bounds);
    RUN_TEST(test_position_per_sensor_kind_and_homing_state);
    RUN_TEST(test_set_sensor_pin_modes_and_resets);
    RUN_TEST(test_set_target_refuses_nan_and_unreadable_and_insets_encoder_targets);
    RUN_TEST(test_update_stops_target_when_position_becomes_unreadable);
    RUN_TEST(test_encoder_control_sign_deadband_and_cap);
    RUN_TEST(test_stall_guard_stops_hall_and_pot_targets);
    RUN_TEST(test_stall_guard_is_not_reset_by_a_repeated_target);
    RUN_TEST(test_stall_guard_trip_stays_stopped_while_the_same_target_is_resent);
    RUN_TEST(test_quiet_stop_holds_only_its_own_target);
    RUN_TEST(test_range_guard_clears_a_wrong_home);
    RUN_TEST(test_range_guard_uses_a_hips_stroke_plus_margin);
    RUN_TEST(test_pot_joint_that_counts_edges_is_demoted);
    RUN_TEST(test_manager_dispatch_hold_and_telemetry);
    RUN_TEST(test_sensor_state_line_on_change_and_period);
    RUN_TEST(test_pending_errors_print_once);
    RUN_TEST(test_refused_targets_are_reported_and_rate_limited);
    RUN_TEST(test_refused_targets_beyond_thirty_two_actuators);
    RUN_TEST(test_calibration_sweeps_hips_and_knees_one_at_a_time_and_saves);
    RUN_TEST(test_hall_board_calibrates_homes_and_accepts_targets);
    RUN_TEST(test_guards_do_not_run_while_calibrating);
    RUN_TEST(test_sensors_refresh_every_loop_while_calibrating);
    RUN_TEST(test_hold_aborts_calibration_and_motors_stay_off);
    RUN_TEST(test_calibration_requires_six_actuators_and_an_output);
    RUN_TEST(test_calibration_plant_forwards_to_the_actuators);
    RUN_TEST(test_plant_home_lost_follows_the_guards);
    RUN_TEST(test_calibration_refuses_a_knee_move_while_its_hip_reads_out_with_a_lost_home);
    RUN_TEST(test_save_and_load_round_trip);
    RUN_TEST(test_load_v2_record_applies_kinds_limits_and_pin_modes);
    RUN_TEST(test_load_rejects_corrupt_v2_records);
    RUN_TEST(test_load_validates_each_joint);
    RUN_TEST(test_load_hall_hip_window_and_derived_line);
    RUN_TEST(test_load_ignores_a_record_from_a_build_without_halls);
    RUN_TEST(test_load_v1_record_per_build);
    RUN_TEST(test_load_without_record_keeps_defaults);
    RUN_TEST(test_save_cut_short_never_loads_a_mixed_record);
    RUN_TEST(test_save_writes_v2_with_kinds_and_check);
    return UNITY_END();
}

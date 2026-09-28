#include "unity.h"
#include "actuator_manager.h"
#include <algorithm>

namespace fakeArduino
{
uint32_t now;
int modes[32], digital[32], pwm[32], analog[32];
std::deque<int> readings[32];
std::vector<Event> events;
}
Print Serial;
FakeEEPROM EEPROM;
int32_t hallHwGetEdgeCount(uint8_t slot) { return 100 + slot; }

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
}
void tearDown() {}

static LinearActuator makeActuator(int8_t slot = -1)
{
    return LinearActuator("FLHY", 0, 1, 2, 3, 4, slot);
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
    actuator.printTelemetry(out, 0);
    TEST_ASSERT_EQUAL_STRING("FLHY 0.500 500 0 1 1 0 100 0 2 0", out.output.c_str());
    analog[4] = 0;
    for (int i = 0; i < 40; ++i) actuator.updateSensors();
    out.output.clear();
    actuator.printTelemetry(out, 0);
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
    actuator.setTarget(-1);
    TEST_ASSERT_EQUAL_INT(100, actuator.currentTarget);
    actuator.setTarget(2);
    TEST_ASSERT_EQUAL_INT(900, actuator.currentTarget);
    actuator.setTarget(0.25f);
    TEST_ASSERT_EQUAL_INT(300, actuator.currentTarget);
    actuator.manualDrive(999);
    assertDrive(255, 0, HIGH);
    actuator.manualDrive(-999);
    assertDrive(0, 255, HIGH);
    TEST_ASSERT_EQUAL_INT(-255, actuator.getStatus().commandedPwm);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorId::FLHY),
        static_cast<int>(actuator.getStatus().actuatorId));
    actuator.manualDrive(1);
    assertDrive(1, 0, HIGH);
    actuator.manualDrive(0);
    assertDrive(0, 0, LOW);
    TEST_ASSERT_TRUE(actuator.hasTarget);
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

static void test_stall_timeout_and_motion_reset()
{
    auto actuator = makeActuator();
    actuator.avgPot = 300;
    TEST_ASSERT_FALSE(actuator.isStalled(250));
    actuator.manualDrive(100);
    TEST_ASSERT_FALSE(actuator.isStalled(250));
    now = 250;
    TEST_ASSERT_FALSE(actuator.isStalled(250));
    now = 251;
    TEST_ASSERT_TRUE(actuator.isStalled(250));
    actuator.avgPot = 303;
    TEST_ASSERT_FALSE(actuator.isStalled(250));
}

static void test_telemetry_fields_and_hall_bounds()
{
    for (int slot : {-1, 0, 5, 6})
    {
        auto actuator = makeActuator(slot);
        actuator.init();
        actuator.minStop = 0;
        actuator.maxStop = 1000;
        actuator.manualDrive(-23);
        Print out;
        actuator.printTelemetry(out, 0);
        const std::string expected = "FLHY 0.500 500 500 1 1 23 0 " +
            std::to_string(slot >= 0 && slot < 6 ? 100 + slot : 0) + " 0 0";
        TEST_ASSERT_EQUAL_STRING(expected.c_str(), out.output.c_str());
        actuator.manualDrive(24);
        out.output.clear();
        actuator.printTelemetry(out, 0);
        TEST_ASSERT_NOT_NULL(std::strstr(out.output.c_str(), " 1 1 0 24 "));
    }
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

// Pot pin of makeActuator() ("FLHY", pot on pin 4); a constant reading makes the
// sweep stall immediately, which is how a real stop looks to sweepToStop().
static constexpr int FLHY_POT_PIN = 4;

static void test_directional_calibration_states_and_persistence()
{
    LinearActuator storage[] = {
        makeActuator(), makeActuator(), makeActuator(),
        makeActuator(), makeActuator(), makeActuator()};
    LinearActuator *acts[6];
    for (int i = 0; i < 6; ++i) acts[i] = &storage[i];
    ActuatorManager manager(acts, 6);
    manager.initAll();
    manager.loadCalibration();
    TEST_ASSERT_EQUAL_INT(1023, acts[0]->maxStop);

    Print reply;
    TEST_ASSERT_FALSE(manager.calibrateJoint("MISSING", "", reply));
    TEST_ASSERT_TRUE(reply.output.empty());

    // One stop recorded: persisted, but not applied (PARTIAL = calState 1).
    analog[FLHY_POT_PIN] = 100;
    TEST_ASSERT_TRUE(manager.calibrateJoint("FLHY", "retract", reply));
    TEST_ASSERT_EQUAL_STRING("CAL FLHY retract 100 saved\n", reply.output.c_str());
    TEST_ASSERT_EQUAL_INT(0, acts[0]->minStop);
    Print telemetry;
    manager.printTelemetry(telemetry);
    TEST_ASSERT_EQUAL_STRING(" 1", telemetry.output.substr(telemetry.output.find(';') - 2, 2).c_str());

    // Second stop completes the slot: applied live (FULL = calState 2).
    analog[FLHY_POT_PIN] = 900;
    acts[0]->avgPot = 900;  // settle the pot filter at the new stop
    reply.output.clear();
    TEST_ASSERT_TRUE(manager.calibrateJoint("FLHY", "extend", reply));
    TEST_ASSERT_EQUAL_STRING("CAL FLHY extend 900 saved\n", reply.output.c_str());
    TEST_ASSERT_EQUAL_INT(100, acts[0]->minStop);
    TEST_ASSERT_EQUAL_INT(900, acts[0]->maxStop);
    TEST_ASSERT_EQUAL_INT(0, acts[0]->currentPwm);
    telemetry.output.clear();
    manager.printTelemetry(telemetry);
    TEST_ASSERT_EQUAL_STRING(" 2", telemetry.output.substr(telemetry.output.find(';') - 2, 2).c_str());

    // Limits survive a reload from EEPROM; untouched slots keep full range.
    acts[0]->minStop = 0;
    acts[0]->maxStop = 1023;
    ActuatorManager reloaded(acts, 6);
    reloaded.loadCalibration();
    TEST_ASSERT_EQUAL_INT(100, acts[0]->minStop);
    TEST_ASSERT_EQUAL_INT(900, acts[0]->maxStop);
    TEST_ASSERT_EQUAL_INT(1023, acts[1]->maxStop);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_init_resets_hardware_filters_and_evidence);
    RUN_TEST(test_probe_averages_in_order_and_restores_input);
    RUN_TEST(test_sensor_update_probe_timing_and_filter_isolation);
    RUN_TEST(test_sensor_rail_jitter_debounce_and_recovery);
    RUN_TEST(test_probe_fractional_boundary_at_different_positions);
    RUN_TEST(test_delayed_probe_runs_once_and_rail_check_remains_independent);
    RUN_TEST(test_high_baselines_fail_rail_check_despite_saturated_probe);
    RUN_TEST(test_telemetry_keeps_valid_position_when_current_is_absent);
    RUN_TEST(test_filtered_current_qualifies_and_preserves_fraction);
    RUN_TEST(test_position_target_and_manual_control);
    RUN_TEST(test_update_ramps_both_directions_and_respects_deadbands);
    RUN_TEST(test_stall_timeout_and_motion_reset);
    RUN_TEST(test_telemetry_fields_and_hall_bounds);
    RUN_TEST(test_manager_dispatch_hold_and_telemetry);
    RUN_TEST(test_directional_calibration_states_and_persistence);
    return UNITY_END();
}

#include "unity.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "host_api.h"
#include "sim_serial.hpp"
#include "version.h"

using krabby::sim::SerialPortId;
using krabby::sim::serialPort;

void setUp() { krabby_host_reset(); }

void tearDown() {}

static void drainUsbTxLines()
{
    while (serialPort(SerialPortId::kUsb).txLineAvailable())
    {
        (void)serialPort(SerialPortId::kUsb).popTxLine();
    }
}

static void writeUsbWire(const char *wire)
{
    krabby_host_serial_usb_rx_write(
        reinterpret_cast<const uint8_t *>(wire),
        static_cast<int>(std::strlen(wire)));
}

static void runSetupDrainAndLoopOnce()
{
    krabby_host_firmware_setup();
    drainUsbTxLines();
    krabby_host_firmware_loop_once();
}

static void stageFrontBoardSensorsForGoldenTelemetry()
{
    static const char *kIds[] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    for (const char *id : kIds)
    {
        krabby_host_set_actuator_stops(id, 100, 900);
    }
    for (int potPin = 0; potPin < 6; ++potPin)
    {
        krabby_host_set_analog_read(potPin, 500);
        krabby_host_set_analog_read(potPin + 6, 12);
    }
}

static void test_telemetry_emits_on_fifty_ms_interval()
{
    krabby_host_firmware_setup();
    drainUsbTxLines();
    krabby_host_advance_millis(10);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_FALSE(serialPort(SerialPortId::kUsb).txLineAvailable());

    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    const std::string line = serialPort(SerialPortId::kUsb).popTxLine();
    TEST_ASSERT_TRUE(line.rfind("FRONT;", 0) == 0);
}

static void test_telemetry_golden_wire_format()
{
    static const char kGolden[] =
        "FRONT; FLHY 0.500 500 12 0 0 0 0 0 0"
        ";FLHL 0.500 500 12 0 0 0 0 0 0"
        ";FLKL 0.500 500 12 0 0 0 0 0 0"
        ";FRHY 0.500 500 12 0 0 0 0 0 0"
        ";FRHL 0.500 500 12 0 0 0 0 0 0"
        ";FRKL 0.500 500 12 0 0 0 0 0 0"
        ";IMU 0.000 0.000 9.806 0.0000 0.0000 0.0000 25.0 1"
        ";BATT 26.40 1.250 33.00 100000.0 13.20 13.20 0 0";

    stageFrontBoardSensorsForGoldenTelemetry();
    krabby_host_firmware_setup();
    drainUsbTxLines();
    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    const std::string line = serialPort(SerialPortId::kUsb).popTxLine();
    TEST_ASSERT_EQUAL_STRING(kGolden, line.c_str());
}

// Wire formats match firmware/krabby_mcu.py send_command_* emitters.
struct McuJogWireCase
{
    const char *wire;
    int en_pin;
    int expected_en;
};

static void test_mcu_wire_jog_commands_table()
{
    static const McuJogWireCase cases[] = {
        {"JFLHY 100\n", 22, 1},
        {"B FLHY -80 FRHL 50 \n", 22, 1},
    };

    for (const McuJogWireCase &entry : cases)
    {
        krabby_host_reset();
        writeUsbWire(entry.wire);
        runSetupDrainAndLoopOnce();
        TEST_ASSERT_EQUAL_INT(entry.expected_en, krabby_host_pin_digital(entry.en_pin));
    }
}

static void test_mcu_wire_joints_command()
{
    krabby_host_set_actuator_stops("FLHY", 100, 900);
    krabby_host_set_analog_read(0, 100);
    writeUsbWire("T FLHY 1.000\n");
    runSetupDrainAndLoopOnce();
    for (int i = 0; i < 8; ++i)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
    }
    TEST_ASSERT_EQUAL_INT(1, krabby_host_pin_digital(22));
}

static void test_mcu_wire_calibrate_command()
{
    writeUsbWire("C\n");
    runSetupDrainAndLoopOnce();
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    const std::string line = serialPort(SerialPortId::kUsb).popTxLine();
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT START pwm=128 passes=SEEK,SWEEP", line.c_str());
    TEST_ASSERT_EQUAL_INT(1, krabby_host_firmware_calibration_active());

    writeUsbWire("H\n");
    krabby_host_firmware_loop_once();
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());
}

// Host FRONT board pins (host_api.cpp): PWM_R, PWM_L, EN, POT per actuator slot.
struct HostJointPins
{
    int pwmR;
    int pwmL;
    int en;
    int pot;
};
static const HostJointPins kFrontJointPins[6] = {
    {2, 3, 22, 0}, {4, 5, 24, 1}, {6, 7, 26, 2},
    {8, 9, 23, 3}, {10, 11, 25, 4}, {12, 13, 27, 5},
};

static void test_mcu_wire_calibrate_completes_on_sim_limits()
{
    constexpr int kLow = 150;
    constexpr int kHigh = 850;
    int pot[6] = {500, 500, 500, 500, 500, 500};
    for (int joint = 0; joint < 6; ++joint)
    {
        krabby_host_set_analog_read(kFrontJointPins[joint].pot, pot[joint]);
    }
    krabby_host_firmware_setup();
    krabby_host_set_actuator_stops("FLHY", 123, 877);
    krabby_host_set_actuator_stops("FRHY", 321, 678);
    drainUsbTxLines();
    writeUsbWire("C 120\n");

    std::string calLog;
    bool isDone = false;
    int maxDriven = 0;
    int maxMagnitude = 0;
    bool isYawDriven = false;
    for (int tick = 0; tick < 20000 && !isDone; ++tick)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
        int driven = 0;
        for (int joint = 0; joint < 6; ++joint)
        {
            const HostJointPins &pins = kFrontJointPins[joint];
            const int drive = krabby_host_pin_digital(pins.en)
                ? krabby_host_pin_pwm(pins.pwmR) - krabby_host_pin_pwm(pins.pwmL)
                : 0;
            if (drive != 0)
            {
                ++driven;
                maxMagnitude = std::max(maxMagnitude, drive < 0 ? -drive : drive);
                if (joint == 0 || joint == 3) isYawDriven = true;
            }
            if (drive > 0) pot[joint] = std::min(kHigh, pot[joint] + 3);
            if (drive < 0) pot[joint] = std::max(kLow, pot[joint] - 3);
            krabby_host_set_analog_read(pins.pot, pot[joint]);
        }
        maxDriven = std::max(maxDriven, driven);
        while (serialPort(SerialPortId::kUsb).txLineAvailable())
        {
            const std::string line = serialPort(SerialPortId::kUsb).popTxLine();
            if (line.rfind("CAL ", 0) != 0) continue;
            calLog += line;
            calLog += '\n';
            if (line == "CAL ROBOT DONE") isDone = true;
        }
    }

    TEST_ASSERT_TRUE_MESSAGE(isDone, calLog.c_str());
    TEST_ASSERT_EQUAL_INT(1, maxDriven);
    TEST_ASSERT_EQUAL_INT(120, maxMagnitude);
    TEST_ASSERT_FALSE(isYawDriven);
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());
    TEST_ASSERT_NOT_NULL(std::strstr(calLog.c_str(), "CAL ROBOT SKIP LEFT no link\n"));
    TEST_ASSERT_NOT_NULL(std::strstr(calLog.c_str(), "CAL FRONT PASS_DONE SWEEP saved\n"));

    static const char *kIds[] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    for (const char *id : kIds)
    {
        krabby_host_set_actuator_stops(id, 1, 2);
    }
    krabby_host_load_calibration();
    for (int joint = 0; joint < 6; ++joint)
    {
        const int minStop = krabby_host_actuator_min_stop(kIds[joint]);
        const int maxStop = krabby_host_actuator_max_stop(kIds[joint]);
        if (joint == 0 || joint == 3)
        {
            // Yaws are never driven; their limits are saved exactly as they were.
            TEST_ASSERT_EQUAL_INT(joint == 0 ? 123 : 321, minStop);
            TEST_ASSERT_EQUAL_INT(joint == 0 ? 877 : 678, maxStop);
            continue;
        }
        TEST_ASSERT_INT_WITHIN(1, kLow, minStop);
        TEST_ASSERT_INT_WITHIN(1, kHigh, maxStop);
    }
}

static void test_mcu_wire_hold_command()
{
    writeUsbWire("JFLHY 100\n");
    runSetupDrainAndLoopOnce();
    TEST_ASSERT_EQUAL_INT(1, krabby_host_pin_digital(22));

    writeUsbWire("H\n");
    krabby_host_firmware_loop_once();
    TEST_ASSERT_EQUAL_INT(0, krabby_host_pin_digital(22));
}

static std::vector<std::string> popUsbTxLines()
{
    std::vector<std::string> lines;
    while (serialPort(SerialPortId::kUsb).txLineAvailable())
        lines.push_back(serialPort(SerialPortId::kUsb).popTxLine());
    return lines;
}

static bool hasTxLine(const std::vector<std::string> &lines, const char *expected)
{
    return std::find(lines.begin(), lines.end(), std::string(expected)) != lines.end();
}

static int countTxLines(const std::vector<std::string> &lines, const char *expected)
{
    return static_cast<int>(std::count(lines.begin(), lines.end(), std::string(expected)));
}

// One joint's telemetry segment ("<name> <pos> ... <conn>") from a FRONT line.
static std::string telemetrySegment(const std::string &line, const char *name)
{
    const std::string key = std::string(name) + " ";
    const std::size_t start = line.find(key);
    if (start == std::string::npos) return std::string();
    const std::size_t end = line.find(';', start);
    return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

// A Rev 3 board (quadrature halls) without a record: every joint is Unknown, so
// it has no position. T is refused with one ERR line per second and nothing
// moves; telemetry prints nan and disconnected; a SENS line names each state.
static void test_quadrature_board_without_calibration_refuses_targets()
{
    krabby_host_board_set_hall_quadrature(0, 1);
    krabby_host_firmware_setup();
    krabby_host_set_actuator_stops("FLHL", 100, 900);  // limits alone give no position
    std::vector<std::string> lines = popUsbTxLines();
    TEST_ASSERT_TRUE(hasTxLine(lines, "No calibration record: defaults"));

    krabby_host_advance_millis(10);
    krabby_host_firmware_loop_once();
    lines = popUsbTxLines();
    TEST_ASSERT_TRUE(hasTxLine(lines, "SENS FRONT FLHY:? FLHL:? FLKL:? FRHY:? FRHL:? FRKL:?"));

    writeUsbWire("T FLHL 0.500 FRKL 0.250\n");
    krabby_host_advance_millis(10);
    krabby_host_firmware_loop_once();
    lines = popUsbTxLines();
    TEST_ASSERT_TRUE(hasTxLine(lines, "ERR T refused FLHL FRKL (no position)"));
    std::string telemetry;
    for (int i = 0; i < 20; ++i)
    {
        writeUsbWire("T FLHL 0.500\n");
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
        TEST_ASSERT_EQUAL_INT(0, krabby_host_pin_digital(24));
        TEST_ASSERT_EQUAL_INT(0, krabby_host_pin_digital(27));
        for (const std::string &line : popUsbTxLines())
        {
            lines.push_back(line);
            if (line.rfind("FRONT;", 0) == 0) telemetry = line;
        }
    }
    // At most one refusal line per second.
    TEST_ASSERT_EQUAL_INT(1, countTxLines(lines, "ERR T refused FLHL FRKL (no position)"));
    TEST_ASSERT_EQUAL_INT(0, countTxLines(lines, "ERR T refused FLHL (no position)"));
    TEST_ASSERT_EQUAL_STRING("FLHL nan 512 512 0 0 0 0 0 2", telemetrySegment(telemetry, "FLHL").c_str());

    krabby_host_advance_millis(1000);
    writeUsbWire("T FLHL 0.500\n");
    krabby_host_firmware_loop_once();
    lines = popUsbTxLines();
    TEST_ASSERT_TRUE(hasTxLine(lines, "ERR T refused FLHL (no position)"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_pin_digital(24));
}

static void test_mcu_wire_version_command()
{
    writeUsbWire("V\n");
    runSetupDrainAndLoopOnce();
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    const std::string line = serialPort(SerialPortId::kUsb).popTxLine();

    char expected[160];
    std::snprintf(
        expected,
        sizeof(expected),
        "VER %s|-|- %s|-|- %s|-|-",
        KRABBY_FW_VERSION,
        KRABBY_FW_BRANCH,
        KRABBY_FW_COMMIT);
    TEST_ASSERT_EQUAL_STRING(expected, line.c_str());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_telemetry_emits_on_fifty_ms_interval);
    RUN_TEST(test_telemetry_golden_wire_format);
    RUN_TEST(test_mcu_wire_jog_commands_table);
    RUN_TEST(test_mcu_wire_joints_command);
    RUN_TEST(test_mcu_wire_calibrate_command);
    RUN_TEST(test_mcu_wire_calibrate_completes_on_sim_limits);
    RUN_TEST(test_mcu_wire_hold_command);
    RUN_TEST(test_quadrature_board_without_calibration_refuses_targets);
    RUN_TEST(test_mcu_wire_version_command);
    return UNITY_END();
}

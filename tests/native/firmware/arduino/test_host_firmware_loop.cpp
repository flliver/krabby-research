#include "unity.h"

#include <cstring>

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
        ";FRKL 0.500 500 12 0 0 0 0 0 0";

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
    TEST_ASSERT_NOT_NULL(std::strstr(line.c_str(), "Starting Auto-Calibration"));
}

static void settlePotBeforeCal(int potPin, int adc)
{
    krabby_host_set_analog_read(potPin, adc);
    for (int i = 0; i < 40; ++i)
    {
        krabby_host_advance_millis(10);
        krabby_host_update_all();
    }
}

static void runHostCalStallStep(int potPin, int adc)
{
    krabby_host_set_analog_read(potPin, adc);
    for (int i = 0; i < 40; ++i)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
    }
    krabby_host_advance_millis(350);
    krabby_host_firmware_loop_once();
}

static void test_mcu_wire_calibrate_completes_on_sim_limits()
{
    static const struct
    {
        int potPin;
        int adc;
    } kStallSteps[] = {
        {0, 900}, {3, 100}, {3, 900}, {1, 100}, {2, 900},
        {2, 100}, {1, 900}, {4, 100}, {5, 900}, {5, 100}, {4, 900},
    };

    for (int potPin = 0; potPin < 6; ++potPin)
    {
        krabby_host_set_analog_read(potPin, 500);
    }
    writeUsbWire("C\n");
    krabby_host_firmware_setup();
    drainUsbTxLines();
    settlePotBeforeCal(0, 100);
    krabby_host_firmware_loop_once();
    krabby_host_advance_millis(300);
    krabby_host_firmware_loop_once();

    for (const auto &step : kStallSteps)
    {
        runHostCalStallStep(step.potPin, step.adc);
    }

    krabby_host_advance_millis(1);
    krabby_host_firmware_loop_once();

    std::string txLog;
    while (serialPort(SerialPortId::kUsb).txLineAvailable())
    {
        txLog.append(serialPort(SerialPortId::kUsb).popTxLine());
        txLog.push_back('\n');
    }
    TEST_ASSERT_NOT_NULL(std::strstr(txLog.c_str(), "CALIBRATION COMPLETE"));

    krabby_host_load_calibration();
    static const char *kIds[] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    int jointsWithTravel = 0;
    for (const char *id : kIds)
    {
        const int minStop = krabby_host_actuator_min_stop(id);
        const int maxStop = krabby_host_actuator_max_stop(id);
        if (maxStop - minStop >= 50)
        {
            jointsWithTravel++;
        }
    }
    TEST_ASSERT_GREATER_OR_EQUAL(4, jointsWithTravel);
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
    RUN_TEST(test_mcu_wire_version_command);
    return UNITY_END();
}

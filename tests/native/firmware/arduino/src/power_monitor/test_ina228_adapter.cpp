#include "unity.h"
#include <math.h>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "src/power_monitor/ina228_adapter.h"
#include "ina228.h"

using ina228_native::Device;
using ina228_native::Register;

static uint32_t fakeNow;
static std::vector<unsigned long> delays;
unsigned long millis() { return fakeNow; }
void delay(unsigned long milliseconds) { delays.push_back(milliseconds); }

void setUp() { fakeNow = 0; delays.clear(); }
void tearDown() {}

static constexpr uint8_t PACK = 0x40, MIDPOINT = 0x41;
static constexpr float PACK_OHMS = 0.000375f, PACK_MAX_AMPS = 200.0f;
// SHUNT_CAL = 13107.2e6 * (200 A / 2^19) * 0.000375 ohm, exactly.
static constexpr uint16_t PACK_SHUNT_CAL = 1875;
static constexpr uint16_t RESET_SHUNT_CAL = 0x1000;
static constexpr uint16_t WRONG_DEVICE_ID = 0x2380;

struct Sample
{
    float voltage, current, power, charge;
};
// Encodes exactly in the INA228 registers at the pack calibration.
static constexpr Sample PACK_SAMPLE = {26.25f, -3.125f, 82.03125f, -1000.0f};

static void applyPackSample(Device &device)
{
    device.busVolts = 26.25;
    device.amps = -3.125;
    device.coulombs = -1000.0;
}

// Without calibration the driver's current LSB is zero.
static Sample uncalibrated(float volts) { return Sample{volts, 0.0f, 0.0f, 0.0f}; }

static void attach(TwoWire &wire, uint8_t address, Device &device) { wire.state().devices[address] = &device; }

static void assertSteps(const Device &device, std::initializer_list<const char *> expected)
{
    const auto steps = device.steps();
    TEST_ASSERT_EQUAL_UINT(expected.size(), steps.size());
    size_t i = 0;
    for (const char *step : expected)
        TEST_ASSERT_EQUAL_STRING(step, steps[i++].c_str());
}

static void assertUnavailable(const PowerMonitorMeasurement &reading)
{
    TEST_ASSERT_FALSE(reading.isValid);
    TEST_ASSERT_TRUE(isnan(reading.voltage.value()));
    TEST_ASSERT_TRUE(isnan(reading.current.value()));
    TEST_ASSERT_TRUE(isnan(reading.power.value()));
    TEST_ASSERT_TRUE(isnan(reading.charge.value()));
}

static void assertValues(const PowerMonitorMeasurement &reading, const Sample &expected)
{
    TEST_ASSERT_TRUE(reading.isValid);
    TEST_ASSERT_EQUAL_FLOAT(expected.voltage, reading.voltage.value());
    TEST_ASSERT_EQUAL_FLOAT(expected.current, reading.current.value());
    TEST_ASSERT_EQUAL_FLOAT(expected.power, reading.power.value());
    TEST_ASSERT_EQUAL_FLOAT(expected.charge, reading.charge.value());
}

using Nack = std::function<bool(uint8_t, bool, uint16_t)>;

// Each configuration step is one register write; NACK that write.
static Nack failStep(const std::string &step)
{
    return [step](uint8_t reg, bool isWrite, uint16_t value) -> bool {
        if (!isWrite) return false;
        if (step == "reset") return reg == Register::CONFIG && (value & ina228_native::CONFIG_RST);
        if (step == "accumulators") return reg == Register::CONFIG && (value & ina228_native::CONFIG_RSTACC);
        if (step == "alert") return reg == Register::DIAG_ALRT;
        if (step == "mode") return reg == Register::ADC_CONFIG;
        if (step == "calibrate") return reg == Register::SHUNT_CAL;
        return false;
    };
}

static Nack failReads(std::vector<uint8_t> registers)
{
    return [registers](uint8_t reg, bool isWrite, uint16_t) -> bool {
        return !isWrite && std::find(registers.begin(), registers.end(), reg) != registers.end();
    };
}

static const char *const CONFIGURATION_STEPS[] = {"begin", "reset", "alert", "mode", "calibrate", "accumulators"};

static void test_unbound_adapter_does_not_attempt_recovery()
{
    Ina228Adapter sensor(MIDPOINT);
    for (int i = 0; i < 10; ++i) assertUnavailable(sensor.measure());
    TEST_ASSERT_FALSE(sensor.begin(nullptr));
    assertUnavailable(sensor.measure());
    TEST_ASSERT_FALSE(sensor.isUp());
    TEST_ASSERT_EQUAL_UINT8(0, sensor.badTicks());
    TEST_ASSERT_TRUE(delays.empty());
}

static void test_initial_configuration_and_current_snapshot()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    Ina228Adapter pack(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_TRUE(pack.begin(&wire));
    TEST_ASSERT_TRUE(pack.isUp());
    TEST_ASSERT_EQUAL_HEX8(PACK, pack.address());
    assertSteps(device, {"begin", "reset", "alert", "mode", "calibrate", "accumulators"});
    TEST_ASSERT_EQUAL_UINT(1, device.resetCount);
    TEST_ASSERT_EQUAL_UINT(1, device.accumulatorResetCount);
    TEST_ASSERT_TRUE(device.registerValue(Register::DIAG_ALRT) & ina228_native::DIAG_CNVR);
    TEST_ASSERT_EQUAL_HEX16(0xF, device.registerValue(Register::ADC_CONFIG) >> 12);
    TEST_ASSERT_EQUAL_UINT16(PACK_SHUNT_CAL, device.registerValue(Register::SHUNT_CAL));
    TEST_ASSERT_EQUAL_UINT(1, delays.size());
    TEST_ASSERT_EQUAL_UINT(2, delays[0]);
    device.clearLog();
    applyPackSample(device);
    const auto first = pack.measure();
    assertValues(first, PACK_SAMPLE);
    assertSteps(device, {"voltage", "current", "power", "charge"});
    device.busVolts = 12.375;
    TEST_ASSERT_EQUAL_FLOAT(12.375f, pack.measure().voltage.value());
    TEST_ASSERT_EQUAL_FLOAT(26.25f, first.voltage.value());
    TEST_ASSERT_TRUE(wire.state().errors.empty());
}

static void test_shunt_and_accumulator_configuration_are_explicit()
{
    TwoWire wire;
    Device midpointDevice, zeroDevice;
    attach(wire, MIDPOINT, midpointDevice);
    attach(wire, 0x42, zeroDevice);
    Ina228Adapter midpoint(MIDPOINT);
    TEST_ASSERT_TRUE(midpoint.begin(&wire));
    assertSteps(midpointDevice, {"begin", "reset", "alert", "mode"});
    TEST_ASSERT_EQUAL_UINT16(RESET_SHUNT_CAL, midpointDevice.registerValue(Register::SHUNT_CAL));
    TEST_ASSERT_EQUAL_UINT(0, midpointDevice.accumulatorResetCount);
    TEST_ASSERT_EQUAL_HEX8(MIDPOINT, midpoint.address());
    // The driver rejects a non-positive shunt before any bus traffic, so an explicit
    // zero configuration fails setup rather than running uncalibrated.
    Ina228Adapter explicitZero(0x42, 0.0f, 0.0f);
    TEST_ASSERT_FALSE(explicitZero.begin(&wire));
    TEST_ASSERT_FALSE(explicitZero.isUp());
    assertSteps(zeroDevice, {"begin", "reset", "alert", "mode"});
    TEST_ASSERT_EQUAL_UINT16(RESET_SHUNT_CAL, zeroDevice.registerValue(Register::SHUNT_CAL));
}

static void test_each_configuration_failure_stops_setup()
{
    for (int failed = 0; failed < 6; ++failed)
    {
        delays.clear();
        TwoWire wire;
        Device device(PACK_OHMS);
        attach(wire, PACK, device);
        if (failed == 0) device.deviceId = WRONG_DEVICE_ID;
        else device.nack = failStep(CONFIGURATION_STEPS[failed]);
        Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
        TEST_ASSERT_FALSE(sensor.begin(&wire));
        TEST_ASSERT_FALSE(sensor.isUp());
        const auto steps = device.steps();
        TEST_ASSERT_EQUAL_UINT(failed + 1, steps.size());
        for (int i = 0; i <= failed; ++i)
            TEST_ASSERT_EQUAL_STRING(CONFIGURATION_STEPS[i], steps[i].c_str());
        TEST_ASSERT_EQUAL_UINT(failed >= 4 ? 1 : 0, delays.size());
        assertUnavailable(sensor.measure());
        TEST_ASSERT_TRUE(steps == device.steps());
        device.deviceId = ina228_native::INA228_DEVICE_ID;
        device.nack = nullptr;
        device.clearLog();
        TEST_ASSERT_TRUE(sensor.begin(&wire));
        TEST_ASSERT_TRUE(sensor.isUp());
        assertSteps(device, {"begin", "reset", "alert", "mode", "calibrate", "accumulators"});
    }
}

static void test_all_read_failure_combinations_preserve_successful_fields()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_TRUE(sensor.begin(&wire));
    applyPackSample(device);
    const uint8_t registers[] = {Register::VBUS, Register::CURRENT, Register::POWER, Register::CHARGE};
    const float expected[] = {PACK_SAMPLE.voltage, PACK_SAMPLE.current, PACK_SAMPLE.power, PACK_SAMPLE.charge};
    for (int mask = 0; mask < 16; ++mask)
    {
        device.nack = nullptr;
        assertValues(sensor.measure(), PACK_SAMPLE);
        std::vector<uint8_t> failing;
        for (int field = 0; field < 4; ++field)
            if (mask & (1 << field)) failing.push_back(registers[field]);
        device.nack = failReads(failing);
        device.clearLog();
        const auto reading = sensor.measure();
        TEST_ASSERT_EQUAL_INT(mask == 0, reading.isValid);
        const float values[] = {reading.voltage.value(), reading.current.value(),
            reading.power.value(), reading.charge.value()};
        for (int field = 0; field < 4; ++field)
            if (mask & (1 << field)) TEST_ASSERT_TRUE(isnan(values[field]));
            else TEST_ASSERT_EQUAL_FLOAT(expected[field], values[field]);
        assertSteps(device, {"voltage", "current", "power", "charge"});
        TEST_ASSERT_TRUE(sensor.isUp());
        TEST_ASSERT_EQUAL_UINT8(mask == 0 ? 0 : 1, sensor.badTicks());
    }
    device.nack = nullptr;
    assertValues(sensor.measure(), PACK_SAMPLE);
}

static void test_readings_round_to_register_resolution()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_TRUE(sensor.begin(&wire));
    device.busVolts = 26.31;
    device.amps = -2.5;
    device.coulombs = -2.5;
    const float currentLsb = PACK_MAX_AMPS / 524288.0f;
    const auto reading = sensor.measure();
    TEST_ASSERT_TRUE(reading.isValid);
    TEST_ASSERT_FLOAT_WITHIN(195.3125e-6f, 26.31f, reading.voltage.value());
    TEST_ASSERT_FLOAT_WITHIN(currentLsb, -2.5f, reading.current.value());
    TEST_ASSERT_FLOAT_WITHIN(currentLsb, -2.5f, reading.charge.value());
    TEST_ASSERT_TRUE(reading.current.value() != -2.5f);
    // POWER is an unsigned magnitude, one current LSB coarse at the bus voltage.
    TEST_ASSERT_FLOAT_WITHIN(26.31f * currentLsb + 3.2f * currentLsb, 65.775f, reading.power.value());
}

static void test_raw_accessors_preserve_units_and_handle_error_status()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_TRUE(sensor.begin(&wire));
    applyPackSample(device);
    device.clearLog();
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.voltage, sensor.readBusVoltage().value());
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.current, sensor.readCurrent().value());
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.power, sensor.readPower().value());
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.charge, sensor.readCharge().value());
    assertSteps(device, {"voltage", "current", "power", "charge"});
    device.nack = failReads({Register::VBUS, Register::CURRENT, Register::POWER, Register::CHARGE});
    TEST_ASSERT_TRUE(isnan(sensor.readBusVoltage().value()));
    TEST_ASSERT_TRUE(isnan(sensor.readCurrent().value()));
    TEST_ASSERT_TRUE(isnan(sensor.readPower().value()));
    TEST_ASSERT_TRUE(isnan(sensor.readCharge().value()));
}

static void test_pack_recovery_preserves_charge_and_reads_in_same_call()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    device.deviceId = WRONG_DEVICE_ID;
    Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_FALSE(sensor.begin(&wire));
    device.clearLog();
    const auto unavailable = sensor.measure();
    assertUnavailable(unavailable);
    assertUnavailable(sensor.measure());
    TEST_ASSERT_TRUE(device.steps().empty());
    device.deviceId = ina228_native::INA228_DEVICE_ID;
    applyPackSample(device);
    assertValues(sensor.measure(), PACK_SAMPLE);
    assertSteps(device, {"begin", "alert", "mode", "calibrate", "voltage", "current", "power", "charge"});
    TEST_ASSERT_EQUAL_UINT(0, device.resetCount);
    TEST_ASSERT_EQUAL_UINT(0, device.accumulatorResetCount);
    TEST_ASSERT_EQUAL_UINT(1, delays.size());
    TEST_ASSERT_EQUAL_UINT(2, delays[0]);
    TEST_ASSERT_EQUAL_UINT8(0, sensor.badTicks());
    assertUnavailable(unavailable);
}

static void test_recovery_reconfigures_when_charge_preservation_is_disabled()
{
    for (bool hasShunt : {false, true})
    {
        TwoWire wire;
        Device device(hasShunt ? PACK_OHMS : 0.0);
        attach(wire, MIDPOINT, device);
        device.deviceId = WRONG_DEVICE_ID;
        Ina228Adapter withShunt(MIDPOINT, PACK_OHMS, PACK_MAX_AMPS);
        Ina228Adapter withoutShunt(MIDPOINT);
        Ina228Adapter &sensor = hasShunt ? withShunt : withoutShunt;
        TEST_ASSERT_FALSE(sensor.begin(&wire));
        assertUnavailable(sensor.measure());
        assertUnavailable(sensor.measure());
        device.deviceId = ina228_native::INA228_DEVICE_ID;
        device.clearLog();
        applyPackSample(device);
        const auto reading = sensor.measure();
        TEST_ASSERT_TRUE(reading.isValid);
        TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.voltage, reading.voltage.value());
        // Without charge preservation, recovery resets the chip and its accumulator.
        TEST_ASSERT_EQUAL_FLOAT(0.0f, reading.charge.value());
        TEST_ASSERT_EQUAL_UINT(1, device.resetCount);
        if (hasShunt)
        {
            TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.current, reading.current.value());
            assertSteps(device, {"begin", "reset", "alert", "mode", "calibrate", "voltage", "current", "power", "charge"});
        }
        else
        {
            TEST_ASSERT_EQUAL_FLOAT(0.0f, reading.current.value());
            assertSteps(device, {"begin", "reset", "alert", "mode", "voltage", "current", "power", "charge"});
        }
    }
}

static void test_absence_and_failed_recovery_respect_exact_retry_boundary()
{
    for (uint32_t start : {uint32_t(0), UINT32_MAX - 999u})
    {
        TwoWire wire;
        Device device;
        attach(wire, MIDPOINT, device);
        device.deviceId = WRONG_DEVICE_ID;
        Ina228Adapter sensor(MIDPOINT);
        TEST_ASSERT_FALSE(sensor.begin(&wire));
        device.clearLog();
        wire.state().events.clear();
        fakeNow = start;
        assertUnavailable(sensor.measure());
        assertUnavailable(sensor.measure());
        TEST_ASSERT_TRUE(wire.state().events.empty());
        assertUnavailable(sensor.measure());
        assertSteps(device, {"begin"});
        fakeNow = start + 1999u;
        for (int i = 0; i < 3; ++i) assertUnavailable(sensor.measure());
        assertSteps(device, {"begin"});
        fakeNow = start + 2000u;
        assertUnavailable(sensor.measure());
        assertSteps(device, {"begin", "begin"});
        TEST_ASSERT_FALSE(sensor.isUp());
        device.deviceId = ina228_native::INA228_DEVICE_ID;
        device.busVolts = 13.125;
        fakeNow = start + 3999u;
        for (int i = 0; i < 3; ++i) assertUnavailable(sensor.measure());
        device.clearLog();
        fakeNow = start + 4000u;
        assertValues(sensor.measure(), uncalibrated(13.125f));
        assertSteps(device, {"begin", "reset", "alert", "mode", "voltage", "current", "power", "charge"});
        TEST_ASSERT_TRUE(wire.state().errors.empty());
    }
}

static void test_failed_recovery_configuration_returns_no_measurement()
{
    const char *const failing[] = {"alert", "mode", "calibrate"};
    for (int failure = 0; failure < 3; ++failure)
    {
        fakeNow = 0;
        TwoWire wire;
        Device device(PACK_OHMS);
        attach(wire, PACK, device);
        device.deviceId = WRONG_DEVICE_ID;
        Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
        TEST_ASSERT_FALSE(sensor.begin(&wire));
        assertUnavailable(sensor.measure());
        assertUnavailable(sensor.measure());
        device.deviceId = ina228_native::INA228_DEVICE_ID;
        device.nack = failStep(failing[failure]);
        device.clearLog();
        assertUnavailable(sensor.measure());
        TEST_ASSERT_FALSE(sensor.isUp());
        const char *const expected[] = {"begin", "alert", "mode", "calibrate"};
        const auto steps = device.steps();
        TEST_ASSERT_EQUAL_UINT(failure + 2, steps.size());
        for (int i = 0; i < failure + 2; ++i)
            TEST_ASSERT_EQUAL_STRING(expected[i], steps[i].c_str());
        device.nack = nullptr;
        fakeNow = 2000;
        assertUnavailable(sensor.measure());
        assertUnavailable(sensor.measure());
        device.clearLog();
        applyPackSample(device);
        assertValues(sensor.measure(), PACK_SAMPLE);
        assertSteps(device, {"begin", "alert", "mode", "calibrate", "voltage", "current", "power", "charge"});
    }
}

static void test_monitor_failures_and_recovery_are_independent()
{
    for (uint8_t failedAddress : {PACK, MIDPOINT})
    {
        fakeNow = 0;
        TwoWire wire;
        Device packDevice(PACK_OHMS), midDevice;
        attach(wire, PACK, packDevice);
        attach(wire, MIDPOINT, midDevice);
        Device &failed = failedAddress == PACK ? packDevice : midDevice;
        Device &healthy = failedAddress == PACK ? midDevice : packDevice;
        const Sample failedSample = failedAddress == PACK ? PACK_SAMPLE : uncalibrated(13.0f);
        const Sample healthySample = failedAddress == PACK ? uncalibrated(13.0f) : PACK_SAMPLE;
        failed.deviceId = WRONG_DEVICE_ID;
        Ina228Adapter pack(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
        Ina228Adapter midpoint(MIDPOINT);
        TEST_ASSERT_EQUAL_INT(failedAddress != PACK, pack.begin(&wire));
        TEST_ASSERT_EQUAL_INT(failedAddress != MIDPOINT, midpoint.begin(&wire));
        applyPackSample(packDevice);
        midDevice.busVolts = 13.0;
        Ina228Adapter &unavailable = failedAddress == PACK ? pack : midpoint;
        Ina228Adapter &available = failedAddress == PACK ? midpoint : pack;
        healthy.clearLog();
        for (int i = 0; i < 2; ++i)
        {
            assertUnavailable(unavailable.measure());
            assertValues(available.measure(), healthySample);
            TEST_ASSERT_EQUAL_UINT8(0, available.badTicks());
        }
        failed.deviceId = ina228_native::INA228_DEVICE_ID;
        assertValues(unavailable.measure(), failedSample);
        assertValues(available.measure(), healthySample);
        assertSteps(healthy, {"voltage", "current", "power", "charge", "voltage", "current", "power", "charge", "voltage", "current", "power", "charge"});
        TEST_ASSERT_TRUE(wire.state().errors.empty());
        TEST_ASSERT_EQUAL_UINT16(PACK_SHUNT_CAL, packDevice.registerValue(Register::SHUNT_CAL));
        TEST_ASSERT_EQUAL_UINT16(RESET_SHUNT_CAL, midDevice.registerValue(Register::SHUNT_CAL));
        failed.nack = failReads({Register::CURRENT});
        const auto partial = unavailable.measure();
        TEST_ASSERT_FALSE(partial.isValid);
        TEST_ASSERT_TRUE(isnan(partial.current.value()));
        TEST_ASSERT_EQUAL_FLOAT(failedSample.voltage, partial.voltage.value());
        assertValues(available.measure(), healthySample);
        fakeNow = 2000;
        TEST_ASSERT_FALSE(unavailable.measure().isValid);
        assertValues(available.measure(), healthySample);
        TEST_ASSERT_EQUAL_UINT8(2, unavailable.badTicks());
        failed.clearLog();
        TEST_ASSERT_FALSE(unavailable.measure().isValid);
        const auto steps = failed.steps();
        TEST_ASSERT_TRUE(std::find(steps.begin(), steps.end(), "begin") != steps.end());
        assertValues(available.measure(), healthySample);
        TEST_ASSERT_EQUAL_UINT8(0, available.badTicks());
        failed.nack = nullptr;
        assertValues(unavailable.measure(), failedSample);
    }
}

static void test_rebinding_to_another_bus_keeps_the_first_bus()
{
    TwoWire firstWire, secondWire;
    Device firstDevice, secondDevice;
    attach(firstWire, PACK, firstDevice);
    attach(secondWire, PACK, secondDevice);
    Ina228Adapter first(PACK), second(PACK);
    TEST_ASSERT_TRUE(first.begin(&firstWire));
    TEST_ASSERT_TRUE(second.begin(&secondWire));
    firstDevice.busVolts = 12.0;
    secondDevice.busVolts = 24.0;
    assertValues(first.measure(), uncalibrated(12.0f));
    assertValues(second.measure(), uncalibrated(24.0f));
    // The SparkFun bus binds its TwoWire once, so a later begin on another bus
    // still talks to the first one.
    firstDevice.clearLog();
    secondDevice.clearLog();
    TEST_ASSERT_TRUE(first.begin(&secondWire));
    assertValues(first.measure(), uncalibrated(12.0f));
    TEST_ASSERT_TRUE(secondDevice.steps().empty());
    TEST_ASSERT_FALSE(first.begin(nullptr));
    firstDevice.clearLog();
    for (int i = 0; i < 3; ++i) assertUnavailable(first.measure());
    TEST_ASSERT_TRUE(firstDevice.steps().empty());
    assertValues(second.measure(), uncalibrated(24.0f));
}

static void test_transient_failures_clear_and_qualified_failure_keeps_partial_sample()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_TRUE(sensor.begin(&wire));
    applyPackSample(device);
    for (int round = 0; round < 2; ++round)
    {
        device.nack = failReads({Register::CURRENT});
        for (int n = 1; n <= 2; ++n)
        {
            device.clearLog();
            TEST_ASSERT_FALSE(sensor.measure().isValid);
            TEST_ASSERT_EQUAL_UINT8(n, sensor.badTicks());
            TEST_ASSERT_TRUE(sensor.isUp());
            assertSteps(device, {"voltage", "current", "power", "charge"});
        }
        device.nack = nullptr;
        assertValues(sensor.measure(), PACK_SAMPLE);
        TEST_ASSERT_EQUAL_UINT8(0, sensor.badTicks());
    }
    device.nack = failReads({Register::CURRENT});
    TEST_ASSERT_FALSE(sensor.measure().isValid);
    TEST_ASSERT_FALSE(sensor.measure().isValid);
    device.clearLog();
    const auto third = sensor.measure();
    TEST_ASSERT_FALSE(third.isValid);
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.voltage, third.voltage.value());
    TEST_ASSERT_TRUE(isnan(third.current.value()));
    TEST_ASSERT_TRUE(sensor.isUp());
    assertSteps(device, {"voltage", "current", "power", "charge", "begin", "alert", "mode", "calibrate"});
}

static void test_runtime_failed_restart_and_recovered_bad_read_count_once()
{
    TwoWire wire;
    Device device(PACK_OHMS);
    attach(wire, PACK, device);
    Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
    TEST_ASSERT_TRUE(sensor.begin(&wire));
    applyPackSample(device);
    device.nack = failReads({Register::CURRENT});
    device.deviceId = WRONG_DEVICE_ID;
    TEST_ASSERT_FALSE(sensor.measure().isValid);
    TEST_ASSERT_FALSE(sensor.measure().isValid);
    const auto third = sensor.measure();
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.voltage, third.voltage.value());
    TEST_ASSERT_FALSE(sensor.isUp());
    assertUnavailable(sensor.measure());
    assertUnavailable(sensor.measure());
    fakeNow = 1999;
    device.clearLog();
    assertUnavailable(sensor.measure());
    TEST_ASSERT_TRUE(device.steps().empty());
    fakeNow = 2000;
    device.deviceId = ina228_native::INA228_DEVICE_ID;
    const auto recovered = sensor.measure();
    TEST_ASSERT_FALSE(recovered.isValid);
    TEST_ASSERT_EQUAL_FLOAT(PACK_SAMPLE.voltage, recovered.voltage.value());
    TEST_ASSERT_EQUAL_UINT8(0, sensor.badTicks());
    assertSteps(device, {"begin", "alert", "mode", "calibrate", "voltage", "current", "power", "charge"});
    device.clearLog();
    TEST_ASSERT_FALSE(sensor.measure().isValid);
    TEST_ASSERT_EQUAL_UINT8(1, sensor.badTicks());
    device.nack = nullptr;
    assertValues(sensor.measure(), PACK_SAMPLE);
}

static void test_continuous_read_failures_respect_cooldown_and_rollover()
{
    for (uint32_t start : {uint32_t(0), UINT32_MAX - 999u})
    {
        TwoWire wire;
        Device device(PACK_OHMS);
        attach(wire, PACK, device);
        Ina228Adapter sensor(PACK, PACK_OHMS, PACK_MAX_AMPS, true);
        TEST_ASSERT_TRUE(sensor.begin(&wire));
        fakeNow = start;
        device.nack = failReads({Register::VBUS, Register::CURRENT, Register::POWER, Register::CHARGE});
        for (int n = 0; n < 3; ++n) assertUnavailable(sensor.measure());
        fakeNow = start + 1999u;
        device.clearLog();
        for (int n = 0; n < 3; ++n) assertUnavailable(sensor.measure());
        TEST_ASSERT_EQUAL_UINT(12, device.steps().size());
        fakeNow = start + 2000u;
        device.clearLog();
        assertUnavailable(sensor.measure());
        assertSteps(device, {"voltage", "current", "power", "charge", "begin", "alert", "mode", "calibrate"});
        TEST_ASSERT_TRUE(sensor.isUp());
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_transient_failures_clear_and_qualified_failure_keeps_partial_sample);
    RUN_TEST(test_runtime_failed_restart_and_recovered_bad_read_count_once);
    RUN_TEST(test_continuous_read_failures_respect_cooldown_and_rollover);
    RUN_TEST(test_unbound_adapter_does_not_attempt_recovery);
    RUN_TEST(test_initial_configuration_and_current_snapshot);
    RUN_TEST(test_shunt_and_accumulator_configuration_are_explicit);
    RUN_TEST(test_each_configuration_failure_stops_setup);
    RUN_TEST(test_all_read_failure_combinations_preserve_successful_fields);
    RUN_TEST(test_readings_round_to_register_resolution);
    RUN_TEST(test_raw_accessors_preserve_units_and_handle_error_status);
    RUN_TEST(test_pack_recovery_preserves_charge_and_reads_in_same_call);
    RUN_TEST(test_recovery_reconfigures_when_charge_preservation_is_disabled);
    RUN_TEST(test_absence_and_failed_recovery_respect_exact_retry_boundary);
    RUN_TEST(test_failed_recovery_configuration_returns_no_measurement);
    RUN_TEST(test_monitor_failures_and_recovery_are_independent);
    RUN_TEST(test_rebinding_to_another_bus_keeps_the_first_bus);
    return UNITY_END();
}

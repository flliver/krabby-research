#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "unity.h"
#include <Arduino.h>
#include "src/power_monitor/ina228_adapter.h"
#include "src/power_monitor/power_calibration.h"
#include "src/power_monitor/power_measurement.h"
#include "src/display/display_renderer.h"
#include "trace_canvas.h"

namespace powerPollFake
{
ina228_native::Device devices[2];
Counters counters[2];
uint32_t voltageReadDuration[2] = {};
uint32_t now = 0;
std::vector<std::string> events;
}

void delay(unsigned long) {}

#include "state.inc"
#include "calibration.inc"
#include "poll.inc"

namespace
{
std::string recordingDirectory;

// Capture declaration-initialized production state before any scenario runs.
const auto initialPack = packPowerMonitor;
const auto initialMidpoint = midpointPowerMonitor;
const auto initialCalibration = powerCalibration;
const auto initialInferredB = inferredBattBVoltage;
const auto initialPackMeasurement = packMeasurement;
const auto initialMidpointMeasurement = midpointMeasurement;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

constexpr uint16_t WRONG_DEVICE_ID = 0x2380;

// NACK reads of the given INA228 registers.
std::function<bool(uint8_t, bool, uint16_t)> failReads(std::initializer_list<uint8_t> registers)
{
    const std::vector<uint8_t> failing(registers);
    return [failing](uint8_t reg, bool isWrite, uint16_t) -> bool {
        return !isWrite && std::find(failing.begin(), failing.end(), reg) != failing.end();
    };
}

// Monitors answer on Wire, and their register traffic becomes the recorded events
// and counters. Alert and mode writes were never part of the recorded trace.
void attachMonitors()
{
    for (size_t i = 0; i < 2; ++i)
    {
        const uint8_t address = i == 0 ? PACK_POWER_MONITOR_ADDRESS : MIDPOINT_POWER_MONITOR_ADDRESS;
        Wire.state().devices[address] = &powerPollFake::devices[i];
        powerPollFake::devices[i].onStep = [i, address](const std::string &step) {
            if (step == "alert" || step == "mode") return;
            auto &observed = powerPollFake::counters[i];
            if (step == "begin") ++observed.beginCount;
            if (step == "calibrate") ++observed.shuntCount;
            if (step == "accumulators") ++observed.resetCount; // the recorded resets are charge resets
            powerPollFake::event(address, step == "accumulators" ? "reset-charge" : step);
            if (step == "voltage") powerPollFake::now += powerPollFake::voltageReadDuration[i];
        };
    }
}

// Compare values without tolerances: preserve signed zero; NaN payloads are not
// part of the contract. Infinity must retain its sign.
bool sameValue(float actual, float expected)
{
    if (std::isnan(expected)) return std::isnan(actual);
    return actual == expected && (actual != 0.0f || std::signbit(actual) == std::signbit(expected));
}

void verifyMeasurement(const std::string &telemetry)
{
    const auto &pack = packMeasurement;
    const auto &midpoint = midpointMeasurement;
    std::istringstream fields(telemetry);
    const std::vector<std::string> tokens{std::istream_iterator<std::string>(fields),
        std::istream_iterator<std::string>()};
    require(tokens.size() == 11, "unexpected battery telemetry shape");
    require((tokens[9] == "1") == pack.isValid, "pack flag must reflect acquisition availability");
    require((tokens[10] == "1") == midpoint.isValid, "midpoint flag must reflect acquisition availability");
    require(sameValue(inferredBattBVoltage.value(), (pack.voltage - midpoint.voltage).value()), "inferred B must equal current difference");
    const float packFields[] = {pack.voltage.value(), pack.current.value(),
        pack.power.value(), pack.charge.value()};
    for (size_t i = 0; i < 4; ++i)
    {
        Print expected;
        expected.output = "x"; // Avoid recording a second telemetry event.
        expected.print(packFields[i], i < 2 ? 2 : 1);
        require(tokens[i + 1] == expected.output.substr(1),
            "telemetry must use current pack measurement");
    }
    // Battery fields use this poll's midpoint and inferred B readings.
    const Volts batteryFields[] = {midpoint.voltage, pack.voltage - midpoint.voltage};
    for (size_t i = 0; i < 2; ++i)
    {
        Print expected;
        expected.output = "x";
        expected.print(batteryFields[i].value(), 2);
        require(tokens[i + 5] == expected.output.substr(1), "telemetry must use current battery values");
    }
    require((tokens[7] == "1") == (!isfinite(midpoint.voltage.value()) || !isfinite((pack.voltage - midpoint.voltage).value()) || fabs((midpoint.voltage - (pack.voltage - midpoint.voltage)).value()) > BATTERY_DIVERGENCE_THRESHOLD.value()), "divergence must use current values");
    require((tokens[10] == "1") == midpointMeasurement.isValid, "measurement midpoint validity differs");
}

void test_measurement_defaults_and_exact_comparison()
{
    const PowerMonitorMeasurement pack;
    const PowerMonitorMeasurement midpoint;
    require(isnan(inferredBattBVoltage.value()), "inferred B defaults to NaN");
    require(!pack.isValid && !midpoint.isValid, "availability defaults to false");
    const float values[] = {pack.voltage.value(), pack.current.value(), pack.power.value(),
        pack.charge.value(), midpoint.voltage.value()};
    for (float value : values) require(isnan(value), "unread measurements must default to NaN");
    require(!sameValue(-0.0f, 0.0f), "comparison must distinguish signed zero");
    require(sameValue(-0.0f, -0.0f), "comparison must preserve negative zero");
    require(sameValue(NAN, NAN) && !sameValue(0.0f, NAN), "comparison must classify NaN");
    require(sameValue(INFINITY, INFINITY) && !sameValue(-INFINITY, INFINITY), "comparison must distinguish infinities");
    require(!sameValue(1.0f, std::nextafter(1.0f, 2.0f)), "comparison must reject even one ULP difference");
}

class Storage
{
public:
    PowerCalibrationRecord record = {};
    bool corruptWrite = false;
    void load(PowerCalibrationRecord &out) { out = record; }
    void writeRecord(const PowerCalibrationRecord &candidate)
    {
        powerPollFake::events.push_back("storage.write-pending");
        record = candidate;
        if (corruptWrite) record.packShuntScale = -1.0f;
    }
    void updateMagic(uint8_t magic)
    {
        powerPollFake::events.push_back("storage.magic=" + std::to_string(magic));
        record.magic = magic;
    }
};

class Scenario
{
public:
    Scenario() : file_(tmpfile()), canvas_(file_), renderer_(canvas_)
    {
        require(file_ != nullptr, "cannot open trace file");
        text_.imbue(std::locale::classic());
        text_ << std::setprecision(std::numeric_limits<float>::max_digits10);
        // The initial board-only frame is constant. Record every subsequent
        // drawing operation, so changes smaller than a pixel must stay silent.
        renderer_.render(DisplayFrame{});
        offset_ = ftell(file_);
        require(offset_ >= 0, "cannot locate initial trace position");
    }
    ~Scenario() { if (file_) fclose(file_); } // Fallback cleanup during exceptions.
    Scenario(const Scenario &) = delete;
    Scenario &operator=(const Scenario &) = delete;

    void begin(bool packPresent = true, bool midpointPresent = true)
    {
        powerPollFake::devices[0].present = packPresent;
        powerPollFake::devices[1].present = midpointPresent;
        packPowerMonitor.begin(&Wire);
        midpointPowerMonitor.begin(&Wire);
    }

    void poll(const char *label, uint32_t now)
    {
        powerPollFake::now = now;
        Print output;
        readPowerMeasurements();
        Print *mainSerial = &output;
#include "telemetry.inc"
        const BoardRole currentRole = ROLE_UNKNOWN;
        ControllerFreshnessTracker controllerFreshnessTrackers[BOARD_ROLE_COUNT]{};
        ActuatorStatus latestActuatorStatus[ActuatorId::ActuatorCount]{};
        const ImuMeasurement imuMeasurement{};
        const uint32_t nowMilliseconds = now;
        const struct { int pwmDeadband; } ACTUATOR_CONFIG = {20};
#include "display.inc"
        verifyMeasurement(output.output);
        const bool drew = renderer_.render(displayFrame);
        text_ << "[" << label << "]\n" << output.output << "\n";
        text_ << "events=";
        for (size_t i = 0; i < powerPollFake::events.size(); ++i)
        {
            if (i) text_ << " | ";
            text_ << powerPollFake::events[i];
        }
        text_ << "\nclock-after=" << powerPollFake::now << "\n";
        powerPollFake::events.clear();
        text_ << "measurement.pack=" << packMeasurement.voltage.value() << ","
              << packMeasurement.current.value() << "," << packMeasurement.power.value()
              << "," << packMeasurement.charge.value() << "\n"
              << "model=" << displayFrame.packVoltage.value() << ","
              << displayFrame.batteryDecivolts[0] << "," << displayFrame.batteryDecivolts[1]
              << "," << displayFrame.batteryLevel[0] << "," << displayFrame.batteryLevel[1] << "\n"
              << "calibration=" << powerCalibration.packVoltageOffset().value() << ","
              << powerCalibration.midpointVoltageOffset().value() << ","
              << powerCalibration.packShuntScale() << "\n";
        for (size_t i = 0; i < 2; ++i)
        {
            const auto &device = powerPollFake::counters[i];
            const auto &adapter = i == 0 ? packPowerMonitor : midpointPowerMonitor;
            text_ << (i == 0 ? "pack" : "mid") << ".up/bad/begins/shunts/resets="
                  << adapter.isUp() << "," << unsigned(adapter.badTicks()) << ","
                  << device.beginCount << "," << device.shuntCount << "," << device.resetCount << "\n";
        }
        text_ << "draw=" << drew << "\n" << drawCalls() << "\n";
    }

    void saveResult(PowerCalibration::SaveResult result)
    {
        text_ << "save-result=" << static_cast<int>(result) << "\n";
    }

    void verify(const char *name)
    {
        FILE *finished = file_;
        file_ = nullptr;
        require(fclose(finished) == 0, "cannot close trace file");
        const std::string actual = text_.str();
        const std::string filename = std::string(name) + ".txt";
        if (!recordingDirectory.empty())
        {
            std::ofstream out(recordingDirectory + "/" + filename, std::ios::binary);
            require(out.good(), "record directory must already exist");
            out << actual;
            out.close();
            require(!out.fail(), "cannot write/close recorded fixture");
            return;
        }
        std::ifstream in(std::string(KRABBY_POWER_POLL_FIXTURES) + "/" + filename, std::ios::binary);
        require(in.good(), ("cannot open fixture: " + filename).c_str());
        const std::string expected((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        require(!in.bad(), "cannot read fixture");
        in.close();
        require(!in.fail(), "cannot close fixture");
        if (expected != actual)
        {
            std::istringstream expectedLines(expected), actualLines(actual);
            std::string e, a, scenario = "initial state";
            size_t line = 0;
            while (true)
            {
                const bool hasExpected = static_cast<bool>(std::getline(expectedLines, e));
                const bool hasActual = static_cast<bool>(std::getline(actualLines, a));
                ++line;
                if (!a.empty() && a[0] == '[') scenario = a;
                if (hasExpected != hasActual || e != a)
                    throw std::runtime_error(filename + " " + scenario + " line " +
                        std::to_string(line) + "\nexpected: " + (hasExpected ? e : "<EOF>") +
                        "\nactual:   " + (hasActual ? a : "<EOF>"));
                if (!hasExpected) throw std::runtime_error(filename + " trailing newline differs");
            }
        }
    }
private:
    std::string drawCalls()
    {
        require(fflush(file_) == 0, "cannot flush trace");
        const long end = ftell(file_);
        require(end >= offset_, "cannot locate trace position");
        const size_t count = static_cast<size_t>(end - offset_);
        std::string result(count, '\0');
        require(fseek(file_, offset_, SEEK_SET) == 0, "cannot seek trace for reading");
        if (count) require(fread(&result[0], 1, count, file_) == count, "cannot read trace");
        require(fseek(file_, end, SEEK_SET) == 0, "cannot seek trace for writing");
        offset_ = end;
        return result;
    }
    FILE *file_;
    TraceCanvas canvas_;
    DisplayRenderer<TraceCanvas> renderer_;
    long offset_ = 0;
    std::ostringstream text_;
};

void readings(float packV = 26.5f, float midV = 13.25f)
{
    // Power follows from voltage and current, as on the chip.
    powerPollFake::devices[0].busVolts = packV;
    powerPollFake::devices[0].amps = -12.5;
    powerPollFake::devices[0].coulombs = 1200.0;
    powerPollFake::devices[1].busVolts = midV;
}

void test_healthy_divergence_and_visible_resolution()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    s.poll("identical", 1050);
    // Exact VBUS multiples (195.3125 uV) just above the 2-dp rounding boundary, so
    // register rounding cannot flip the printed battery voltages.
    readings(26.51015625f, 13.255078125f);
    s.poll("below-display-resolution", 1100);
    readings(27.0f, 13.25f);
    s.poll("exact-divergence-threshold", 1150);
    readings(27.125f, 13.25f);
    s.poll("above-divergence-threshold", 1200);
    readings(28.0f, 14.0f);
    s.poll("both-bars-full", 1250);
    readings(28.1f, 14.0f);
    s.poll("voltage-change-with-full-bars", 1300);
    s.verify("healthy");
}

void test_startup_and_initial_recovery()
{
    Scenario s;
    s.begin(false, false);
    readings();
    s.poll("never-read", 0);
    s.poll("second-failure", 50);
    s.poll("first-probe-absent", 100);
    powerPollFake::devices[0].present = true;
    powerPollFake::devices[1].present = true;
    s.poll("retry-count-one", 1899);
    s.poll("retry-count-two", 1999);
    s.poll("present-before-retry", 2099);
    s.poll("recovery-tick-still-invalid", 2100);
    s.poll("first-read-after-recovery", 2150);
    s.verify("startup");
}

void test_only_one_monitor_at_startup()
{
    Scenario s;
    s.begin(false, true);
    readings();
    s.poll("midpoint-only-first-sample", 1000);
    powerPollFake::devices[0].present = true;
    s.poll("pack-waiting-for-third-failure", 1050);
    s.poll("pack-reinitialized-no-sample-yet", 1100);
    s.poll("both-available", 1150);
    s.verify("midpoint_only_startup");
}

void test_pack_only_at_startup()
{
    Scenario s;
    s.begin(true, false);
    readings();
    s.poll("pack-only-first-sample", 1000);
    s.verify("pack_only_startup");
}

void test_partial_read_failure_recovers_without_discarding_voltage()
{
    Scenario s;
    s.begin();
    readings();
    powerPollFake::devices[0].nack = failReads({ina228_native::CURRENT});
    for (uint32_t now : {1000u, 1050u, 1100u})
    {
        s.poll("current-read-failed", now);
        require(!packMeasurement.isValid && isnan(packMeasurement.current.value()), "current failure must invalidate sample");
        require(packMeasurement.voltage.value() == 26.5f, "current failure must preserve voltage");
        require(midpointMeasurement.isValid, "pack failure must not invalidate midpoint");
    }
    require(powerPollFake::counters[0].beginCount == 2, "three failed samples must restart pack");
    require(powerPollFake::counters[0].resetCount == 1, "recovery must preserve charge");
    powerPollFake::devices[0].nack = nullptr;
    s.poll("current-read-restored", 1150);
    require(packMeasurement.isValid, "successful reads must restore validity");
    s.verify("partial_read_recovery");
}

void test_pack_disconnect_and_recovery_keeps_midpoint_live()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    readings(26.0f, 12.75f);
    powerPollFake::devices[0].present = false;
    s.poll("pack-read-failed-midpoint-changed", 1050);
    s.poll("pack-second-failed-sample", 1100);
    s.poll("pack-restart-failed", 1150);
    readings(26.0f, 12.5f);
    powerPollFake::devices[0].present = true;
    s.poll("retry-count-one", 2949);
    s.poll("retry-count-two", 3049);
    s.poll("reconnected-before-deadline", 3149);
    s.poll("recovery-publishes-current-sample", 3150);
    s.poll("new-pack-reading-next-tick", 3200);
    s.verify("pack_disconnect");
}

void test_midpoint_disconnect_keeps_pack_live()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    readings(25.5f);
    powerPollFake::devices[1].present = false;
    s.poll("midpoint-failed-pack-changed", 1050);
    s.poll("midpoint-second-failed-sample", 1100);
    s.poll("midpoint-restart-failed", 1150);
    readings(25.0f, 12.25f);
    powerPollFake::devices[1].present = true;
    s.poll("retry-count-one", 2949);
    s.poll("retry-count-two", 3049);
    s.poll("midpoint-before-retry", 3149);
    s.poll("midpoint-driver-recovery", 3150);
    s.poll("split-updated-next-tick", 3200);
    s.verify("midpoint_disconnect");
}

void test_impossible_split_reports_B_unavailable()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    readings(10.0f, 12.0f);
    s.poll("both-monitors-valid-but-negative-battery-B", 1050);
    readings(40.0f, 10.0f);
    s.poll("both-monitors-valid-but-battery-B-above-20V", 1100);
    readings(26.0f, 13.0f);
    s.poll("consistent-again", 1150);
    s.verify("impossible_split");
}

void test_divergence_is_not_retained_when_midpoint_fails()
{
    Scenario s;
    s.begin();
    readings(26.0f, 12.5f);
    s.poll("diverged", 1000);
    powerPollFake::devices[1].nack = failReads({ina228_native::VBUS});
    s.poll("midpoint-fails-pack-remains-valid", 1050);
    require(packMeasurement.isValid && !midpointMeasurement.isValid,
        "a failed midpoint read must not hide the pack");
    s.verify("divergence_then_missing_midpoint");
}

void test_both_monitors_fail_after_a_good_sample()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    for (auto &device : powerPollFake::devices) device.nack = failReads({ina228_native::VBUS});
    s.poll("both-fail-clear-battery-telemetry-hide-display", 1050);
    s.poll("both-down", 1100);
    s.poll("both-recover-no-read-yet", 1150);
    for (auto &device : powerPollFake::devices) device.nack = nullptr;
    readings(26.0f, 12.5f);
    s.poll("fresh-values-after-recovery", 1200);
    s.verify("both_disconnect");
}

void test_voltage_boundaries_and_invalid_values()
{
    Scenario s;
    s.begin();
    readings(0.0f, 0.0f);
    s.poll("zero-is-valid", 0);
    readings(40.0f, 20.0f);
    s.poll("inclusive-upper-limits", 50);
    readings(40.001f, 20.001f);
    s.poll("above-limits", 100);
    s.poll("down-no-read", 150);
    s.poll("ack-recovery", 200);
    s.verify("voltage_boundaries");
}

void test_retry_rollover()
{
    Scenario s;
    s.begin(false, false);
    readings();
    s.poll("failure-one", UINT32_MAX - 150);
    s.poll("failure-two", UINT32_MAX - 100);
    s.poll("probe-before-rollover", UINT32_MAX - 50);
    powerPollFake::devices[0].present = true;
    powerPollFake::devices[1].present = true;
    s.poll("retry-count-one", 1748);
    s.poll("retry-count-two", 1848);
    s.poll("elapsed-1999-after-rollover", 1948);
    s.poll("elapsed-2000-after-rollover", 1949);
    s.poll("read-after-recovery", 1999);
    s.verify("retry_rollover");
}

void test_recovery_uses_one_timestamp_before_blocking_reads()
{
    Scenario s;
    s.begin(false, true);
    readings();
    s.poll("failure-one", 1000);
    s.poll("failure-two", 1050);
    powerPollFake::voltageReadDuration[1] = 75;
    s.poll("probe-timestamp-precedes-midpoint-read", 1100);
    powerPollFake::devices[0].present = true;
    s.poll("retry-count-one", 2899);
    s.poll("retry-count-two", 2999);
    s.poll("read-crosses-retry-deadline", 3099);
    s.poll("retry-eligible-at-start-of-tick", 3174);
    s.poll("recovered-read", 3250);
    s.verify("read_order_and_clock");
}

void test_failed_begin_retries_without_publishing_a_read()
{
    Scenario s;
    powerPollFake::devices[0].deviceId = WRONG_DEVICE_ID;
    s.begin();
    readings();
    s.poll("failed-begin", 1000);
    s.poll("waiting", 1050);
    s.poll("ack-but-begin-fails", 1100);
    powerPollFake::devices[0].deviceId = ina228_native::INA228_DEVICE_ID;
    s.poll("retry-count-one", 2899);
    s.poll("retry-count-two", 2999);
    s.poll("waiting-before-interval", 3099);
    s.poll("successful-begin-still-invalid", 3100);
    s.poll("first-measurement", 3150);
    s.verify("failed_begin");
}

void test_calibration_changes_rejections_and_reload()
{
    Scenario s;
    Storage storage;
    s.begin();
    readings(26.0f, 13.0f);
    s.poll("identity", 1000);
    s.saveResult(powerCalibration.captureVoltage(storage, Volts(26), Volts(13), Volts(26.5f), Volts(13.25f)));
    s.saveResult(powerCalibration.captureCurrent(storage, Amps(-12.5f), Amps(-15.0f)));
    s.poll("offset-and-gain-applied", 1050);
    s.saveResult(powerCalibration.captureVoltage(storage, Volts(26), Volts(13), Volts(40), Volts(13)));
    s.saveResult(powerCalibration.captureCurrent(storage, Amps(-12.5f), Amps(15.0f)));
    s.poll("rejected-input-keeps-calibration", 1100);
    powerCalibration = PowerCalibration{};
    require(powerCalibration.load(storage), "saved calibration must reload");
    s.poll("reload-keeps-calibration", 1150);
    storage.corruptWrite = true;
    s.saveResult(powerCalibration.captureCurrent(storage, Amps(-12.5f), Amps(-20.0f)));
    s.poll("failed-save-keeps-RAM-calibration", 1200);
    powerCalibration = PowerCalibration{};
    require(!powerCalibration.load(storage), "failed save must invalidate persisted calibration");
    s.poll("failed-save-invalidates-persisted-calibration", 1250);
    s.verify("calibration");
}

void test_midpoint_begin_failure_then_recovery()
{
    Scenario s;
    powerPollFake::devices[1].deviceId = WRONG_DEVICE_ID;
    s.begin();
    readings();
    s.poll("midpoint-begin-failed", 1000);
    s.poll("second-failure", 1050);
    s.poll("midpoint-retry-begin-failed", 1100);
    powerPollFake::devices[1].deviceId = ina228_native::INA228_DEVICE_ID;
    s.poll("retry-count-one", 1150);
    s.poll("retry-count-two", 1200);
    s.poll("before-deadline", 3099);
    s.poll("midpoint-recovered-still-invalid", 3100);
    require(midpointPowerMonitor.isUp() && midpointMeasurement.isValid, "midpoint recovery must return a fresh sample");
    s.poll("fresh-midpoint", 3150);
    s.verify("midpoint_failed_begin");
}

void test_staggered_recovery_after_both_fail()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    powerPollFake::devices[0].present = false;
    powerPollFake::devices[1].present = false;
    s.poll("both-fail", 1050);
    powerPollFake::devices[1].present = false;
    s.poll("second-failure", 1100);
    powerPollFake::devices[0].present = true;
    s.poll("only-pack-recovers", 1150);
    require(packPowerMonitor.isUp() && !midpointPowerMonitor.isUp(), "pack reads resume while midpoint restart fails");
    readings(26.0f, 12.5f);
    s.poll("pack-fresh-midpoint-down", 1200);
    require(packMeasurement.isValid && !midpointMeasurement.isValid, "midpoint outage must not invalidate pack");
    s.poll("midpoint-waits", 1250);
    powerPollFake::devices[1].present = true;
    s.poll("midpoint-recovers", 3150);
    s.poll("both-fresh", 3200);
    s.verify("staggered_recovery");
}

void test_second_disconnect_preserves_retry_interval_and_charge()
{
    Scenario s;
    s.begin();
    readings();
    s.poll("healthy", 1000);
    powerPollFake::devices[0].nack = failReads({ina228_native::VBUS});
    s.poll("first-failure", 1050);
    s.poll("second-failure", 1100);
    s.poll("first-recovery", 1150);
    readings();
    powerPollFake::devices[0].nack = nullptr;
    powerPollFake::devices[0].coulombs = 1234.375;
    s.poll("fresh-after-first-recovery", 1200);
    powerPollFake::devices[0].nack = failReads({ina228_native::VBUS});
    s.poll("fails-again", 1250);
    s.poll("second-failure-again", 1300);
    s.poll("count-ready-interval-not-ready", 1350);
    s.poll("before-second-retry", 3149);
    require(packPowerMonitor.isUp() && packPowerMonitor.badTicks() >= 3 && powerPollFake::counters[0].beginCount == 2, "qualified failures must respect cooldown");
    s.poll("second-recovery", 3150);
    require(!packMeasurement.isValid && packMeasurement.charge.value() == powerPollFake::devices[0].coulombs, "charge survives a failed voltage read");
    readings();
    powerPollFake::devices[0].nack = nullptr;
    powerPollFake::devices[0].coulombs = 1237.5;
    s.poll("fresh-after-second-recovery", 3200);
    require(powerPollFake::counters[0].resetCount == 1, "recovery must not reset charge");
    s.verify("repeated_disconnect");
}

void test_calibration_while_pack_unavailable()
{
    Scenario s;
    Storage storage;
    s.begin();
    readings(26.0f, 13.0f);
    s.poll("healthy", 1000);
    powerPollFake::devices[0].nack = failReads({ina228_native::VBUS});
    s.poll("pack-fails", 1050);
    s.saveResult(powerCalibration.captureVoltage(storage, Volts(26), Volts(13), Volts(26.5f), Volts(13.25f)));
    s.saveResult(powerCalibration.captureCurrent(storage, Amps(-12.5f), Amps(-15.0f)));
    s.poll("calibration-changed-while-pack-down", 1100);
    require(isnan(packMeasurement.voltage.value()) && fabs(packMeasurement.current.value() + 15.0f) < 0.00001f, "calibration must preserve NaN and correct current");
    require(midpointMeasurement.isValid && !packMeasurement.isValid && midpointMeasurement.voltage.value() == 13.25f, "live midpoint uses new calibration independently");
    s.poll("pack-recovered-still-stale", 1150);
    powerPollFake::devices[0].nack = nullptr;
    readings(26.0f, 13.0f);
    s.poll("fresh-pack-uses-new-calibration", 1200);
    s.verify("calibration_while_unavailable");
}

void test_divergence_with_battery_A_higher()
{
    Scenario s;
    s.begin();
    readings(26.5f, 13.5f);
    s.poll("A-higher-exact-threshold", 1000);

    readings(26.375f, 13.5f);
    s.poll("A-higher-above-threshold", 1050);

    readings(26.625f, 13.5f);
    s.poll("A-higher-below-threshold", 1100);
    s.verify("divergence_A_higher");
}

void test_calibrated_voltage_is_used_for_validity()
{
    Scenario s;
    Storage storage;
    s.begin();
    s.saveResult(powerCalibration.captureVoltage(storage, Volts(26), Volts(13), Volts(27), Volts(14)));
    readings(39.5f, 19.5f);
    s.poll("raw-in-range-corrected-out-of-range", 1000);
    s.poll("wait", 1050);
    s.poll("recover", 1100);
    s.verify("calibrated_validity");
}
}

void setUp()
{
    powerPollFake::devices[0] = ina228_native::Device(PACK_SHUNT_RESISTANCE_OHMS);
    powerPollFake::devices[1] = ina228_native::Device();
    for (size_t i = 0; i < 2; ++i)
    {
        powerPollFake::counters[i] = powerPollFake::Counters{};
        powerPollFake::voltageReadDuration[i] = 0;
    }
    powerPollFake::events.clear();
    powerPollFake::now = 0;
    Wire.reset();
    attachMonitors();
    packPowerMonitor = initialPack;
    midpointPowerMonitor = initialMidpoint;
    powerCalibration = initialCalibration;
    inferredBattBVoltage = initialInferredB;
    packMeasurement = initialPackMeasurement;
    midpointMeasurement = initialMidpointMeasurement;
}
void tearDown()
{
    TEST_ASSERT_TRUE(Wire.state().errors.empty());
}

int runTests(int argc, char **argv)
{
    if (argc == 3 && std::string(argv[1]) == "--record") recordingDirectory = argv[2];
    else if (argc != 1) return 2;
    UNITY_BEGIN();
    RUN_TEST(test_measurement_defaults_and_exact_comparison);
    RUN_TEST(test_healthy_divergence_and_visible_resolution);
    RUN_TEST(test_startup_and_initial_recovery);
    RUN_TEST(test_only_one_monitor_at_startup);
    RUN_TEST(test_pack_only_at_startup);
    RUN_TEST(test_partial_read_failure_recovers_without_discarding_voltage);
    RUN_TEST(test_pack_disconnect_and_recovery_keeps_midpoint_live);
    RUN_TEST(test_midpoint_disconnect_keeps_pack_live);
    RUN_TEST(test_impossible_split_reports_B_unavailable);
    RUN_TEST(test_divergence_is_not_retained_when_midpoint_fails);
    RUN_TEST(test_both_monitors_fail_after_a_good_sample);
    RUN_TEST(test_voltage_boundaries_and_invalid_values);
    RUN_TEST(test_retry_rollover);
    RUN_TEST(test_recovery_uses_one_timestamp_before_blocking_reads);
    RUN_TEST(test_failed_begin_retries_without_publishing_a_read);
    RUN_TEST(test_calibration_changes_rejections_and_reload);
    RUN_TEST(test_calibrated_voltage_is_used_for_validity);
    RUN_TEST(test_midpoint_begin_failure_then_recovery);
    RUN_TEST(test_staggered_recovery_after_both_fail);
    RUN_TEST(test_second_disconnect_preserves_retry_interval_and_charge);
    RUN_TEST(test_calibration_while_pack_unavailable);
    RUN_TEST(test_divergence_with_battery_A_higher);
    return UNITY_END();
}

int main(int argc, char **argv)
{
    try { return runTests(argc, argv); }
    catch (const std::exception &error)
    {
        fprintf(stderr, "power harness failure: %s\n", error.what());
        return 1;
    }
}

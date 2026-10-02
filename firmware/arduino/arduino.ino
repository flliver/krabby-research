/*
 * Krabby-Uno: 18-Joint Distributed Controller (3 boards × 6 actuators)
 * Front: FL + FR, on USB. Left: RL + ML, on pins 14/15 (Serial3). Right: MR + RR, on pins 16/17 (Serial2).
 * All three boards use the same pinout; role election selects which 6 actuators this board drives.
 */

#include <Arduino.h>
// Use overloads, including unit-preserving abs, instead of Arduino's macro.
#undef abs
#include <EEPROM.h>
#include <math.h>
#include "src/imu/imu_calibrator.h"
#include "src/imu/lsm6dso_adapter.h"
#include "src/display/ssd1306_adapter.h"
#include "src/display/display_renderer.h"
#include "src/display/display_frame_model.h"
#include "board_pins.h"
#include "eeprom_layout.h"
#include "command.h"
#include "actuator_manager.h"
#include "src/imu/imu_constants.h"
#include "src/telemetry.h"
#include "src/power_monitor/ina228_adapter.h"
#include "src/power_monitor/power_calibration.h"
#include "src/power_monitor/power_measurement.h"
#include "src/cli/power_calibration_command.h"
#include "src/power_monitor/power_monitor_constants.h"
#include "version.h"

// --- Serial: left follower = Serial1 (TX1/RX1 on Krabby-Uno v0.1 shield), right follower = Serial2 ---
#define SERIAL_LEFT  Serial1  // pins 18 (TX1), 19 (RX1) — Krabby-Uno v0.1 shield Serial1 connector
#define SERIAL_RIGHT Serial2   // pins 16 (TX2), 17 (RX2) — Krabby-Uno v0.1 shield Serial2 connector
// Exact on the Mega's 16 MHz clock, with headroom for the leader to transmit
// joint data from all three controller boards plus I2C sensor data.
#define BAUD_RATE 250000
#define SYNC_TOKEN "SYNC"
#define ASSIGN_LEFT  "ROLE:LEFT"
#define ASSIGN_RIGHT "ROLE:RIGHT"

BoardRole currentRole = ROLE_UNKNOWN;

ControllerFreshnessTracker controllerFreshnessTrackers[BOARD_ROLE_COUNT];
ActuatorStatus latestActuatorStatus[ActuatorId::ActuatorCount];
ImuMeasurement imuMeasurement;
Ssd1306Adapter oledDisplay;
DisplayRenderer<Ssd1306Adapter> oledRenderer(oledDisplay);
unsigned long lastOledDrawMilliseconds = 0;
constexpr unsigned long OLED_REDRAW_INTERVAL_MILLISECONDS = 250;

static constexpr uint8_t EEPROM_ROLE_MAGIC = 0xAB;

static void saveRole(BoardRole r)
{
    EEPROM.update(EEPROM_ROLE_ADDR,     EEPROM_ROLE_MAGIC);
    EEPROM.update(EEPROM_ROLE_ADDR + 1, (uint8_t)r);
}

static BoardRole loadRole()
{
    if (EEPROM.read(EEPROM_ROLE_ADDR) != EEPROM_ROLE_MAGIC)
        return ROLE_UNKNOWN;
    uint8_t r = EEPROM.read(EEPROM_ROLE_ADDR + 1);
    if (r == ROLE_FRONT || r == ROLE_LEFT || r == ROLE_RIGHT)
        return (BoardRole)r;
    return ROLE_UNKNOWN;
}

static bool i2cAddressResponds(uint8_t address)
{
    Wire.clearWireTimeoutFlag();
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
}

static bool hasFrontImu()
{
    Wire.begin();
    Wire.setClock(I2C_DEFAULT_BUS_CLOCK_HZ);
    Wire.setWireTimeout(I2C_BUS_TIMEOUT_MICROSECONDS, true);
    return i2cAddressResponds(LSM6DSO_PRIMARY_ADDRESS) ||
           i2cAddressResponds(LSM6DSO_ALTERNATE_ADDRESS);
}

// --- All 18 actuators (names fixed; each board uses the same physical pins for its 6) ---
// Pin numbers from board_pins.h (KRABBY_PIN_REV 1 = legacy, 2 = MOTOR_HEADER_PINOUT).
// Leader/Default Board
LinearActuator flhy("FLHY", PIN_S0_PWMR, PIN_S0_PWML, PIN_S0_EN, A6, A0, 0);
LinearActuator flhl("FLHL", PIN_S1_PWMR, PIN_S1_PWML, PIN_S1_EN, A7, A1, 1);
LinearActuator flkl("FLKL", PIN_S2_PWMR, PIN_S2_PWML, PIN_S2_EN, A8, A2, 2);
LinearActuator frhy("FRHY", PIN_S3_PWMR, PIN_S3_PWML, PIN_S3_EN, A9, A3, 3);
LinearActuator frhl("FRHL", PIN_S4_PWMR, PIN_S4_PWML, PIN_S4_EN, A10, A4, 4);
LinearActuator frkl("FRKL", PIN_S5_PWMR, PIN_S5_PWML, PIN_S5_EN, A11, A5, 5);
// Left Follower Board
LinearActuator rlhy("RLHY", PIN_S0_PWMR, PIN_S0_PWML, PIN_S0_EN, A6, A0, 0);
LinearActuator rlhl("RLHL", PIN_S1_PWMR, PIN_S1_PWML, PIN_S1_EN, A7, A1, 1);
LinearActuator rlkl("RLKL", PIN_S2_PWMR, PIN_S2_PWML, PIN_S2_EN, A8, A2, 2);
LinearActuator mlhy("MLHY", PIN_S3_PWMR, PIN_S3_PWML, PIN_S3_EN, A9, A3, 3);
LinearActuator mlhl("MLHL", PIN_S4_PWMR, PIN_S4_PWML, PIN_S4_EN, A10, A4, 4);
LinearActuator mlkl("MLKL", PIN_S5_PWMR, PIN_S5_PWML, PIN_S5_EN, A11, A5, 5);
// Right Follower Board
LinearActuator rrhy("RRHY", PIN_S0_PWMR, PIN_S0_PWML, PIN_S0_EN, A6, A0, 0);
LinearActuator rrhl("RRHL", PIN_S1_PWMR, PIN_S1_PWML, PIN_S1_EN, A7, A1, 1);
LinearActuator rrkl("RRKL", PIN_S2_PWMR, PIN_S2_PWML, PIN_S2_EN, A8, A2, 2);
LinearActuator mrhy("MRHY", PIN_S3_PWMR, PIN_S3_PWML, PIN_S3_EN, A9, A3, 3);
LinearActuator mrhl("MRHL", PIN_S4_PWMR, PIN_S4_PWML, PIN_S4_EN, A10, A4, 4);
LinearActuator mrkl("MRKL", PIN_S5_PWMR, PIN_S5_PWML, PIN_S5_EN, A11, A5, 5);

// Role → which 6 actuators this board drives (no mutation)
static const size_t ACT_COUNT = 6;
LinearActuator* ACT_LIST_FRONT[]  = { &flhy, &flhl, &flkl, &frhy, &frhl, &frkl };
LinearActuator* ACT_LIST_LEFT[]   = { &rlhy, &rlhl, &rlkl, &mlhy, &mlhl, &mlkl };  // RL + ML
LinearActuator* ACT_LIST_RIGHT[]  = { &rrhy, &rrhl, &rrkl, &mrhy, &mrhl, &mrkl }; // MR + RR

// Set once after role election.
ActuatorManager* actuatorManager = nullptr;
HardwareSerial* mainSerial = nullptr;  // USB (front) or uplink (left/right)
HardwareSerial* leftSerial = nullptr;  // serial to left board (from front only)
HardwareSerial* rightSerial = nullptr; // serial to right board (from front only)

const LinearActuator::ControlConfig ACTUATOR_CONFIG = {
    5,  // PWM_RAMP_STEP
    10, // RAMP_INTERVAL_MS
    20, // PWM_DEADBAND
    10, // PWM_ERROR_DEADBAND
    2.0 // Kp
};

const size_t CMD_BUF_SIZE = 18;
Command cmdBuf[CMD_BUF_SIZE];

unsigned long lastTelemetry = 0;
// Schedules blocking OLED writes after telemetry.
bool wasTelemetryEmittedOnPreviousLoop = false;

// Leader-only INA228s share the IMU/OLED bus. Pack measures V/I/P/charge
// using the external shunt; midpoint supplies battery A voltage.
// Adapter availability is tracked independently at each address.
Ina228Adapter packPowerMonitor(PACK_POWER_MONITOR_ADDRESS,
    PACK_SHUNT_RESISTANCE_OHMS, PACK_SHUNT_MAX_CURRENT_AMPS, true);
Ina228Adapter midpointPowerMonitor(MIDPOINT_POWER_MONITOR_ADDRESS);

// Current power readings shared by telemetry and the deferred OLED render.
Volts inferredBattBVoltage(NAN);
PowerMonitorMeasurement packMeasurement;
PowerMonitorMeasurement midpointMeasurement;

// --- I2C sensor cluster (Milestone 16) — leader board only ---
// The LSM6DSO IMU rides the leader's telemetry tick; followers never touch the bus.
Lsm6dsoAdapter imuSensor;
static_assert(
    sizeof(ImuCalibrationRecord) == EEPROM_IMU_CAL_SIZE,
    "update EEPROM_IMU_CAL_SIZE in eeprom_layout.h");

// EEPROM binding for ImuCalibrator. Kept out of src/imu/ because it needs
// <EEPROM.h> and that directory compiles on the host.
class EepromImuCalibrationStorage
{
public:
    void load(ImuCalibrationRecord &record)
    {
        EEPROM.get(EEPROM_IMU_CAL_ADDR, record);
    }

    void writeRecord(const ImuCalibrationRecord &record)
    {
        EEPROM.put(EEPROM_IMU_CAL_ADDR, record);
    }

    void updateMagic(uint8_t magic)
    {
        EEPROM.update(EEPROM_IMU_CAL_ADDR, magic);
    }
};

class EepromPowerCalibrationStorage
{
public:
    void load(PowerCalibrationRecord &record)
    {
        EEPROM.get(EEPROM_POWER_CAL_ADDR, record);
    }

    void writeRecord(const PowerCalibrationRecord &record)
    {
        EEPROM.put(EEPROM_POWER_CAL_ADDR, record);
    }

    void updateMagic(uint8_t magic)
    {
        EEPROM.update(EEPROM_POWER_CAL_ADDR, magic);
    }
};

PowerCalibration powerCalibration;

static void logImuInitFailure(Lsm6dsoInitializationResult result)
{
    if (result == Lsm6dsoInitializationResult::NotDetected)
    {
        Serial.println(F("IMU CAL: LSM6DSO not detected at configured addresses; shipping valid=0."));
        return;
    }

    if (result == Lsm6dsoInitializationResult::ConfigurationFailed)
    {
        Serial.println(F("IMU CAL: LSM6DSO detected but register configuration failed; shipping valid=0."));
        return;
    }

    Serial.println(F("IMU CAL: unexpected initialization result; shipping valid=0."));
}

static void logImuCalibrationResult(ImuCalibrationResult result)
{
    switch (result)
    {
        case ImuCalibrationResult::Loaded:
            Serial.println(F("IMU CAL: loaded from EEPROM."));
            break;
        case ImuCalibrationResult::Captured:
            Serial.println(F("IMU CAL: gyro bias captured and saved to EEPROM."));
            break;
        case ImuCalibrationResult::ReadFailed:
            Serial.println(F("IMU CAL: sensor read failed; bias left at zero, not saved."));
            break;
        case ImuCalibrationResult::MotionDetected:
            Serial.println(F("IMU CAL: motion detected; bias left at zero, not saved."));
            break;
        case ImuCalibrationResult::VerificationFailed:
            Serial.println(F("IMU CAL: EEPROM verification failed; bias left at zero."));
            break;
    }
}

static bool readIna228Register(
    uint8_t address, uint8_t reg, uint8_t *bytes, uint8_t byteCount)
{
    Wire.beginTransmission(address);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0 ||
        Wire.requestFrom(address, byteCount) != byteCount)
        return false;

    for (uint8_t i = 0; i < byteCount; ++i)
        bytes[i] = Wire.read();
    return true;
}

static void printPowerMonitorDiagnostics(
    const __FlashStringHelper *label,
    Ina228Adapter &monitor)
{
    static constexpr uint8_t INA228_ADC_CONFIG_REGISTER = 0x01;
    static constexpr uint8_t INA228_BUS_VOLTAGE_REGISTER = 0x05;
    uint8_t adcConfig[2];
    uint8_t vbus[3];
    const bool didReadAdcConfig = readIna228Register(
        monitor.address(), INA228_ADC_CONFIG_REGISTER, adcConfig, sizeof(adcConfig));
    const bool didReadBusVoltage = readIna228Register(
        monitor.address(), INA228_BUS_VOLTAGE_REGISTER, vbus, sizeof(vbus));

    Serial.print(F("INA228 DIAG: "));
    Serial.print(label);
    if (!didReadAdcConfig || !didReadBusVoltage)
    {
        Serial.println(F(" register read failed"));
        return;
    }

    const uint16_t adcRaw =
        (static_cast<uint16_t>(adcConfig[0]) << 8) | adcConfig[1];
    const uint32_t vbusRaw =
        (static_cast<uint32_t>(vbus[0]) << 16) |
        (static_cast<uint32_t>(vbus[1]) << 8) |
        vbus[2];
    const float decodedVoltage =
        static_cast<float>(vbusRaw >> 4) * 195.3125f / 1000000.0f;

    Serial.print(F(" adc=0x"));
    Serial.print(adcRaw, HEX);
    Serial.print(F(" mode=0x"));
    Serial.print(adcRaw >> 12, HEX);
    Serial.print(F(" vbus=0x"));
    Serial.print(vbusRaw, HEX);
    Serial.print(F(" decoded="));
    Serial.print(decodedVoltage, 4);
    Serial.print(F(" library="));
    Serial.println(monitor.readBusVoltage().value(), 4);
}

static void printPowerCalibration()
{
    // F() keeps these literals in flash; this bench-only text would otherwise
    // cost most of a kilobyte of the Mega's 8 KB of SRAM.
    Serial.print(F("POWER CAL: packVoltageOffset=")); Serial.print(powerCalibration.packVoltageOffset().value(), 4);
    Serial.print(F(" midpointVoltageOffset="));      Serial.print(powerCalibration.midpointVoltageOffset().value(), 4);
    Serial.print(F(" packShuntCal="));               Serial.println(powerCalibration.packShuntScale(), 5);
    printPowerMonitorDiagnostics(F("Pack"), packPowerMonitor);
    printPowerMonitorDiagnostics(F("Midpoint"), midpointPowerMonitor);
}

static void printPowerCalibrationUsage()
{
    Serial.println(F("POWER CAL usage (leader bench only):"));
    Serial.println(F("  C PWR_SENSE VOLTAGE <packReferenceVolts> <midpointReferenceVolts>"));
    Serial.println(F("  C PWR_SENSE CURRENT <knownAmps>"));
    Serial.println(F("  C PWR_SENSE SHOW"));
    Serial.println(F("  C PWR_SENSE ?"));
}

// The leading 'C' has been consumed. Bare C calibrates the board's actuators;
// C PWR_SENSE calibrates the leader's power monitors without forwarding.
// Invalid numeric references are rejected before calibration is persisted.
static void handleCalibrationCommand(const String& line)
{
    int idx = 0;
    const int len = line.length();
    String tokenStorage[5];
    const char* tokens[5];
    size_t tokenCount = 0;
    while (tokenCount < 5)
    {
        tokenStorage[tokenCount] = nextTok(line, idx, len);
        if (tokenStorage[tokenCount].length() == 0)
            break;
        tokens[tokenCount] = tokenStorage[tokenCount].c_str();
        ++tokenCount;
    }

    if (isActuatorCalibrationCommand(tokenCount, tokens))
    {
        actuatorManager->startAutoCalibration();
        if (leftSerial) leftSerial->println(CALIBRATION_COMMAND_PREFIX);
        if (rightSerial) rightSerial->println(CALIBRATION_COMMAND_PREFIX);
        return;
    }

    PowerCalibrationCommand command = {
        PowerCalibrationAction::Invalid, 0.0f, 0.0f};
    if (!parsePowerCalibrationCommand(tokenCount, tokens, command))
    {
        Serial.println(F("POWER CAL: invalid command; no write."));
        printPowerCalibrationUsage();
        return;
    }

    if (command.action == PowerCalibrationAction::ShowCalibration)
    {
        printPowerCalibration();
        return;
    }
    if (command.action == PowerCalibrationAction::ShowHelp)
    {
        printPowerCalibrationUsage();
        return;
    }

    // Pack shunt calibration requires only the pack monitor.
    if (!packPowerMonitor.isUp())
    {
        Serial.println(F("POWER CAL: Pack monitor offline; aborting (no write)."));
        return;
    }

    if (command.action == PowerCalibrationAction::CalibrateVoltageOffsets)
    {
        if (!midpointPowerMonitor.isUp())
        {
            Serial.println(F("POWER CAL: Midpoint offline; voltage calibration needs both; aborting (no write)."));
            return;
        }
        EepromPowerCalibrationStorage storage;
        const PowerCalibration::SaveResult result =
            powerCalibration.captureVoltage(
                storage,
                packPowerMonitor.readBusVoltage(),
                midpointPowerMonitor.readBusVoltage(),
                Volts(command.firstReference),
                Volts(command.secondReference));
        if (result == PowerCalibration::SaveResult::InvalidInput)
        {
            Serial.println(F("POWER CAL: invalid reference, reading, or solved offset; aborting (no write)."));
            return;
        }
        if (result == PowerCalibration::SaveResult::VerificationFailed)
        {
            Serial.println(F("POWER CAL: EEPROM verification failed; prior calibration retained."));
            return;
        }
        Serial.println(F("POWER CAL: voltage offsets saved and applied."));
        printPowerCalibration();
        return;
    }

    if (command.action == PowerCalibrationAction::CalibrateCurrentScale)
    {
        // The operator forces a known current through the pack shunt, signed to
        // match the sensor's convention.
        const Amps knownCurrent(command.firstReference);
        const Amps measuredCurrent = packPowerMonitor.readCurrent();
        EepromPowerCalibrationStorage storage;
        const PowerCalibration::SaveResult result =
            powerCalibration.captureCurrent(
                storage, measuredCurrent, knownCurrent);
        if (result == PowerCalibration::SaveResult::InvalidInput)
        {
            Serial.println(F("POWER CAL: invalid current pair or solved shunt trim; aborting (no write)."));
            return;
        }
        if (result == PowerCalibration::SaveResult::VerificationFailed)
        {
            Serial.println(F("POWER CAL: EEPROM verification failed; prior calibration retained."));
            return;
        }
        Serial.println(F("POWER CAL: Pack current calibration saved and applied."));
        printPowerCalibration();
        return;
    }
}

static void powerMonitorSetup()
{
    const bool isPackMonitorUp = packPowerMonitor.begin(&Wire);
    Serial.print(F("POWER MONITOR: Pack (INA228 0x"));
    Serial.print(packPowerMonitor.address(), HEX);
    Serial.println(isPackMonitorUp
        ? F(") online, external shunt calibrated.")
        : F(") init FAILED; BATT fields marked invalid."));

    const bool isMidpointMonitorUp = midpointPowerMonitor.begin(&Wire);
    Serial.print(F("POWER MONITOR: Midpoint (INA228 0x"));
    Serial.print(midpointPowerMonitor.address(), HEX);
    Serial.println(isMidpointMonitorUp ? F(") online.") : F(") init FAILED."));

    EepromPowerCalibrationStorage storage;
    Serial.println(powerCalibration.load(storage)
        ? F("POWER CAL: loaded from EEPROM.")
        : F("POWER CAL: none/invalid; running identity trims."));
}

// Acquire and calibrate both power monitors without writing telemetry.
static void readPowerMeasurements()
{
    packMeasurement = powerCalibration.applyPackCalibration(packPowerMonitor.measure());
    midpointMeasurement = powerCalibration.applyMidpointCalibration(midpointPowerMonitor.measure());
    inferredBattBVoltage = packMeasurement.voltage - midpointMeasurement.voltage;
}

static void imuSetup()
{
    const Lsm6dsoInitializationResult initResult =
        imuSensor.initialize();
    if (initResult != Lsm6dsoInitializationResult::Ok)
    {
        logImuInitFailure(initResult);
        return;
    }

    EepromImuCalibrationStorage storage;
    logImuCalibrationResult(imuSensor.calibrate(storage, delay));

    Serial.println(F("IMU CAL: LSM6DSO online."));
}

// One line = "ROLE; " + ACT_COUNT segments; allow ~55 chars per segment to avoid truncation.
#define TELEMETRY_LINE_MAX (8 + (ACT_COUNT * 55))

static char leftPartial[TELEMETRY_LINE_MAX];
static char rightPartial[TELEMETRY_LINE_MAX];
static size_t leftPartialPos = 0;
static size_t rightPartialPos = 0;

void updateActuatorStatusFromTelemetry(
    const char *line,
    BoardRole boardRole)
{
    ActuatorStatus status[CONTROLLER_ACTUATOR_COUNT];
    if (!parseActuatorStatus(line, boardRole, status))
        return;

    controllerFreshnessTrackers[boardRole] =
        ControllerFreshnessTracker::seenAt(millis());
    for (const ActuatorStatus &actuatorStatus : status)
        latestActuatorStatus[actuatorStatus.actuatorId] = actuatorStatus;
}

// Forward only complete lines (up to and including \n) from follower serial to mainSerial.
void forwardFullLines(
    HardwareSerial* from,
    HardwareSerial* to,
    char* partial,
    size_t cap,
    size_t* partialPos,
    BoardRole boardRole)
{
    if (!from || !to || !partial || !partialPos) return;
    while (from->available())
    {
        char c = (char)from->read();
        if (c == '\n')
        {
            partial[*partialPos] = '\0';
            if (*partialPos > 0)
            {
                to->println(partial);
                updateActuatorStatusFromTelemetry(partial, boardRole);
            }
            *partialPos = 0;
            continue;
        }
        if (c == '\r')
            continue; // skip \r (part of \r\n); don't treat as line end or we'd send empty line on \n
        if (*partialPos < cap - 1)
            partial[(*partialPos)++] = c;
        else
        {
            // TODO: THIS SHOULD THROW SOME KIND OF BAD ERROR CONDITION
            // Buffer full before \n: discard rest of line so we don't forward a partial or get stuck
            while (from->available())
            {
                char d = (char)from->read();
                if (d == '\n' || d == '\r') break;
            }
            *partialPos = 0;
        }
    }
}

void determineRole()
{
    Serial.println("--- SYNC ---");

    // Emit cached role before election so USB probe can label this port correctly
    // even when the board is probed alone (and would otherwise appear as ROLE_UNKNOWN).
    switch (loadRole())
    {
        case ROLE_FRONT: Serial.println("ROLE_HINT: FRONT"); break;
        case ROLE_LEFT:  Serial.println("ROLE_HINT: LEFT");  break;
        case ROLE_RIGHT: Serial.println("ROLE_HINT: RIGHT"); break;
        default: break;
    }

    pinMode(LED_BUILTIN, OUTPUT);
    SERIAL_LEFT.begin(BAUD_RATE);
    SERIAL_RIGHT.begin(BAUD_RATE);

    const bool isI2cHost = hasFrontImu();
    bool hasSyncFromLeft = false, hasSyncFromRight = false;
    bool isLeftAssigned = false, isRightAssigned = false;
    unsigned long start = millis();
    unsigned long lastSync = 0;

    do
    {
        // Everyone sends a SYNC_TOKEN every 10ms to see what serial lines are connected
        if (millis() - lastSync >= 10)
        {
            lastSync = millis();
            SERIAL_LEFT.println(SYNC_TOKEN);
            SERIAL_RIGHT.println(SYNC_TOKEN);
        }
        // If the left serial line is available, we're either the left follower or the leader
        if (SERIAL_LEFT.available())
        {
            String s = SERIAL_LEFT.readStringUntil('\n');
            // If the leader has sent us an ASSIGN_LEFT command, we're the left follower
            if (!isI2cHost &&
                s.indexOf(ASSIGN_LEFT) >= 0)
            {
                currentRole = ROLE_LEFT;
                actuatorManager = new ActuatorManager(ACT_LIST_LEFT, ACT_COUNT);
                mainSerial = &SERIAL_LEFT;
                saveRole(ROLE_LEFT);
                Serial.println("ROLE: LEFT");
                return;
            }
            if (s.indexOf(SYNC_TOKEN) >= 0) hasSyncFromLeft = true;
        }
        if (SERIAL_RIGHT.available())
        {
            String s = SERIAL_RIGHT.readStringUntil('\n');
            // If the leader has sent us an ASSIGN_RIGHT command, we're the right follower
            if (!isI2cHost &&
                s.indexOf(ASSIGN_RIGHT) >= 0)
            {
                currentRole = ROLE_RIGHT;
                actuatorManager = new ActuatorManager(ACT_LIST_RIGHT, ACT_COUNT);
                mainSerial = &SERIAL_RIGHT;
                saveRole(ROLE_RIGHT);
                Serial.println("ROLE: RIGHT");
                return;
            }
            if (s.indexOf(SYNC_TOKEN) >= 0) hasSyncFromRight = true;
        }

        // The front controller assigns discovered followers.
        if (isI2cHost || (hasSyncFromLeft && hasSyncFromRight))
        {
            if (hasSyncFromLeft && !isLeftAssigned)
            {
                SERIAL_LEFT.println(ASSIGN_LEFT);
                isLeftAssigned = true;
            }
            if (hasSyncFromRight && !isRightAssigned)
            {
                SERIAL_RIGHT.println(ASSIGN_RIGHT);
                isRightAssigned = true;
            }
        }
    }
    while (millis() - start < 3000 && !(hasSyncFromLeft && hasSyncFromRight));

    // The I2C host is the front controller.
    if (isI2cHost || (hasSyncFromLeft && hasSyncFromRight))
    {
        currentRole = ROLE_FRONT;
        actuatorManager = new ActuatorManager(ACT_LIST_FRONT, ACT_COUNT);
        mainSerial = &Serial;
        leftSerial = &SERIAL_LEFT;
        rightSerial = &SERIAL_RIGHT;
        saveRole(ROLE_FRONT);
        Serial.println("ROLE: FRONT");
        return;
    }

    // Timeout: no both-sync, default to front actuators but report UNKNOWN.
    currentRole = ROLE_UNKNOWN;
    actuatorManager = new ActuatorManager(ACT_LIST_FRONT, ACT_COUNT);
    mainSerial = &Serial;
    leftSerial = hasSyncFromLeft ? &SERIAL_LEFT : nullptr;
    rightSerial = hasSyncFromRight ? &SERIAL_RIGHT : nullptr;
    Serial.println("ROLE: UNKNOWN (front actuators)");
}

void setup()
{
    Serial.begin(BAUD_RATE);
    determineRole();

    // TODO: This should not need to be done here, it should be done when actuators are instantiated, and we should delay instantiation until after role election is complete.
    LinearActuator** list = (currentRole == ROLE_LEFT) ? ACT_LIST_LEFT : (currentRole == ROLE_RIGHT) ? ACT_LIST_RIGHT : ACT_LIST_FRONT;
    for (size_t i = 0; i < ACT_COUNT; i++)
        list[i]->setControlConfig(ACTUATOR_CONFIG);
    actuatorManager->initAll();
    hallHwInit();
    actuatorManager->loadCalibration();

    if (currentRole == ROLE_FRONT || currentRole == ROLE_UNKNOWN)
    {
        pinMode(STATUS_LED_PIN, OUTPUT);
        digitalWrite(STATUS_LED_PIN, LOW);
        imuSetup();
        powerMonitorSetup();
        if (!oledRenderer.initialize())
            Serial.println(F("OLED: initialization failed at 0x3D."));
    }

    Serial.print("Krabby Ready ");
    Serial.print(boardPinRevisionLabel());
    Serial.print(". ");
    Serial.println(list[0]->getName());
}

// Read lines from a follower serial until one starts with "VER "; discard telemetry lines.
static String readVerLine(HardwareSerial* port, unsigned long timeout_ms)
{
    unsigned long deadline = millis() + timeout_ms;
    String line = "";
    while (millis() < deadline)
    {
        if (!port->available()) continue;
        char c = (char)port->read();
        if (c == '\n')
        {
            if (line.startsWith("VER ")) return line;
            line = "";
            continue;
        }
        if (c != '\r') line += c;
        if (line.length() > 128) line = ""; // guard against runaway
    }
    return "";
}

// Parse a per-board VER reply: "VER <version> <branch> <commit>"
static void parseVerToken(const String& reply, String& ver, String& branch, String& commit)
{
    ver = "-"; branch = "-"; commit = "-";
    if (!reply.startsWith("VER ")) return;
    String rest = reply.substring(4);
    int sp1 = rest.indexOf(' ');
    if (sp1 < 0) { ver = rest; return; }
    ver = rest.substring(0, sp1);
    rest = rest.substring(sp1 + 1);
    int sp2 = rest.indexOf(' ');
    if (sp2 < 0) { branch = rest; return; }
    branch = rest.substring(0, sp2);
    commit = rest.substring(sp2 + 1);
    commit.trim();
}

void loop()
{
    while (mainSerial->available())
    {
        char cmdType = mainSerial->peek();
        if (cmdType == 'T')
        {
            mainSerial->read();
            String payload = mainSerial->readStringUntil('\n');
            size_t cmdCount = parseCommands(payload, cmdBuf, CMD_BUF_SIZE);
            // Keeping it simple, we send all commands to all actuator managers, and let each actuator manager ignore any commands that aren't for them
            actuatorManager->applyCommands(cmdBuf, cmdCount);
            if (leftSerial)  { leftSerial->print("T ");  leftSerial->println(payload); }
            if (rightSerial) { rightSerial->print("T "); rightSerial->println(payload); }
        }
        else if (cmdType == 'B')
        {
            mainSerial->read();
            // Read the whole line first; token-by-token reads could spin forever on a truncated line.
            String payload = mainSerial->readStringUntil('\n');
            int i = 0;
            const int len = payload.length();
            while (true)
            {
                String name = nextTok(payload, i, len);
                String pwm = nextTok(payload, i, len);
                if (name.length() == 0 || pwm.length() == 0)
                    break;
                actuatorManager->handleJog(name, pwm.toInt());
            }
            if (leftSerial)  { leftSerial->print("B ");  leftSerial->println(payload); }
            if (rightSerial) { rightSerial->print("B "); rightSerial->println(payload); }
        }
        else if (cmdType == 'J')
        {
            // Host format is J<name> <pwm> (no space after J). Skip any spaces so a
            // legacy "J <name> <pwm>" forward still parses instead of yielding an
            // empty name / pwm 0 (which left followers dead while FRONT still jogged).
            mainSerial->read();
            while (mainSerial->available() && mainSerial->peek() == ' ')
                mainSerial->read();
            String name = mainSerial->readStringUntil(' ');
            int pwm = mainSerial->readStringUntil('\n').toInt();
            actuatorManager->handleJog(name, pwm);
            // Forward in the same J<name> <pwm> shape the host uses.
            if (leftSerial)  { leftSerial->print("J");  leftSerial->print(name);  leftSerial->print(" ");  leftSerial->println(pwm); }
            if (rightSerial) { rightSerial->print("J"); rightSerial->print(name); rightSerial->print(" "); rightSerial->println(pwm); }
        }
        else if (cmdType == 'C')
        {
            mainSerial->read();
            handleCalibrationCommand(mainSerial->readStringUntil('\n'));
        }
        else if (cmdType == 'H')
        {
            mainSerial->read();
            mainSerial->readStringUntil('\n');
            actuatorManager->holdAll();
            if (leftSerial)  leftSerial->println("H");
            if (rightSerial) rightSerial->println("H");
        }
        else if (cmdType == 'V')
        {
            mainSerial->read();
            mainSerial->readStringUntil('\n');

            if (currentRole == ROLE_LEFT || currentRole == ROLE_RIGHT)
            {
                // Follower: reply with own version on mainSerial (uplink to leader)
                mainSerial->print("VER ");
                mainSerial->print(KRABBY_FW_VERSION);
                mainSerial->print(" ");
                mainSerial->print(KRABBY_FW_BRANCH);
                mainSerial->print(" ");
                mainSerial->println(KRABBY_FW_COMMIT);
            }
            else
            {
                // Leader (FRONT or UNKNOWN): collect follower versions, combine, reply to host
                String lVer = "-", lBranch = "-", lCommit = "-";
                String rVer = "-", rBranch = "-", rCommit = "-";

                if (leftSerial)
                {
                    leftSerial->println("V");
                    String reply = readVerLine(leftSerial, 300);
                    parseVerToken(reply, lVer, lBranch, lCommit);
                }
                if (rightSerial)
                {
                    rightSerial->println("V");
                    String reply = readVerLine(rightSerial, 300);
                    parseVerToken(reply, rVer, rBranch, rCommit);
                }

                mainSerial->print("VER ");
                mainSerial->print(KRABBY_FW_VERSION); mainSerial->print("|"); mainSerial->print(lVer); mainSerial->print("|"); mainSerial->print(rVer);
                mainSerial->print(" ");
                mainSerial->print(KRABBY_FW_BRANCH); mainSerial->print("|"); mainSerial->print(lBranch); mainSerial->print("|"); mainSerial->print(rBranch);
                mainSerial->print(" ");
                mainSerial->print(KRABBY_FW_COMMIT); mainSerial->print("|"); mainSerial->print(lCommit); mainSerial->print("|"); mainSerial->println(rCommit);
            }
        }
        else
        {
            String s = mainSerial->readStringUntil('\n');
            // If leader (or another board) sent SYNC, reply so a restarted leader can discover us
            if (s.indexOf(SYNC_TOKEN) >= 0)
                mainSerial->println(SYNC_TOKEN);
        }
    }

    // Drain follower serial so RX buffers don't overflow (64-byte default drops middle of ~200-byte lines).
    // Only flush once after both drains so we don't block in flush() twice per loop (~35 ms each at 115200).
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos, ROLE_LEFT);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos, ROLE_RIGHT);

    actuatorManager->updateAll();

    if (currentRole == ROLE_FRONT || currentRole == ROLE_UNKNOWN)
    {
        for (LinearActuator *actuator : ACT_LIST_FRONT)
        {
            const ActuatorStatus status = actuator->getStatus();
            latestActuatorStatus[status.actuatorId] = status;
        }

        const uint32_t nowMilliseconds = millis();
        controllerFreshnessTrackers[ROLE_FRONT] =
            ControllerFreshnessTracker::seenAt(nowMilliseconds);

        DisplayFrame displayFrame = buildDisplayFrame(
            currentRole,
            controllerFreshnessTrackers,
            latestActuatorStatus,
            imuMeasurement,
            nowMilliseconds,
            ACTUATOR_CONFIG.pwmDeadband,
            packMeasurement,
            midpointMeasurement,
            inferredBattBVoltage
        );

        const bool isActuatorDisconnected = hasDisconnectedActuator(displayFrame);
        digitalWrite(STATUS_LED_PIN, isActuatorDisconnected ? HIGH : LOW);

        // A full OLED transfer takes ~29 ms; start it after telemetry.
        if (wasTelemetryEmittedOnPreviousLoop &&
            nowMilliseconds - lastOledDrawMilliseconds >= OLED_REDRAW_INTERVAL_MILLISECONDS)
        {
            lastOledDrawMilliseconds = nowMilliseconds;
            oledRenderer.render(displayFrame);
        }
    }

    // Drain again in case bytes arrived during updateAll()
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos, ROLE_LEFT);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos, ROLE_RIGHT);
    mainSerial->flush();

    wasTelemetryEmittedOnPreviousLoop = false;
    const unsigned long telemetryNowMilliseconds = millis();
    if (isTelemetryPollDue(telemetryNowMilliseconds, lastTelemetry))
    {
        wasTelemetryEmittedOnPreviousLoop = true;
        lastTelemetry = telemetryNowMilliseconds;

        mainSerial->print(boardTelemetryRoleLabel(currentRole));
        mainSerial->print(TELEMETRY_SEGMENT_DELIMITER);
        mainSerial->print(TELEMETRY_FIELD_SEPARATOR);
        actuatorManager->printTelemetry(*mainSerial);

        // Leader appends its sensor segments to its own line only; forwarded
        // LEFT/RIGHT lines pass through forwardFullLines() untouched.
        if (currentRole == ROLE_FRONT || currentRole == ROLE_UNKNOWN)
        {
            imuMeasurement = imuSensor.measure();
            appendImuMeasurement(*mainSerial, imuMeasurement);

            packMeasurement = powerCalibration.applyPackCalibration(packPowerMonitor.measure());
            midpointMeasurement = powerCalibration.applyMidpointCalibration(midpointPowerMonitor.measure());
            inferredBattBVoltage = packMeasurement.voltage - midpointMeasurement.voltage;

            // Assume divergence when either battery cannot be read.
            const bool isDiverged =
                !isfinite(midpointMeasurement.voltage.value()) ||
                !isfinite(inferredBattBVoltage.value()) ||
                abs(midpointMeasurement.voltage - inferredBattBVoltage)
                    > BATTERY_DIVERGENCE_THRESHOLD;

            appendBatteryTelemetry(*mainSerial, packMeasurement, midpointMeasurement,
                inferredBattBVoltage, isDiverged, PACK_REGION_NORMAL);
        }
        mainSerial->println();
        mainSerial->flush();  // ensure full line is sent before next loop (avoids two "LEFT;" in one buffer on host)
    }
}

/*
 * Krabby-Uno: 18-Joint Distributed Controller (3 boards × 6 actuators)
 * Front: FL + FR, on USB. Left: RL + ML, on pins 14/15 (Serial3). Right: MR + RR, on pins 16/17 (Serial2).
 * All three boards use the same pinout; the board's role (stored in EEPROM, set
 * with `SET role …`) selects which 6 actuators this board drives.
 */

#include <Arduino.h>
#include <EEPROM.h>
#include <math.h>
#include "src/imu/imu_calibrator.h"
#include "src/imu/lsm6dso_adapter.h"
#include "src/display/ssd1306_adapter.h"
#include "src/display/display_renderer.h"
#include "src/display/display_frame_model.h"
#include "board_pins.h"
#include "command.h"
#include "actuator_manager.h"
#include "src/imu/imu_constants.h"
#include "src/telemetry.h"
#include "version.h"

// --- Serial: left follower = Serial1 (TX1/RX1 on Krabby-Uno v0.1 shield), right follower = Serial2 ---
#define SERIAL_LEFT  Serial1  // pins 18 (TX1), 19 (RX1) — Krabby-Uno v0.1 shield Serial1 connector
#define SERIAL_RIGHT Serial2   // pins 16 (TX2), 17 (RX2) — Krabby-Uno v0.1 shield Serial2 connector
// Exact on the Mega's 16 MHz clock, with headroom for the leader to transmit
// joint data from all three controller boards plus I2C sensor data.
#define BAUD_RATE 250000

BoardRole currentRole = ROLE_UNKNOWN;

ControllerFreshnessTracker controllerFreshnessTrackers[BOARD_ROLE_COUNT];
ActuatorStatus latestActuatorStatus[ActuatorId::ActuatorCount];
ImuMeasurement latestImuMeasurement;
Ssd1306Adapter oledDisplay;
DisplayRenderer<Ssd1306Adapter> oledRenderer(oledDisplay);
unsigned long lastOledDrawMilliseconds = 0;
constexpr unsigned long OLED_REDRAW_INTERVAL_MILLISECONDS = 250;

// EEPROM address 32: magic sentinel byte (0xAB); address 33: BoardRole value.
// Loaded on boot; written only by `SET role …`. Unset → ROLE_UNKNOWN.
// Calibration data (CalData) occupies addresses 0–25; gap at 26–31 kept for alignment.
#define EEPROM_ROLE_ADDR  32
#define EEPROM_ROLE_MAGIC 0xAB

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

// Set by applyRole() from the EEPROM role (on boot and on `SET role …`).
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

// --- I2C sensor cluster — leader board only ---
// The LSM6DSO IMU rides the leader's telemetry tick; followers never touch the bus.
Lsm6dsoAdapter imuSensor;
static_assert(
    sizeof(ImuCalibrationRecord) == EEPROM_IMU_CAL_SIZE,
    "update EEPROM_IMU_CAL_SIZE in src/imu/imu_constants.h");

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

// Apply a role: select this board's 6 actuators and its serial channels, then
// (re)initialize the actuators. Called on boot with the EEPROM role and again
// whenever `SET role …` changes it — no reboot needed.
//   FRONT  : commands/telemetry on USB; forwards to followers on Serial1/Serial2.
//   LEFT   : its uplink Serial1.   RIGHT : its uplink Serial2.
//   UNKNOWN: drives no actuators and sends no telemetry; answers SET/GET (and V)
//            on USB and Serial1/Serial2 so it can be assigned a role.
void applyRole(BoardRole role)
{
    currentRole = role;

    LinearActuator** list = nullptr;
    if (role == ROLE_FRONT)      list = ACT_LIST_FRONT;
    else if (role == ROLE_LEFT)  list = ACT_LIST_LEFT;
    else if (role == ROLE_RIGHT) list = ACT_LIST_RIGHT;

    if (role == ROLE_LEFT)       mainSerial = &SERIAL_LEFT;
    else if (role == ROLE_RIGHT) mainSerial = &SERIAL_RIGHT;
    else                         mainSerial = &Serial;
    leftSerial  = (role == ROLE_FRONT) ? &SERIAL_LEFT  : nullptr;
    rightSerial = (role == ROLE_FRONT) ? &SERIAL_RIGHT : nullptr;

    if (actuatorManager) { delete actuatorManager; actuatorManager = nullptr; }
    if (list)
    {
        for (size_t i = 0; i < ACT_COUNT; i++)
            list[i]->setControlConfig(ACTUATOR_CONFIG);
        actuatorManager = new ActuatorManager(list, ACT_COUNT);
        actuatorManager->initAll();
        actuatorManager->loadCalibration();
    }
}

void setup()
{
    Serial.begin(BAUD_RATE);
    SERIAL_LEFT.begin(BAUD_RATE);
    SERIAL_RIGHT.begin(BAUD_RATE);
    pinMode(LED_BUILTIN, OUTPUT);

    applyRole(loadRole());
    hallHwInit();

    // ROLE_HINT lets `krabby-firmware show` label this port when probed on its own.
    Serial.print("ROLE_HINT: ");
    Serial.println(roleConfigName(currentRole));

    if (currentRole == ROLE_FRONT || currentRole == ROLE_UNKNOWN)
    {
        pinMode(STATUS_LED_PIN, OUTPUT);
        digitalWrite(STATUS_LED_PIN, LOW);
        imuSetup();
        if (!oledRenderer.initialize())
            Serial.println(F("OLED: initialization failed at 0x3D."));
    }

    Serial.print("Krabby Ready ");
    Serial.print(boardPinRevisionLabel());
    Serial.print(". role=");
    Serial.println(roleConfigName(currentRole));
}

// Read lines from a follower serial until one starts with `prefix`; discard telemetry
// and any other lines. Collects a follower's tagged reply ("VER …", "GET …").
static String readPrefixedLine(HardwareSerial* port, const char* prefix, unsigned long timeout_ms)
{
    unsigned long deadline = millis() + timeout_ms;
    String line = "";
    while (millis() < deadline)
    {
        if (!port->available()) continue;
        char c = (char)port->read();
        if (c == '\n')
        {
            if (line.startsWith(prefix)) return line;
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

// SET / GET config commands. The payload is a "key val [key val …]" list, walked
// with the same tokenizer as the T command.
//   SET role <FRONT|LEFT|RIGHT|UNKNOWN> — persist the role and apply it now. No reply.
//   GET <role|version> …               — reply "GET <key> <val> …".
//   SET_LEFT / GET_LEFT, SET_RIGHT / GET_RIGHT — front only: relay the bare command to
//     the follower on Serial1 / Serial2; for GET, re-tag its reply "GET_LEFT …" / "GET_RIGHT …".
// Unknown keys and commands are silently ignored — the SDK validates before sending.
void handleConfig(const String &cmd, const String &payload, HardwareSerial &out)
{
    if (cmd == "SET_LEFT" || cmd == "GET_LEFT" || cmd == "SET_RIGHT" || cmd == "GET_RIGHT")
    {
        bool isLeft = cmd.endsWith("_LEFT");
        HardwareSerial *follower = isLeft ? leftSerial : rightSerial;
        if (!follower) return;  // not the front board
        bool isGet = cmd.startsWith("GET");
        follower->print(isGet ? "GET " : "SET ");
        follower->println(payload);
        if (isGet)
        {
            String reply = readPrefixedLine(follower, "GET ", 300);
            if (reply.length())
            {
                out.print(isLeft ? "GET_LEFT" : "GET_RIGHT");
                out.println(reply.substring(3));  // keep " <key> <val> …"
            }
        }
        return;
    }

    const int len = payload.length();
    int i = 0;
    if (cmd == "SET")
    {
        while (true)
        {
            String key = nextTok(payload, i, len);
            String val = nextTok(payload, i, len);
            if (key.length() == 0 || val.length() == 0) break;
            BoardRole role;
            if (key == "role" && parseRole(val.c_str(), role))
            {
                saveRole(role);
                applyRole(role);
            }
        }
    }
    else if (cmd == "GET")
    {
        out.print("GET");
        while (true)
        {
            String key = nextTok(payload, i, len);
            if (key.length() == 0) break;
            if (key == "role")
            {
                out.print(" role ");
                out.print(roleConfigName(currentRole));
            }
            else if (key == "version")
            {
                // version|branch|commit as one token; unlike V, works on a follower over USB.
                out.print(" version ");
                out.print(KRABBY_FW_VERSION); out.print("|");
                out.print(KRABBY_FW_BRANCH);  out.print("|");
                out.print(KRABBY_FW_COMMIT);
            }
        }
        out.println();
    }
}

// Read one "<CMD> <payload>" config line from `port` and dispatch it.
static void dispatchConfigLine(HardwareSerial &port)
{
    String line = port.readStringUntil('\n');
    int sp = line.indexOf(' ');
    String cmd = (sp < 0) ? line : line.substring(0, sp);
    cmd.trim();
    String payload = (sp < 0) ? String("") : line.substring(sp + 1);
    handleConfig(cmd, payload, port);
}

// SET/GET on a channel other than the board's main one, so a board stays
// configurable over USB (and an UNKNOWN board over Serial1/Serial2). Handles at
// most one line or byte per call; anything that isn't S/G is discarded.
static void processConfig(HardwareSerial &port)
{
    if (!port.available()) return;
    char c = port.peek();
    if (c == 'S' || c == 'G')
        dispatchConfigLine(port);
    else
        port.read();
}

void loop()
{
    while (mainSerial->available())
    {
        char cmdType = mainSerial->peek();
        if (cmdType == 'S' || cmdType == 'G')
        {
            dispatchConfigLine(*mainSerial);
        }
        else if (cmdType == 'T')
        {
            mainSerial->read();
            String payload = mainSerial->readStringUntil('\n');
            size_t cmdCount = parseCommands(payload, cmdBuf, CMD_BUF_SIZE);
            // Keeping it simple, we send all commands to all actuator managers, and let each actuator manager ignore any commands that aren't for them
            if (actuatorManager) actuatorManager->applyCommands(cmdBuf, cmdCount);
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
                if (actuatorManager) actuatorManager->handleJog(name, pwm.toInt());
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
            if (actuatorManager) actuatorManager->handleJog(name, pwm);
            // Forward in the same J<name> <pwm> shape the host uses.
            if (leftSerial)  { leftSerial->print("J");  leftSerial->print(name);  leftSerial->print(" ");  leftSerial->println(pwm); }
            if (rightSerial) { rightSerial->print("J"); rightSerial->print(name); rightSerial->print(" "); rightSerial->println(pwm); }
        }
        else if (cmdType == 'C')
        {
            mainSerial->read();
            mainSerial->readStringUntil('\n');
            if (actuatorManager) actuatorManager->startAutoCalibration();
            if (leftSerial)  leftSerial->println("C");
            if (rightSerial) rightSerial->println("C");
        }
        else if (cmdType == 'H')
        {
            mainSerial->read();
            mainSerial->readStringUntil('\n');
            if (actuatorManager) actuatorManager->holdAll();
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
                    String reply = readPrefixedLine(leftSerial, "VER ", 300);
                    parseVerToken(reply, lVer, lBranch, lCommit);
                }
                if (rightSerial)
                {
                    rightSerial->println("V");
                    String reply = readPrefixedLine(rightSerial, "VER ", 300);
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
            mainSerial->readStringUntil('\n');
        }
    }

    if (mainSerial != &Serial)
        processConfig(Serial);
    if (currentRole == ROLE_UNKNOWN)
    {
        processConfig(SERIAL_LEFT);
        processConfig(SERIAL_RIGHT);
    }

    // Drain follower serial so RX buffers don't overflow (64-byte default drops middle of ~200-byte lines).
    // Only flush once after both drains so we don't block in flush() twice per loop (~35 ms each at 115200).
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos, ROLE_LEFT);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos, ROLE_RIGHT);

    if (actuatorManager) actuatorManager->updateAll();

    if (currentRole == ROLE_FRONT || currentRole == ROLE_UNKNOWN)
    {
        const uint32_t nowMilliseconds = millis();
        // UNKNOWN drives no actuators, so there is no local status to report.
        if (currentRole == ROLE_FRONT)
        {
            for (LinearActuator *actuator : ACT_LIST_FRONT)
            {
                const ActuatorStatus status = actuator->getStatus();
                latestActuatorStatus[status.actuatorId] = status;
            }
            controllerFreshnessTrackers[ROLE_FRONT] =
                ControllerFreshnessTracker::seenAt(nowMilliseconds);
        }

        DisplayFrame displayFrame = buildDisplayFrame(
            currentRole,
            controllerFreshnessTrackers,
            latestActuatorStatus,
            latestImuMeasurement,
            nowMilliseconds,
            ACTUATOR_CONFIG.pwmDeadband
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
    if (actuatorManager && telemetryNowMilliseconds - lastTelemetry >= TELEMETRY_INTERVAL_MS)
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
            const ImuMeasurement measurement = imuSensor.measure();
            latestImuMeasurement = measurement;
            appendImuMeasurement(*mainSerial, measurement);
        }
        mainSerial->println();
        mainSerial->flush();  // ensure full line is sent before next loop (avoids two "LEFT;" in one buffer on host)
    }
}

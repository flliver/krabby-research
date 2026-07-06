/*
 * Krabby-Uno: 18-Joint Distributed Controller (3 boards × 6 actuators)
 * Front: FL + FR, on USB. Left: RL + ML, on pins 14/15 (Serial3). Right: MR + RR, on pins 16/17 (Serial2).
 * All three boards use the same pinout; the board's configured role selects which 6 actuators it drives.
 */

#include <Arduino.h>
#include <EEPROM.h>
#include "board_pins.h"
#include "command.h"
#include "eeprom_layout.h"
#include "actuator_manager.h"
#include "version.h"

// --- Serial: left follower = Serial1 (TX1/RX1 on Krabby-Uno v0.1 shield), right follower = Serial2 ---
#define SERIAL_LEFT  Serial1  // pins 18 (TX1), 19 (RX1) — Krabby-Uno v0.1 shield Serial1 connector
#define SERIAL_RIGHT Serial2   // pins 16 (TX2), 17 (RX2) — Krabby-Uno v0.1 shield Serial2 connector
#define SERIAL_LEFT_RX  19    // RX1 — pulled up so a disconnected uplink idles high, not noise
#define SERIAL_RIGHT_RX 17    // RX2 — same
#define BAUD_RATE 115200

// Max input lines drained from a board's main channel per loop() pass. Bounds the
// drain so a flooded/noisy uplink (e.g. a disconnected follower RX picking up EMI)
// can't starve the config + actuator-update work that runs after the drain loop.
// 16, not 64: garbage that starts with a command letter still costs a blocking
// readStringUntil() (≤50 ms) per iteration, so the budget also caps the worst-case
// pass at ~0.8 s under continuous line noise. Legit traffic is ≤~100 lines/s and
// loop() runs far faster than that, so 16/pass is still ample headroom.
constexpr int RX_DRAIN_BUDGET = 16;

// Persistent board config (role, serial); see eeprom_layout.h for the struct and its
// EEPROM helpers. Loaded on boot, changed at runtime by the SET command, and applied
// to the board with applyRole().
EepromLayout g_config;
BoardRole currentRole = ROLE_UNKNOWN;

static const char* roleName(BoardRole r)
{
    switch (r)
    {
        case ROLE_UNKNOWN: return "UNKWN";
        case ROLE_FRONT:   return "FRONT";
        case ROLE_LEFT:   return "LEFT ";
        case ROLE_RIGHT:  return "RIGHT";
        default:          return "UNKWN";
    }
}

// --- All 18 actuators (names fixed; each board uses the same physical pins for its 6) ---
// Pin numbers from board_pins.h (KRABBY_PIN_REV 1 = legacy, 2 = MOTOR_HEADER_PINOUT).
// Sensor type is fixed per joint by the hardware on that slot, NOT auto-detected:
//   HY (hip-yaw) — ZD gear motor w/ rear incremental encoder, read on the Hall lines
//   HL (hip-lift) — YH8 Hall linear actuator
//   KL (knee)     — potentiometer linear actuator
// One leg therefore has exactly one pot (KL) and two Hall sensors (HY, HL); a leg with
// two pots or two Halls on HL+KL is a wiring/config error, never a calibration outcome.
// Leader/Default Board
LinearActuator flhy("FLHY", PIN_S0_PWMR, PIN_S0_PWML, PIN_S0_EN, A6, A0, 0, SENSOR_HALL);
LinearActuator flhl("FLHL", PIN_S1_PWMR, PIN_S1_PWML, PIN_S1_EN, A7, A1, 1, SENSOR_HALL);
LinearActuator flkl("FLKL", PIN_S2_PWMR, PIN_S2_PWML, PIN_S2_EN, A8, A2, 2, SENSOR_POT);
LinearActuator frhy("FRHY", PIN_S3_PWMR, PIN_S3_PWML, PIN_S3_EN, A9, A3, 3, SENSOR_HALL);
LinearActuator frhl("FRHL", PIN_S4_PWMR, PIN_S4_PWML, PIN_S4_EN, A10, A4, 4, SENSOR_HALL);
LinearActuator frkl("FRKL", PIN_S5_PWMR, PIN_S5_PWML, PIN_S5_EN, A11, A5, 5, SENSOR_POT);
// Left Follower Board
LinearActuator rlhy("RLHY", PIN_S0_PWMR, PIN_S0_PWML, PIN_S0_EN, A6, A0, 0, SENSOR_HALL);
LinearActuator rlhl("RLHL", PIN_S1_PWMR, PIN_S1_PWML, PIN_S1_EN, A7, A1, 1, SENSOR_HALL);
LinearActuator rlkl("RLKL", PIN_S2_PWMR, PIN_S2_PWML, PIN_S2_EN, A8, A2, 2, SENSOR_POT);
LinearActuator mlhy("MLHY", PIN_S3_PWMR, PIN_S3_PWML, PIN_S3_EN, A9, A3, 3, SENSOR_HALL);
LinearActuator mlhl("MLHL", PIN_S4_PWMR, PIN_S4_PWML, PIN_S4_EN, A10, A4, 4, SENSOR_HALL);
LinearActuator mlkl("MLKL", PIN_S5_PWMR, PIN_S5_PWML, PIN_S5_EN, A11, A5, 5, SENSOR_POT);
// Right Follower Board
LinearActuator rrhy("RRHY", PIN_S0_PWMR, PIN_S0_PWML, PIN_S0_EN, A6, A0, 0, SENSOR_HALL);
LinearActuator rrhl("RRHL", PIN_S1_PWMR, PIN_S1_PWML, PIN_S1_EN, A7, A1, 1, SENSOR_HALL);
LinearActuator rrkl("RRKL", PIN_S2_PWMR, PIN_S2_PWML, PIN_S2_EN, A8, A2, 2, SENSOR_POT);
LinearActuator mrhy("MRHY", PIN_S3_PWMR, PIN_S3_PWML, PIN_S3_EN, A9, A3, 3, SENSOR_HALL);
LinearActuator mrhl("MRHL", PIN_S4_PWMR, PIN_S4_PWML, PIN_S4_EN, A10, A4, 4, SENSOR_HALL);
LinearActuator mrkl("MRKL", PIN_S5_PWMR, PIN_S5_PWML, PIN_S5_EN, A11, A5, 5, SENSOR_POT);

// Role → which 6 actuators this board drives (no mutation)
static const size_t ACT_COUNT = 6;
LinearActuator* ACT_LIST_FRONT[]  = { &flhy, &flhl, &flkl, &frhy, &frhl, &frkl };
LinearActuator* ACT_LIST_LEFT[]   = { &rlhy, &rlhl, &rlkl, &mlhy, &mlhl, &mlkl };  // RL + ML
LinearActuator* ACT_LIST_RIGHT[]  = { &rrhy, &rrhl, &rrkl, &mrhy, &mrhl, &mrkl }; // MR + RR

// Set in applyRole() once the role is known.
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

const int TELEMETRY_INTERVAL_MS = 50;
unsigned long lastTelemetry = 0;

// One line = "ROLE; " + ACT_COUNT segments; allow ~55 chars per segment to avoid truncation.
#define TELEMETRY_LINE_MAX (8 + (ACT_COUNT * 55))

static char leftPartial[TELEMETRY_LINE_MAX];
static char rightPartial[TELEMETRY_LINE_MAX];
static size_t leftPartialPos = 0;
static size_t rightPartialPos = 0;

// Forward only complete lines (up to and including \n) from follower serial to mainSerial.
// Drain is BOUNDED per call: on a bench with no followers these RX lines idle on a weak
// pullup, and a brushed motor's EMI bursts punch through it as a continuous garbage-byte
// stream — an unbounded drain here captured loop() (telemetry, jog watchdog, and command
// parsing all dead until motor power was cut; bench 2026-07-03, runaway FLHY). Same
// failure mode as COMMS_DEBUG.md root cause #1, leader side.
static const int FWD_DRAIN_BUDGET = 256;  // bytes per call ≈ one full telemetry line + margin

static bool lineIsPrintable(const char* s, size_t len)
{
    for (size_t i = 0; i < len; i++)
        if (s[i] < 0x20 || s[i] > 0x7E) return false;
    return true;
}

void forwardFullLines(HardwareSerial* from, HardwareSerial* to, char* partial, size_t cap, size_t* partialPos)
{
    if (!from || !to || !partial || !partialPos) return;
    int budget = FWD_DRAIN_BUDGET;
    while (budget-- > 0 && from->available())
    {
        char c = (char)from->read();
        if (c == '\n')
        {
            partial[*partialPos] = '\0';
            // Forward only clean printable-ASCII lines. Motor EMI on these ports
            // arrives as framing garbage (control/high-bit bytes); forwarding it
            // upstream turns noise into blocking TX writes that stall the loop and
            // delay jog-stop processing (bench 2026-07-03: successive FLHY jogs
            // stopped later and later as the junk backlog grew). Real follower
            // lines (telemetry/ERR/VER/config replies) are pure printable ASCII.
            if (*partialPos > 0 && lineIsPrintable(partial, *partialPos))
                to->println(partial);
            *partialPos = 0;
            continue;
        }
        if (c == '\r')
            continue; // skip \r (part of \r\n); don't treat as line end or we'd send empty line on \n
        if (*partialPos < cap - 1)
            partial[(*partialPos)++] = c;
        else
        {
            // Buffer full before \n: discard rest of line (still within budget) so we
            // don't forward a partial or get stuck.
            // TODO (comms-buffer work): surface this overrun on the ERR channel, e.g.
            // emitError("system", "forward_overrun"), once §5's system-token vocabulary is set.
            while (budget-- > 0 && from->available())
            {
                char d = (char)from->read();
                if (d == '\n' || d == '\r') break;
            }
            *partialPos = 0;
        }
    }
}

// Emit one "ERR <token> <code>" line on this board's host channel (Task 1 §5).
// ERR is asynchronous fault telemetry, not a command reply: any command may emit it
// while running, and the firmware never blocks on it. The token scopes the error (a
// joint name, or "system"); the code is a string literal from the §5 vocabulary,
// supplied by the caller at the point the fault is detected. On a follower this goes
// out the UART uplink (mainSerial), and the leader's forwardFullLines() relays it to
// USB unchanged, so the host sees every board's errors on one port. Callers throttle
// to one line per active fault event — the fault state that drives that lives with
// the joint model (Task 2), so the emit sites and their latches land there.
void emitError(const char* token, const char* code)
{
    if (!mainSerial) return;
    mainSerial->print("ERR ");
    mainSerial->print(token);
    mainSerial->print(' ');
    mainSerial->println(code);
}

// Apply a role: select this board's 6 actuators and wire up the serial channels, then
// (re)initialize the actuators. Called on boot with the EEPROM-loaded role and again
// whenever `SET role …` changes it — no reboot needed.
//
// `mainSerial` is each role's main channel — where it takes T/B/J/C/H/V commands and
// emits telemetry:
//   FRONT  : USB; also forwards commands to the two followers and relays their telemetry
//            back to USB (leftSerial = Serial1, rightSerial = Serial2).
//   LEFT   : its uplink Serial1 (the primary drives it and reads its telemetry there).
//   RIGHT  : its uplink Serial2.
//   UNKNOWN: USB, until it is assigned a role.
// Independently of the main channel, every board also answers SET/GET on USB, and an
// UNKNOWN board on Serial1/Serial2, so it can always be configured — see loop().
void applyRole(BoardRole role)
{
    currentRole = role;

    LinearActuator** list = nullptr;
    if (role == ROLE_FRONT)      list = ACT_LIST_FRONT;
    else if (role == ROLE_LEFT)  list = ACT_LIST_LEFT;
    else if (role == ROLE_RIGHT) list = ACT_LIST_RIGHT;
    // ROLE_UNKNOWN: no actuators driven.

    if (role == ROLE_LEFT)       mainSerial = &SERIAL_LEFT;
    else if (role == ROLE_RIGHT) mainSerial = &SERIAL_RIGHT;
    else                         mainSerial = &Serial;   // FRONT / UNKNOWN
    leftSerial  = (role == ROLE_FRONT) ? &SERIAL_LEFT  : nullptr;
    rightSerial = (role == ROLE_FRONT) ? &SERIAL_RIGHT : nullptr;

    if (actuatorManager) { delete actuatorManager; actuatorManager = nullptr; }
    if (list)
    {
        for (size_t i = 0; i < ACT_COUNT; i++)
            list[i]->setControlConfig(ACTUATOR_CONFIG);
        actuatorManager = new ActuatorManager(list, ACT_COUNT);
        actuatorManager->initAll();
        actuatorManager->setErrorOutput(mainSerial);  // ERR <joint> <code> → this board's channel
    }
}

void setup()
{
    Serial.begin(BAUD_RATE);
    SERIAL_LEFT.begin(BAUD_RATE);
    SERIAL_RIGHT.begin(BAUD_RATE);
    // Bound readStringUntil() so a partial/garbled line — e.g. an unconnected follower
    // uplink floating on the bench — can't stall the loop for the 1 s stream default.
    Serial.setTimeout(50);
    SERIAL_LEFT.setTimeout(50);
    SERIAL_RIGHT.setTimeout(50);
    // Pull up the follower-uplink RX pins so a disconnected/dangling cable idles high
    // (UART idle) instead of floating and picking up EMI as a stream of phantom bytes.
    // A driven uplink (the leader's TX) still overrides the weak pull-up. Done after
    // begin() so it isn't reset by USART init.
    pinMode(SERIAL_LEFT_RX, INPUT_PULLUP);
    // Same treatment for RX0 (pin 0): the USB serial chip drives this line when
    // healthy, but it drops off the bus under motor EMI (observed re-enumerating on
    // the bench) and tri-states — leaving RX0 floating next to a 120 W brushed motor.
    // The pull-up makes a dead/absent USB chip read as UART idle (silence) instead of
    // a garbage-byte stream into the command dispatcher.
    pinMode(0, INPUT_PULLUP);
    pinMode(SERIAL_RIGHT_RX, INPUT_PULLUP);
    pinMode(LED_BUILTIN, OUTPUT);

    eepromLoad(g_config);          // invalid/blank EEPROM → g_config.role == ROLE_UNKNOWN
    applyRole(g_config.role);
    hallHwInit();

    // ROLE_HINT lets `krabby-firmware show` label this port even when the board is
    // probed on its own.
    Serial.print("ROLE_HINT: ");
    Serial.println(roleConfigName(currentRole));

    Serial.print("Krabby Ready ");
    Serial.print(boardPinRevisionLabel());
    Serial.print(". role=");
    Serial.println(roleConfigName(currentRole));
}

static String readPrefixedLine(HardwareSerial* port, const char* prefix, unsigned long timeout_ms);  // defined below

// Handle a config command parsed from `port`. The payload is a "key val [key val …]"
// list, walked with the same tokenizer as the T command.
//   SET / GET           — act on this board. SET is fire-and-forget; GET replies "GET …".
//   SET_LEFT / GET_LEFT — primary only: strip the suffix, relay the bare command to the
//   SET_RIGHT/ GET_RIGHT  LEFT/RIGHT follower over Serial1/Serial2; for GET, read the
//                         follower's reply and re-tag it "GET_LEFT …"/"GET_RIGHT …".
// Unknown keys and unknown commands are silently ignored — the SDK validates first.
void handleConfig(const String &cmd, const String &payload, HardwareSerial &out)
{
    if (cmd == "SET_LEFT" || cmd == "GET_LEFT" || cmd == "SET_RIGHT" || cmd == "GET_RIGHT")
    {
        bool isLeft = cmd.endsWith("_LEFT");
        HardwareSerial *follower = isLeft ? leftSerial : rightSerial;
        if (!follower) return;  // not the primary (no follower serial): silently dropped
        bool isGet = cmd.startsWith("GET");
        // Whole-board cal dump (Task 4 §4): forward and let the follower's 6 CAL lines
        // relay up via forwardFullLines — the host attributes them by joint name. No
        // synchronous single-line read (that path expects one "GET …" reply, not 6 CAL).
        if (isGet && payload == "calibration")
        {
            follower->println("GET calibration");
            return;
        }
        follower->print(isGet ? "GET " : "SET ");
        follower->println(payload);
        if (isGet)
        {
            String reply = readPrefixedLine(follower, "GET ", 300);   // "GET <key> <val> …"; skips telemetry
            if (reply.length())
            {
                out.print(isLeft ? "GET_LEFT" : "GET_RIGHT");
                out.println(reply.substring(3));          // drop "GET", keep " <key> <val> …"
            }
        }
        return;
    }

    const int len = payload.length();

    if (cmd == "SET")
    {
        int i = 0;
        bool roleChanged = false;
        while (true)
        {
            String key = nextTok(payload, i, len);
            String val = nextTok(payload, i, len);
            if (key.length() == 0 || val.length() == 0) break;
            if (key == "role")
            {
                BoardRole r;
                if (parseRole(val, r)) { g_config.role = r; roleChanged = true; }
            }
            else if (key == "serial")
            {
                memset(g_config.serial, 0, EEPROM_SERIAL_LEN);
                val.toCharArray(g_config.serial, EEPROM_SERIAL_LEN);
            }
            // unknown keys: silently ignored
        }
        eepromSave(g_config);
        if (roleChanged) applyRole(g_config.role);
        // no reply (fire-and-forget)
    }
    else if (cmd == "GET")
    {
        if (payload == "calibration")   // Task 4 §4: whole-board cal dump, one CAL line/joint
        {
            if (actuatorManager) actuatorManager->printAllCal(out);
            return;
        }
        out.print("GET");
        int i = 0;
        while (true)
        {
            String key = nextTok(payload, i, len);
            if (key.length() == 0) break;
            if (key == "role")
            {
                out.print(" role ");
                out.print(roleConfigName(g_config.role));
            }
            else if (key == "serial")
            {
                out.print(" serial ");
                out.print(g_config.serial[0] ? g_config.serial : "-");
            }
            else if (key == "version")
            {
                // Read-only: the build's version|branch|commit, pipe-joined so the
                // whole triple is one space-free token in the key/value reply. This
                // is the config-path equivalent of the V/VER command, and unlike V
                // it works on a follower over USB (V is only handled on mainSerial).
                out.print(" version ");
                out.print(KRABBY_FW_VERSION); out.print("|");
                out.print(KRABBY_FW_BRANCH);  out.print("|");
                out.print(KRABBY_FW_COMMIT);
            }
            // unknown keys: silently skipped
        }
        out.println();
    }
}

// Read lines from a follower serial until one starts with `prefix`; discard telemetry
// and any other lines. The primary uses it to collect a follower's tagged reply
// (e.g. "VER …" after a forwarded V, or "GET …" after a forwarded GET).
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

// Process only SET/GET (config) commands arriving on a secondary serial — used so a board
// stays configurable on channels other than its main one (see loop()). Non-config bytes
// on these channels are drained and ignored.
void processConfig(HardwareSerial &port)
{
    int rxBudget = RX_DRAIN_BUDGET;
    while (port.available() && rxBudget-- > 0)
    {
        char c = port.peek();
        if (c == 'S' || c == 'G')
        {
            String line = port.readStringUntil('\n');
            int sp = line.indexOf(' ');
            String cmd = (sp < 0) ? line : line.substring(0, sp);
            cmd.trim();
            String payload = (sp < 0) ? String("") : line.substring(sp + 1);
            handleConfig(cmd, payload, port);
        }
        else
        {
            port.readStringUntil('\n');
        }
    }
}

void loop()
{
    hallHwLoopPet();  // storm breaker heartbeat: proves loop() is alive (see hall_hw.cpp)

    int rxBudget = RX_DRAIN_BUDGET;
    while (mainSerial->available() && rxBudget-- > 0)
    {
        char cmdType = mainSerial->peek();
        if (cmdType == 'S' || cmdType == 'G')
        {
            // Multi-letter config command (SET / GET). Read the whole line and split
            // into <command> <payload>. The single-letter T/B/J/C/H/V commands never
            // start with S or G, so they stay on the char-dispatch paths below.
            String line = mainSerial->readStringUntil('\n');
            int sp = line.indexOf(' ');
            String cmd = (sp < 0) ? line : line.substring(0, sp);
            cmd.trim();
            String cfgPayload = (sp < 0) ? String("") : line.substring(sp + 1);
            handleConfig(cmd, cfgPayload, *mainSerial);
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
            while (mainSerial->available() && mainSerial->peek() == ' ')
                mainSerial->read();
            if(leftSerial) leftSerial->print("B ");
            if(rightSerial) rightSerial->print("B ");
            while (true)
            {
                String name = mainSerial->readStringUntil(' ');
                int pwm = mainSerial->readStringUntil(' ').toInt();

                if (actuatorManager) actuatorManager->handleJog(name, pwm);
                if (leftSerial)  { 
                    leftSerial->print(name);
                    leftSerial->print(" ");
                    leftSerial->print(pwm);
                    leftSerial->print(" ");
                }
                if (rightSerial) { 
                    rightSerial->print(name);
                    rightSerial->print(" ");
                    rightSerial->print(pwm);
                    rightSerial->print(" ");
                }
                if(mainSerial->peek() == '\n') { mainSerial->readStringUntil('\n'); break; }
            }
            if (leftSerial)  { leftSerial->println(); }
            if (rightSerial) { rightSerial->println(); }
        }
        else if (cmdType == 'J')
        {
            mainSerial->read();
            String name = mainSerial->readStringUntil(' ');
            int pwm = mainSerial->readStringUntil('\n').toInt();
            if (actuatorManager) actuatorManager->handleJog(name, pwm);
            if (leftSerial)  { leftSerial->print("J ");  leftSerial->print(name);  leftSerial->print(" ");  leftSerial->println(pwm); }
            if (rightSerial) { rightSerial->print("J "); rightSerial->print(name); rightSerial->print(" "); rightSerial->println(pwm); }
        }
        else if (cmdType == 'C')
        {
            // "C" = legacy auto-calibrate; "CALL [noyaw]" = whole-board cal sequence (M17 Task 3).
            String line = mainSerial->readStringUntil('\n');
            line.trim();
            if (line.startsWith("CALL"))
            {
                // CALL [noyaw] [skipval] = cal sequence (+ Task-4 validation unless skipval);
                // CALL valonly = standalone current-sense validation (assumes cal already ran).
                if (line.indexOf("valonly") >= 0) {
                    if (actuatorManager) actuatorManager->validateCurrentSense();
                } else {
                    bool includeYaw = (line.indexOf("noyaw") < 0);
                    bool validate   = (line.indexOf("skipval") < 0);
                    if (actuatorManager) actuatorManager->calibrateAll(includeYaw, validate);
                }
                // Not forwarded: whole-robot cross-board sequencing (one board at a time, so
                // multiple boards don't stall the shared 24V rail at once) is chassis-gated.
            }
            else
            {
                if (actuatorManager) actuatorManager->startAutoCalibration();
                if (leftSerial)  leftSerial->println("C");
                if (rightSerial) rightSerial->println("C");
            }
        }
        else if (cmdType == 'K')   // K <name> [extend|retract|left|right]: per-joint cal (M17 Task 2)
        {
            mainSerial->read();
            String rest = mainSerial->readStringUntil('\n');
            rest.trim();
            // Split "<name> <direction>"; empty direction = full both-ends sweep.
            int sp = rest.indexOf(' ');
            String name = (sp < 0) ? rest : rest.substring(0, sp);
            String dir  = (sp < 0) ? String() : rest.substring(sp + 1);
            dir.trim();
            if (actuatorManager) actuatorManager->calibrateJointByName(name, dir);
            // forward the full line so a follower's joint can be calibrated through the leader
            if (leftSerial)  { leftSerial->print("K"); leftSerial->println(rest); }
            if (rightSerial) { rightSerial->print("K"); rightSerial->println(rest); }
        }
        else if (cmdType == 'Q')   // Q <name>: read back stored calibration (M17 Task 2)
        {
            mainSerial->read();
            String name = mainSerial->readStringUntil('\n');
            name.trim();
            // The owning board prints its "CAL <name> …" reply; a follower's reply is
            // relayed up to USB by forwardFullLines, like telemetry/ERR.
            if (actuatorManager) actuatorManager->queryCalByName(name, *mainSerial);
            if (leftSerial)  { leftSerial->print("Q"); leftSerial->println(name); }
            if (rightSerial) { rightSerial->print("Q"); rightSerial->println(name); }
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
            // Unknown byte: discard it and move on — do NOT line-drain. The SDK
            // validates before sending, so an unknown byte is line noise, not a
            // command: when the USB bridge chip glitches under motor EMI the RX0
            // line floats and delivers continuous garbage, and a readStringUntil()
            // here costs a 50 ms timeout PLUS a heap String allocation per call —
            // 64 of those per pass stalled the loop for seconds and fragmented the
            // 3.5 kB heap toward a hard hang (bench 2026-07-03, runaway FLHY).
            // Single-byte discard is non-blocking and self-resynchronizing.
            mainSerial->read();
        }
    }

    // Config (SET/GET) is also accepted on secondary channels so a board can always be
    // configured: every non-FRONT board answers config over USB (bench reachability), and
    // an UNKNOWN board also listens on Serial1/Serial2 to receive its first forwarded role.
    if (mainSerial != &Serial)
        processConfig(Serial);
    if (currentRole == ROLE_UNKNOWN)
    {
        processConfig(SERIAL_LEFT);
        processConfig(SERIAL_RIGHT);
    }

    // Drain follower serial so RX buffers don't overflow (64-byte default drops middle of ~200-byte lines).
    // Only flush once after both drains so we don't block in flush() twice per loop (~35 ms each at 115200).
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos);

    if (actuatorManager) actuatorManager->updateAll();

    // Drain again in case bytes arrived during updateAll()
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos);
    mainSerial->flush();

    // ROLE_UNKNOWN drives nothing and has no actuators, so it emits no telemetry
    // stream — it still answers V and GET so the operator can identify and assign it.
    if (actuatorManager && millis() - lastTelemetry >= TELEMETRY_INTERVAL_MS)
    {
        lastTelemetry = millis();
        mainSerial->print(roleName(currentRole));
        mainSerial->print("; ");
        actuatorManager->printTelemetry(*mainSerial);
        mainSerial->flush();  // ensure full line is sent before next loop (avoids two "LEFT;" in one buffer on host)
    }
}

/*
 * Krabby-Uno: 18-Joint Distributed Controller (3 boards × 6 actuators)
 * Front: FL + FR, on USB. Left: RL + ML, on pins 14/15 (Serial3). Right: MR + RR, on pins 16/17 (Serial2).
 * All three boards use the same pinout; role election selects which 6 actuators this board drives.
 */

#include <Arduino.h>
#include <EEPROM.h>
#include "board_pins.h"
#include "command.h"
#include "actuator_manager.h"

// --- Serial: left follower = 14/15 (Serial3), right follower = 16/17 (Serial2) ---
#define SERIAL_LEFT  Serial3  // pins 14 (TX3), 15 (RX3)
#define SERIAL_RIGHT Serial2   // pins 16 (TX2), 17 (RX2)
#define BAUD_RATE 115200
#define SYNC_TOKEN "SYNC"
#define ASSIGN_LEFT  "ROLE:LEFT"
#define ASSIGN_RIGHT "ROLE:RIGHT"

enum BoardRole { ROLE_UNKNOWN, ROLE_FRONT, ROLE_LEFT, ROLE_RIGHT };
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

const int TELEMETRY_INTERVAL_MS = 50;
unsigned long lastTelemetry = 0;

// One line = "ROLE; " + ACT_COUNT segments; allow ~55 chars per segment to avoid truncation.
#define TELEMETRY_LINE_MAX (8 + (ACT_COUNT * 55))

static char leftPartial[TELEMETRY_LINE_MAX];
static char rightPartial[TELEMETRY_LINE_MAX];
static size_t leftPartialPos = 0;
static size_t rightPartialPos = 0;

// Forward only complete lines (up to and including \n) from follower serial to mainSerial.
void forwardFullLines(HardwareSerial* from, HardwareSerial* to, char* partial, size_t cap, size_t* partialPos)
{
    if (!from || !to || !partial || !partialPos) return;
    while (from->available())
    {
        char c = (char)from->read();
        if (c == '\n')
        {
            partial[*partialPos] = '\0';
            if (*partialPos > 0)
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
    pinMode(LED_BUILTIN, OUTPUT);
    SERIAL_LEFT.begin(BAUD_RATE);
    SERIAL_RIGHT.begin(BAUD_RATE);

    bool syncFromLeft = false, syncFromRight = false;
    unsigned long start = millis();
    unsigned long lastSync = 0;

    while (millis() - start < 3000)
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
            if (s.indexOf(ASSIGN_LEFT) >= 0)
            {
                currentRole = ROLE_LEFT;
                actuatorManager = new ActuatorManager(ACT_LIST_LEFT, ACT_COUNT);
                mainSerial = &SERIAL_LEFT;
                Serial.println("ROLE: LEFT");
                return;
            }
            if (s.indexOf(SYNC_TOKEN) >= 0) syncFromLeft = true;
        }
        if (SERIAL_RIGHT.available())
        {
            String s = SERIAL_RIGHT.readStringUntil('\n');
            // If the leader has sent us an ASSIGN_RIGHT command, we're the right follower
            if (s.indexOf(ASSIGN_RIGHT) >= 0)
            {
                currentRole = ROLE_RIGHT;
                actuatorManager = new ActuatorManager(ACT_LIST_RIGHT, ACT_COUNT);
                mainSerial = &SERIAL_RIGHT;
                Serial.println("ROLE: RIGHT");
                return;
            }
            if (s.indexOf(SYNC_TOKEN) >= 0) syncFromRight = true;
        }
        // Received SYNC from both sides: we are the leader. Assign followers then set ourselves as FRONT.
        if (syncFromLeft && syncFromRight)
        {
            SERIAL_LEFT.println(ASSIGN_LEFT);
            SERIAL_RIGHT.println(ASSIGN_RIGHT);
            currentRole = ROLE_FRONT;
            actuatorManager = new ActuatorManager(ACT_LIST_FRONT, ACT_COUNT);
            mainSerial = &Serial;
            leftSerial = &SERIAL_LEFT;
            rightSerial = &SERIAL_RIGHT;
            Serial.println("ROLE: FRONT");
            return;
        }
    }

    // Timeout: no both-sync, default to front actuators but report UNKNOWN.
    currentRole = ROLE_UNKNOWN;
    actuatorManager = new ActuatorManager(ACT_LIST_FRONT, ACT_COUNT);
    mainSerial = &Serial;
    leftSerial = &SERIAL_LEFT;
    rightSerial = &SERIAL_RIGHT;
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

    Serial.print("Krabby Ready ");
    Serial.print(boardPinRevisionLabel());
    Serial.print(". ");
    Serial.println(list[0]->name);
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
            if(leftSerial) leftSerial->print("B ");
            if(rightSerial) rightSerial->print("B ");
            while (true)
            {
                String name = mainSerial->readStringUntil(' ');
                int pwm = mainSerial->readStringUntil(' ').toInt();

                actuatorManager->handleJog(name, pwm);
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
            actuatorManager->handleJog(name, pwm);
            if (leftSerial)  { leftSerial->print("J ");  leftSerial->print(name);  leftSerial->print(" ");  leftSerial->println(pwm); }
            if (rightSerial) { rightSerial->print("J "); rightSerial->print(name); rightSerial->print(" "); rightSerial->println(pwm); }
        }
        else if (cmdType == 'C')
        {
            mainSerial->read();
            mainSerial->readStringUntil('\n');
            actuatorManager->startAutoCalibration();
            if (leftSerial)  leftSerial->println("C");
            if (rightSerial) rightSerial->println("C");
        }
        else if (cmdType == 'H')
        {
            mainSerial->read();
            mainSerial->readStringUntil('\n');
            actuatorManager->holdAll();
            if (leftSerial)  leftSerial->println("H");
            if (rightSerial) rightSerial->println("H");
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
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos);

    actuatorManager->updateAll();

    // Drain again in case bytes arrived during updateAll()
    forwardFullLines(leftSerial, mainSerial, leftPartial, TELEMETRY_LINE_MAX, &leftPartialPos);
    forwardFullLines(rightSerial, mainSerial, rightPartial, TELEMETRY_LINE_MAX, &rightPartialPos);
    mainSerial->flush();

    if (millis() - lastTelemetry >= TELEMETRY_INTERVAL_MS)
    {
        lastTelemetry = millis();
        mainSerial->print(roleName(currentRole));
        mainSerial->print("; ");
        actuatorManager->printTelemetry(*mainSerial);
        mainSerial->flush();  // ensure full line is sent before next loop (avoids two "LEFT;" in one buffer on host)
    }
}

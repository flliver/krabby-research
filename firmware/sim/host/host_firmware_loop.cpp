// Host equivalent of arduino.ino setup/loop: command dispatch, updateAll, 50 ms telemetry.

#include "host_api.h"
#include "host_firmware_internal.hpp"

#include "Arduino.h"
#include "command.h"
#include "hall_hw.h"
#include "src/controller/board_role.h"
#include "src/telemetry.h"
#include "version.h"

namespace {

constexpr size_t kCmdBufSize = 18;
Command s_cmdBuf[kCmdBufSize];
unsigned long s_lastTelemetryMs = 0;
bool s_firmwareSetupDone = false;

void drainUsbCommands()
{
    using krabby::sim::host::actuatorManager;

    while (Serial.available())
    {
        const char cmdType = static_cast<char>(Serial.peek());
        if (cmdType == 'T')
        {
            Serial.read();
            const String payload = Serial.readStringUntil('\n');
            const size_t cmdCount = parseCommands(payload, s_cmdBuf, kCmdBufSize);
            actuatorManager().applyCommands(s_cmdBuf, cmdCount);
        }
        else if (cmdType == 'B')
        {
            Serial.read();
            const String payload = Serial.readStringUntil('\n');
            int i = 0;
            const int len = payload.length();
            while (true)
            {
                const String name = nextTok(payload, i, len);
                const String pwm = nextTok(payload, i, len);
                if (name.length() == 0 || pwm.length() == 0) break;
                actuatorManager().handleJog(name, pwm.toInt());
            }
        }
        else if (cmdType == 'J')
        {
            Serial.read();
            while (Serial.available() && Serial.peek() == ' ')
            {
                Serial.read();
            }
            const String name = Serial.readStringUntil(' ');
            const int pwm = Serial.readStringUntil('\n').toInt();
            actuatorManager().handleJog(name, pwm);
        }
        else if (cmdType == 'C')
        {
            Serial.read();
            Serial.readStringUntil('\n');
            actuatorManager().startAutoCalibration();
        }
        else if (cmdType == 'H')
        {
            Serial.read();
            Serial.readStringUntil('\n');
            actuatorManager().holdAll();
        }
        else if (cmdType == 'V')
        {
            Serial.read();
            Serial.readStringUntil('\n');
            Serial.print("VER ");
            Serial.print(KRABBY_FW_VERSION);
            Serial.print(" ");
            Serial.print(KRABBY_FW_BRANCH);
            Serial.print(" ");
            Serial.println(KRABBY_FW_COMMIT);
        }
        else
        {
            Serial.readStringUntil('\n');
        }
    }
}

void emitTelemetryIfDue()
{
    using krabby::sim::host::actuatorManager;

    const unsigned long nowMs = millis();
    if (nowMs - s_lastTelemetryMs < TELEMETRY_INTERVAL_MS) return;

    s_lastTelemetryMs = nowMs;
    Serial.print(boardTelemetryRoleLabel(ROLE_FRONT));
    Serial.print(TELEMETRY_SEGMENT_DELIMITER);
    Serial.print(TELEMETRY_FIELD_SEPARATOR);
    actuatorManager().printTelemetry(Serial);
    Serial.println();
}

}  // namespace

extern "C" void krabby_host_firmware_setup(void)
{
    if (s_firmwareSetupDone) return;
    hallHwInit();
    krabby::sim::host::actuatorManager().loadCalibration();
    s_lastTelemetryMs = millis();
    s_firmwareSetupDone = true;
}

extern "C" void krabby_host_firmware_loop_once(void)
{
    if (!s_firmwareSetupDone) krabby_host_firmware_setup();
    drainUsbCommands();
    krabby::sim::host::actuatorManager().updateAll();
    emitTelemetryIfDue();
}

extern "C" void krabby_host_firmware_reset_loop_state(void)
{
    s_lastTelemetryMs = 0;
    s_firmwareSetupDone = false;
}

// Host equivalent of arduino.ino setup/loop: command dispatch, updateAll, 50 ms telemetry.

#include "host_api.h"
#include "host_firmware_internal.hpp"
#include "host_battery_telemetry.hpp"
#include "host_calibration_command.hpp"
#include "host_display.hpp"
#include "host_leader_imu.hpp"
#include "host_multi_board.hpp"

#include "Arduino.h"
#include "command.h"
#include "hall_hw.h"
#include "src/actuator/calibration_command_dispatch.h"
#include "src/actuator/robot_calibration.h"
#include "src/controller/board_role.h"
#include "src/telemetry.h"
#include "version.h"

namespace {

constexpr size_t kCmdBufSize = 18;
Command s_cmdBuf[kCmdBufSize];
unsigned long s_lastTelemetryMs = 0;
bool s_firmwareSetupDone = false;
RobotCalibration s_robotCalibration;

// Single-board host models one leader with no follower links, labelled with the
// role its calibration output was configured for (as arduino.ino uses currentRole).
class HostRobotCalibrationHooks final : public RobotCalibrationHooks
{
public:
    uint32_t nowMs() override { return millis(); }
    CalibrationTextSink &out() override { return usb_; }
    const char *localRoleLabel() override
    {
        return boardRoleLabel(krabby::sim::host::actuatorManager().calibrationOutputRole());
    }
    CalibrationTextSink *followerLink(BoardRole) override { return nullptr; }
    bool startLocalPass(uint8_t pass, int pwm) override
    {
        return krabby::sim::host::actuatorManager().startCalibration(pass, pwm);
    }
    bool isLocalRunning() override { return krabby::sim::host::actuatorManager().isCalibrating(); }
    LegCalibrationOutcome localOutcome() override
    {
        return krabby::sim::host::actuatorManager().calibrationOutcome();
    }
    void printLocalFailure(CalibrationTextSink &out) override
    {
        krabby::sim::host::actuatorManager().printCalibrationFailure(out);
    }
    void abortLocal(char cause) override
    {
        krabby::sim::host::actuatorManager().abortCalibration(cause);
    }
    void stopLocalMotors() override { krabby::sim::host::actuatorManager().holdAll(); }

private:
    PrintCalibrationSink usb_{&Serial};
};

HostRobotCalibrationHooks s_robotCalibrationHooks;

// H, J, B and T end any calibration before they run (the sketch's glue).
void abortCalibrationForCommand(char command)
{
    abortBoardCalibrationForCommand(
        s_robotCalibration, s_robotCalibrationHooks, krabby::sim::host::actuatorManager(), command);
}

// C PWR_SENSE is a no-op here (no power monitors); everything else goes through
// the sketch's dispatch as the configured leader role.
void handleCalibrationCommand(const String &line)
{
    ActuatorCalibrationRequest request = {CAL_PASS_ALL, CAL_DEFAULT_PWM};
    const ActuatorCalibrationCommandKind kind =
        krabby::sim::host::parseHostCalibrationLine(line, request);
    PrintCalibrationSink out(&Serial);
    dispatchActuatorCalibrationCommand(
        krabby::sim::host::actuatorManager().calibrationOutputRole(), kind, request, out,
        s_robotCalibration, s_robotCalibrationHooks, krabby::sim::host::actuatorManager());
}

void drainUsbCommands()
{
    using krabby::sim::host::actuatorManager;

    while (Serial.available())
    {
        const char cmdType = static_cast<char>(Serial.peek());
        if (cmdType == 'T')
        {
            Serial.read();
            abortCalibrationForCommand('T');
            const String payload = Serial.readStringUntil('\n');
            const size_t cmdCount = parseCommands(payload, s_cmdBuf, kCmdBufSize);
            actuatorManager().applyCommands(s_cmdBuf, cmdCount);
        }
        else if (cmdType == 'B')
        {
            Serial.read();
            abortCalibrationForCommand('B');
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
            abortCalibrationForCommand('J');
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
            handleCalibrationCommand(Serial.readStringUntil('\n'));
        }
        else if (cmdType == 'H')
        {
            Serial.read();
            abortCalibrationForCommand('H');
            Serial.readStringUntil('\n');
            actuatorManager().holdAll();
        }
        else if (cmdType == 'V')
        {
            Serial.read();
            Serial.readStringUntil('\n');
            // Host USB path models FRONT leader: combined VER with |- placeholders for absent followers.
            Serial.print("VER ");
            Serial.print(KRABBY_FW_VERSION);
            Serial.print("|-|");
            Serial.print("-");
            Serial.print(" ");
            Serial.print(KRABBY_FW_BRANCH);
            Serial.print("|-|");
            Serial.print("-");
            Serial.print(" ");
            Serial.print(KRABBY_FW_COMMIT);
            Serial.print("|-|");
            Serial.println("-");
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
    const ImuMeasurement measurement = krabby::sim::host::leaderImu().measure();
    appendImuMeasurement(Serial, measurement);
    if (krabby::sim::host::leaderBattery().isOnline())
    {
        krabby::sim::host::appendBatteryTelemetry(
            Serial, krabby::sim::host::leaderBattery().sample());
    }
    Serial.println();
}

}  // namespace

extern "C" void krabby_host_firmware_setup(void)
{
    if (s_firmwareSetupDone) return;
    hallHwInit();
    // Sensor probes are best-effort: missing I2C devices must not block setup or the loop
    // (IMU still appends with valid=0; BATT/OLED are skipped when offline).
    krabby::sim::host::leaderImu().setup();
    krabby::sim::host::leaderBattery().setup();
    krabby::sim::host::hostDisplaySetup();
    krabby::sim::host::actuatorManager().setCalibrationOutput(&Serial, ROLE_FRONT);
    krabby::sim::host::actuatorManager().loadCalibration();
    s_lastTelemetryMs = millis();
    s_firmwareSetupDone = true;
}

extern "C" void krabby_host_firmware_loop_once(void)
{
    krabby::sim::host::hostDisplayStep(millis());
    if (krabby::sim::host::singleLeaderEighteenJointEnabled())
    {
        krabby::sim::host::singleLeaderEighteenFirmwareLoopOnce();
        return;
    }
    if (krabby::sim::host::threeBoardFirmwareLoopEnabled())
    {
        krabby::sim::host::threeBoardFirmwareLoopOnce();
        return;
    }
    if (!s_firmwareSetupDone) krabby_host_firmware_setup();
    drainUsbCommands();
    krabby::sim::host::actuatorManager().updateAll();
    s_robotCalibration.update(s_robotCalibrationHooks);
    emitTelemetryIfDue();
}

extern "C" void krabby_host_firmware_reset_loop_state(void)
{
    s_lastTelemetryMs = 0;
    s_firmwareSetupDone = false;
    s_robotCalibration.reset();
    krabby::sim::host::threeBoardFirmwareResetLoopState();
}

extern "C" int krabby_host_firmware_calibration_active(void)
{
    return s_robotCalibration.isActive() ||
                   krabby::sim::host::actuatorManager().isCalibrating() ||
                   krabby::sim::host::threeBoardCalibrationActive()
        ? 1
        : 0;
}

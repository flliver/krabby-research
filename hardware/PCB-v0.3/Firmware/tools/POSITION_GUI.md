# Krabby control console v3 ? PC closed-loop position

Requires firmware configuration schema 4. Motor polarity is now stored on the board and applies to every control interface; the PC loop sends logical commands without local reversal.

The Position tab shows board settings as plain read-only text in each motor's Board configuration box: travel enforcement enabled/disabled, native or normalized units, lower/upper thresholds, output reversal, and current boundary status. These readouts follow periodic board configuration/status samples, including during active control. They are not editable and provide no configuration-write buttons. Configuration readouts reflect sampled RAM, not staged edits or necessarily saved flash.

Edit these settings only under Configuration > Motor 1/2/3. Write staged page applies the motor settings to RAM while disarmed; SAVE persists them. Native limits use filtered potentiometer or encoder counts; normalized limits use calibrated 0..65535 units independently of report mode. The target sliders remain normalized requests and may request positions beyond the permitted travel; firmware enforces the active limits.

From Firmware/tools:

```powershell
python .\krabby_position_gui.py --port COM7 --node 7
# Or direct USB, with COM3 released by other terminals/tools:
python .\krabby_position_gui.py --usb --port COM3
```

## Operation

1. Verify power/current limits, selected position sensor, calibration endpoints and
   command-to-position direction using the existing Jog controls. A wrong sign makes
   position feedback positive rather than corrective; it must be corrected before
   operating closed-loop. Set output_reversed on the relevant motor Configuration page and
   write that page to RAM; this polarity applies to every command source.
2. Connect and open Position control. All maximum-drive boxes start at 0, so channels
   are initially disabled. Position sliders range 0..65535. Readouts below each motor
   show live normalized position, measured current (nominal mA), and last drive sent.
3. Set maximum drive limits 0..127, initially conservatively. They limit PWM/duty,
   not physical velocity. Press Apply speed/tuning or Enter after typing limits;
   spinbox arrows apply changes immediately. Zero disables that motor.
4. Enable the toggle. The PC reads configuration and feedback, ARMs, and starts
   heartbeat/control transactions. It initializes targets to measured positions;
   previously staged slider positions are not driven on enable. Each channel stays
   neutral until its target slider is moved after enable.
5. Move sliders to command position. Speed/tuning edits apply with Apply/Enter;
   moving a slider also applies the currently entered limits/tuning.
6. Disable, STOP or Escape sends ZERO then DISARM. Leaving this tab, losing window
   focus, minimizing, disconnecting or closing cancels the mode. Faults require an
   explicit new Enable; there is no automatic rearm.

## Loop and units

The controller is a proportional position loop in Python, not firmware PID and not
trajectory generation. Normalized position is calculated from the selected native
pot/encoder measurement and the RAM calibration endpoints:

`position = clamp((native - calibration_min)*65535/(calibration_max-calibration_min), 0, 65535)`

This is independent of the firmware report_mode, so configuration is not changed.
Calibration must have increasing endpoints. For active potentiometer channels, raw
ADC rails 0/4095 are treated as invalid feedback and stop the mode; inactive
(maximum-drive=0) channels can still display these native endpoints.

Default gain is 400 drive units per full normalized travel. Default deadband is
200 normalized units (about 0.3% of travel). Outside the deadband, command magnitude
is `min(maximum_drive, max(1, int(abs(error)*gain/65535)))`, with direction determined
by error and the Reverse checkbox. Inside the band it sends zero. No integral or
velocity estimator is implemented. Gain/deadband need supervised tuning; loaded
tracking accuracy, stability and disturbance rejection are not qualified.

The loop reads all three sensor pages before each control snapshot. Default cycle
is 20 ms, capped at watchdog/4 using the existing priority scheduler. Windows/USB
are not real-time; serial errors or excessive command gaps stop/disarm rather than
blindly keep old commands alive. Very short watchdogs may not be sustainable with
three feedback queries per cycle. The tool never changes or disables the watchdog.
Use a nonzero board watchdog; link loss can prevent host ZERO/DISARM delivery.

Zero drive obeys the existing configured brake/Hi-Z policy. That policy and the
load determine physical holding/coasting behavior. This tool does not certify
mechanical calibration or impose new firmware arming prerequisites.

## Verification

Latest schema-4 acceptance and firmware artifacts: [motor-page acceptance](../bench/motor_pages/README.md). Previous schema-3 evidence: [travel-limit acceptance](../bench/travel_limits/README.md). The original acceptance below is historical and predates firmware-owned polarity.

Synthetic tests covered native pot/encoder normalization, sign/deadband/saturation,
zero speed, reverse sign, neutral enable, two-motor simulated-plant convergence,
disable, cancellation while ARM is pending, railed-feedback inhibition and GUI
readouts. These are simulation tests, not physical motor movement.

Physical CAN and USB tests on node 7 ran the loop with all speed limits ZERO,
including changing targets and disabling. Live position/current snapshots and
watchdog-paced neutral commands worked; RAM configuration was unchanged. Hidden
real-GUI tests covered enable, target initialization, sliders, readouts, disable
and close. The board ended disarmed. No firmware or flash writes were performed.
No nonzero energized closed-loop tracking was tested.

Evidence: ../bench/application/position_gui_test_results.json and
../bench/application/position_gui_ui_test_results.json.

```powershell
python .\test_position_gui.py
# Optional physical zero-limit checks require an already disarmed Hi-Z fixture:
python .\test_position_gui.py --hardware COM7 --node 7
python .\test_position_gui.py --hardware COM3 --usb --node 7
```

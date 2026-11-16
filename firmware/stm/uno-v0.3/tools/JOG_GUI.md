# Krabby control console v2 ? hold-to-run Jog

The previous Python tools remain unchanged. This version extends/imports the
existing writable console and adds a Jog tab; all previous tabs remain available.

From Firmware/tools:

```powershell
python .\krabby_jog_gui.py --port COM7 --node 7
# Or USB, after closing any other COM3 terminal/viewer:
python .\krabby_jog_gui.py --usb --port COM3
```

Use the actual physical CAN node address (the current bench board reported 7).
Connect, then select Jog. Entering Jog stops any existing motor-tab repetition
and requests DISARM, so an old latched command is not carried into hold-to-run mode.

## Operation

- Each motor has Negative and Positive buttons and its own editable drive field.
- Drive accepts -127..127. The button determines direction; the field's absolute
  value determines magnitude. Default magnitude 20 is about 15.7% PWM; 127 is full
  commanded duty. Nonzero drive is enabled; there is no hidden zero/test inhibit.
- Hold a direction button: the worker stops previous commands, ARMs, waits for its
  acknowledgment, checks that the button is STILL held, and sends fresh motor
  snapshots. Only that motor receives nonzero duty; the other two receive zero.
- Only one jog button is accepted at once. Release before choosing another.
- Magnitude is latched on press. Release and press again after editing it.
- Release sends a three-motor ZERO snapshot, then DISARM. Repetition stops and a
  fresh press is required to arm again. A short click released during ARM cannot
  turn into a delayed nonzero command.

The repeat interval is 20 ms or watchdog/4, whichever is shorter, inherited from
the existing priority scheduler. Watchdog settings are not changed. Windows/USB
are best effort; missed deadlines/faults stop jogging without automatic rearming.
Use a nonzero board watchdog when exercising motors.

Release outside the button is handled using mouse capture and a global release
binding. Windows also polls left-button state as a missed-release fallback.
Window focus loss, minimization, leaving the Jog tab, Escape, STOP, disconnect and
window close cancel the hold. The worker serializes all commands; an in-flight
transaction cannot be recalled. Release is serviced as soon as that transaction
finishes. Loss of the link may prevent ZERO/DISARM delivery, so the board watchdog
is still required. This is a commissioning GUI, not a hardware emergency stop.

ZERO and DISARM follow the firmware's configured brake or Hi-Z safe policy;
mechanical stopping also depends on the motor/load. Power, fuses, DAC limits,
feedback/travel settings and the actual driver prerequisites still apply.

## Verification

`python .\test_jog_gui.py` tests all six signed routes, periodic hold behavior,
release, no restart, release before/during ARM, ZERO-before-DISARM packet order,
close while held, ARM rejection and Jog widgets using a synthetic target.

Actual CAN and USB tests exercised all six hold/release routes with drive ZERO on
node 7, with the board already configured for Hi-Z. Configuration was unchanged;
the board finished disarmed with all three commands zero. No nonzero energized
movement or motor/load behavior was tested. No firmware or flash changes were made.

Evidence: ../bench/application/jog_gui_test_results.json.
Optional physical zero-only tests:

```powershell
python .\test_jog_gui.py --hardware COM7 --node 7
python .\test_jog_gui.py --hardware COM3 --usb --node 7
```

Physical tests require an already disarmed board and all three existing safe states
set to Hi-Z. Existing read-only and writable console source files are fingerprinted
alongside the new source in ../bench/application/jog_gui_sha256.json.

## Firmware polarity and travel units (schema 4)

Positive commands mean increasing selected feedback for every interface. Each motor Configuration page now includes field 12 `output_reversed` (0/1) and field 13 `normalized_limits` (0 native, 1 normalized). No node masks. Limits, enable, polarity and units are on the same motor page and can be written atomically with Write staged page. SAVE persists RAM. Reporting mode does not select limit units. Existing schema-2/3 saved configuration and imported JSON migrate without losing the older settings.

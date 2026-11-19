# Krabby writable control console

2026-11-12. Separate from the unchanged read-only `krabby_debug_gui.py`.
Requires the existing Python tkinter + pyserial setup. No firmware change.

From Firmware/tools:

```powershell
# Current physical board reported node 7 during testing:
python .\krabby_control_gui.py --port COM7 --node 7

# Direct USB; close any COM3 terminal/viewer first:
python .\krabby_control_gui.py --usb --port COM3
```

Select the actual CAN switch address for another board. USB connects directly to
its attached node; the status display shows its physical address. One tool owns
each port. Interface/address/poll settings are locked until disconnect.

## Motor exercise

1. Establish motor power, fuses, reference voltages, load and safe-output prerequisites
   on the physical fixture before applying drive. The tool does not prove these.
2. Connect; inspect configuration and status. Repeat starts OFF and never auto-arms.
3. `ARM (neutral only)` grants permission and clears all motor values. For continuous
   exercise, use `ARM + repeat ZERO`: it explicitly arms, sends a fresh zero snapshot,
   then starts the heartbeat. This is useful when watchdog timing is too short for
   two separate mouse clicks.
4. Enter M1/M2/M3 signed duty, -127..127, and press `Apply / send three motor values`.
   Values are latched as one snapshot; editing a spinbox alone does not change drive.
   With Repeat ON the latched snapshot repeats. ZERO sends zero values without
   deliberately disarming; zero while awake follows configured brake/Hi-Z policy.
5. Turning Repeat OFF or pressing STOP sends priority DISARM and clears queued work
   and command values. Config editing, SAVE/LOAD/defaults, raw/legacy commands and
   reboot stop repeat and disarm before performing the operation.

Requested repeat interval defaults to 20 ms (50 Hz). The worker caps it at one
quarter of configured watchdog time. For example 50 ms watchdog requests <=12.5 ms,
10 ms requests <=2.5 ms. Watchdog=0 uses the requested interval and visibly warns
that the board command watchdog is disabled. This tool never changes the watchdog
just to keep streaming alive.

Motor deadlines take precedence over monitoring, with one serial request at a time.
Windows/USB are not real-time: short watchdog settings may be unsustainable. A
large gap, command failure or observed disarm stops repeat; explicit ARM and fresh
values are required. No automatic rearm or retry of mutating requests. The display
shows maximum observed motor TX gap, not a guaranteed maximum. A blocked I/O call
can delay host DISARM; the board watchdog remains the independent protection.
Disconnect/window close make a best-effort DISARM, then release the port. They cannot
guarantee delivery after link loss, especially with watchdog disabled.

Queued motor jobs carry the console ARM generation. Commands submitted before an
ARM acknowledgment cannot become drive after that ARM. Raw ARM does not grant the
console's dedicated motor gate; use the normal ARM controls before motor exercise.

## Configuration

Select a page/row and double-click or Stage selected field. `*` denotes a local edit:
not active RAM, not saved flash. Field ranges/units/defaults come from the schema.

- Write selected field updates one parameter in RAM.
- Write staged page merges edits with a fresh read and sends the whole page (48-byte node or 56-byte motor),
  permitting related endpoints to change together under firmware validation.
- Defaults selected/all and LOAD change active RAM; they stop/disarm first.
- SAVE writes current active RAM to flash. It does NOT apply staged edits first.
- Export writes the last read active RAM dataset, excluding local staged changes.
- Import validates schema/shape/int32 values and stages all four pages. Write each
  page, then SAVE if desired. Firmware enforces parameter/cross-field constraints.
- Fresh connections discard staged edits; import/stage after connecting to a target.

An acknowledged RAM change is not calibration approval. Hardware-action errors may
leave RAM changed while reporting failure; refresh/inspect rather than blindly retry.
One-page flash persistence retains its accepted interruption/config-loss risk.

## Other features

- Live sensors/config/status are retained from the diagnostic viewer.
- Management: version/capabilities; individual REBOOT_ARM and REBOOT_NOW (500 ms),
  or combined two-stage reboot. Connection ends after a successful reboot.
- Set node telemetry_ms through configuration to exercise periodic transmission.
  A CAN receiver taps telemetry alongside replies and displays all five latest topics.
  USB connections can configure telemetry but cannot receive the CAN frames.
- Raw application packets support protocol/negative testing and show hex responses.
  They stop/disarm first. Use dedicated motor controls for positive drive sequences.
- Legacy USB line entry exposes hello/status/can status/LED/tx/txfd/etc with a USB
  connection. Text commands also stop/disarm first; do not use them as a heartbeat.
- Firmware update tab validates SREC/BIN, disarms/releases the console connection,
  then runs the existing OpenBLT uploader on the specified CANable port. While
  connected it uses the observed board address; while disconnected select the CAN
  node in the header. Config is retained by the existing loader. Reconnect manually
  after completion. Closing the window waits for an ongoing upload rather than
  terminating it. Do not share the adapter with another program.

## Validation and limits

Synthetic tests cover dispatch, all service operation families, default/load/save,
reboot stages, raw packets, interval selection, missed deadlines, emergency stop,
queued pre-ARM command rejection, worker shutdown and staged-value rendering.

Actual board at node 7: control workers passed CAN and USB zero-command repetition,
RAM writes, flash SAVE/LOAD, and restoration of original saved and active datasets.
After replacing coarse event waits with short sleeps, observed 20 ms repeat TX
gaps peaked around 20.7 ms. Additional 50 ms and 10 ms watchdog tests used 12.5 ms
and 2.5 ms repeat intervals; observed gaps remained below their watchdogs over
short 0.3-second tests. Disabled-watchdog repetition was also exercised. These
measurements are observations, not long-duration or real-time guarantees. Actual
hidden-window GUI handlers passed connect, ARM+repeat-zero, repeat-toggle stop,
staged field write/restore and enabling/receiving/disabling all telemetry topics.
Evidence: ../bench/application/control_gui_test_results.json and
../bench/application/control_gui_ui_test_results.json.

No nonzero energized drive, brake output waveform or loaded current limiting was
physically tested. Reboot/default/raw/update handlers were validated synthetically;
this GUI did not perform a firmware upload during its tests. Existing firmware and
uploader bench evidence remains separate. Very short Windows heartbeat timing is
not guaranteed. Original configuration was restored and the board left disarmed.

Run synthetic checks: `python .\test_control_gui.py`.
Optional physical checks (changes/restores configuration and flash, requires an
already disarmed board with both RAM and saved safe states Hi-Z):
`python .\test_control_gui.py --hardware COM7 --node 7` or add `--usb --hardware COM3`.

## Firmware polarity and travel units (schema 4)

Positive commands mean increasing selected feedback for every interface. Each motor Configuration page now includes field 12 `output_reversed` (0/1) and field 13 `normalized_limits` (0 native, 1 normalized). No node masks. Limits, enable, polarity and units are on the same motor page and can be written atomically with Write staged page. SAVE persists RAM. Reporting mode does not select limit units. Existing schema-2/3 saved configuration and imported JSON migrate without losing the older settings.

## MCU identity

The main window above the tabs shows the 96-bit MCU unique ID, read once on each connection over the selected CAN/USB transport. It is read-only and independent of switch-selected node ID. The shared display is also present in Debug, Jog and Position GUIs. It is labelled last connected so retained identity is not mistaken for a live connection. Older firmware without UID capability shows Unavailable (older firmware).

From Firmware/tools: `python .\krabby.py --node 1 uid` (CANable), or `python .\krabby.py --usb --port COM3 uid`. The USB terminal command is `node uid`, using the usual APP binary-response encoding. Display order is word0, word1, word2, each eight uppercase hex digits.

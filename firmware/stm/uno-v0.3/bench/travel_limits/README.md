# Travel-limit and motor-polarity acceptance ? 2026-11-12

Actual STM32 board, CANable COM7, node 7. Motors 1/2 are secured linear actuators with potentiometer feedback; motor 3 is a secured geared motor with encoder feedback. Energized motion was authorized by the user.

Before modification, all four active RAM pages, sensor samples and status were captured in before_config.json. after_migration.json exactly matches these pages after schema-3 firmware startup from existing schema-2 flash: no existing values were lost.

application.srec/.elf/.map are the tested firmware (84,812 application bytes). CAN OpenBLT programming verified all image bytes including signature. No bootloader, peripheral configuration, flash allocation or adapter firmware changes.

physical_results.json: 42 physical checks: 3 electrical polarity probes, 3 corrected logical-positive movements, 24 outward-blocking combinations (three motors, native/normalized units, both polarity settings, lower/upper regions), and 12 actual retreat movements using measured polarity. These use each motor's installed feedback source, not every hypothetical source on every motor. All motor report modes were normalized while native and normalized enforcement were tested independently.

Measured output polarity: M1 reversed, M2 reversed, M3 normal; node field 8 = 3. This setting alone was added to the original saved configuration. Original limit thresholds, enables, calibration, current limits, watchdog and temperature calibration were restored. Native limits remain selected (node field 9 = 0). M1 retains enabled native limits 100..1000; M2/M3 retain limits disabled. M1 can be outside that original interval; only retreat is permitted while outside it. Review those original thresholds if full-stroke motion is wanted.

additional_checks.json: invalid mask rejection, exact encoder lower/upper equality inhibition, saved configuration persistence through reset and disarmed startup. gui_transaction.json: actual control worker's per-motor polarity/normalized-unit write over CAN, followed by original RAM restoration.

position_motion.json: actual Python PositionController moved all three motors in both logical directions towards normalized targets. Test cap 35, gain 6000, deadband 500, acceptance tolerance 650 normalized units; limits temporarily disabled for this tuning exercise and then restored. An initial run at gain 800 stalled short because requested low PWM did not overcome motor load/friction; no default controller tuning was changed. This is functional motion acceptance, not precision or load characterization.

The first polarity_probe.json is historical diagnostic evidence ONLY: its M1 delta included an artificial zero ramp after a configuration write. The firmware now seeds filter accumulators from latest ADC values instead of zero. The final physical_results.json supersedes that early probe.

Synthetic control, Jog and Position GUI tests passed, including worker cancellation, neutral ARM, repeat scheduling and per-motor mask writes preserving other motors. Those are simulations, distinct from physical records above.

Repeat energized acceptance only on the secured fixture:

```powershell
# From Firmware root; these explicitly move motors.
python tools/test_travel_limits.py --hardware COM7 --node 7
python tools/test_position_motion.py --hardware COM7 --node 7
```

The travel test reads RAM first, derives polarity with limits temporarily disabled, and on success SAVEs original settings plus measured polarity. The position test restores RAM and does not SAVE. Both disarm on completion/failure. Unexpected communication loss can prevent restoration; the untouched before_config.json is the original backup. Configuration editors work on RAM; SAVE is explicit.

Final board state: disarmed, commands zero, original configuration plus polarity mask 3 saved in schema 3. No power-cut, mechanical stopping-distance or loaded thermal characterization performed.

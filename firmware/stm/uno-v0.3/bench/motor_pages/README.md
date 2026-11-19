# Per-motor configuration pages ? 2026-11-12

Schema-4 tested application: application.srec/.elf/.map. Built 85,328 bytes; programmed through OpenBLT/CAN with full-image readback including signature. Board: actual node 7, CANable COM7, USB CDC COM3. MCU/peripherals, linker, bootloader, CAN timings and flash allocation unchanged.

Motor pages are now 56 bytes / 14 int32 fields. Each contains field 12 output_reversed and field 13 normalized_limits (0/1). Node page remains 48 bytes / 12 fields; its former masks are reserved zero again. The 224-byte stored dataset uses four 56-byte records; node's last two fields are internal zero padding. Stored header+dataset is 240 bytes within the original 4 KiB reservation. Full motor SET (62 bytes) fits one CAN FD frame; response fits too.

before_config.json captured all active configuration before changes. Firmware migrated saved schema-3 flash; migrated_config.json exactly matches host migration: original motor values unchanged, M1/M2 reversal=1, M3 reversal=0, normalized limits=0 on all three. SAVE/LOAD and reset persistence passed. Firmware also accepts schema-2 flash (new flags zero); JSON migration for schemas 2/3/4 has synthetic tests. Schema-2 flash migration was not re-exercised physically in this revision.

physical_results.json: repeated 42 actual motor/polarity/travel checks, using potentiometers on M1/M2 and encoder on M3. Both unit modes, either configured polarity, both outward blocks, and actual retreat for correct polarity. position_motion.json: actual PC PositionController drove all three motors toward normalized targets in both directions; test gain 6000, cap 35, deadband 500. These motion tests temporarily adjust RAM then restore original settings; no default tuning changes.

protocol_checks.json: CAN full-page lengths/bounds, invalid boolean rejection, actual GUI-worker atomic motor-page write without node changes, USB full-page reads and motor field-12 text command, and SAVE/reset configuration retention. Synthetic Control/Jog/Position GUI regressions passed separately.

Final state: disarmed, commands zero. Original user limits and enables retained, including M1 native 100..1000 enabled; M2/M3 limits disabled. M1/M2 reversed and M3 normal are persisted on their own motor pages. All limit-unit fields remain native. Modify thresholds explicitly if a different operating range is wanted.

From Firmware/tools:

```powershell
python .\krabby_position_gui.py --port COM7 --node 7
# Edit polarity/units directly on Configuration > Motor 1/2/3.
# Write staged page updates that motor atomically; SAVE persists RAM.
python .\openblt_canable.py ..\bench\motor_pages\application.srec --node 7
```

Repeat energized checks on the secured fixture (from Firmware root):

```powershell
python tools/test_travel_limits.py --hardware COM7 --node 7
python tools/test_position_motion.py --hardware COM7 --node 7
```

The travel test derives polarity and SAVEs restored original settings plus measured polarity; the position test restores RAM without SAVE. Historical schema-3 evidence remains in ../travel_limits. No power-cut or mechanical stopping-distance characterization in this change.

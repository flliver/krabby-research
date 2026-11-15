# Integrated application ? physical acceptance evidence

See ../../docs/DEVELOPMENT_LOG.md for implementation, decisions and limitations,
and ../../docs/APPLICATION_PROTOCOL.md for the wire format, schema and host commands.

- test.log/results.json: 37 physical checks on the final firmware.
- stress.log/stress_results.json: USB backpressure, bounded diagnostic overflow,
  watchdog and application/CAN/USB liveness. One real node, five absent addresses.
- can_regression.log: 1,680 verified bridge deliveries at 1M/5M.
- known_good.elf/.srec/.bin/.map: matching accepted application, 112 KiB linker limit.
- boot_before.bin and options_before.txt: before-work protection/boot baseline.
- config_before.bin: pre-implementation page containing historical test data.
- config_corrupt.bin/erased.bin/partial-header.bin: deliberately invalid test fixtures;
  never load them as firmware or as intended configuration.
- pre_regenerate.zip: pre-CubeMX-regeneration source backup, not final application.

From Firmware/tools:

```powershell
python .\openblt_canable.py ..\bench\application\known_good.srec --node 0
python .\krabby.py status
python .\krabby.py --usb --port COM3 sensors 0
```

Destructive acceptance replaces the configuration and application, uses ST-Link
resets/configuration-page fixture writes, and leaves all three safe states Hi-Z:

```powershell
python .\test_application.py --hardware
python .\test_application_stress.py --hardware
```

Both scripts locate images and results relative to their own repository location.
When running from Firmware root, use `python tools/test_application.py --hardware` instead.
Never run energized tests until VM/fuses, DAC voltage and unloaded fixture are verified.
After upload/reset, reopen USB. No protected flash or option settings may be changed.

Latest polarity/travel-limit firmware and energized acceptance: ../travel_limits/README.md. Historical known_good images above predate schema 3.

Current schema-4 motor-page layout and latest tested image: ../motor_pages/README.md.

The historical known_good.bin was regenerated from its matching ELF/S-record to remove erroneous flash-to-RAM padding. It is 84,280 bytes; its firmware contents and historical version are unchanged.

# M16 Task 4 bench validation

Run the interactive power-management validation as one stateful session:

```sh
python3 -m firmware.bench_tests list --suite task4
python3 -m firmware.bench_tests run --suite task4 \
  --port /dev/cu.usbmodem101 --remote USER@ORIN
```

The session spans Orin-owned and Mac-owned serial phases. It requires physical
confirmations and does not support `--yes` or running individual phases with
`--only`. Opening the serial port can reset the Mega, so the runner keeps a
single connection during each Mac-owned phase.

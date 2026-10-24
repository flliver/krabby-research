# GUI current-sense demo (simulated MCU + Isaac)

Exercise the firmware test GUI **Cur** column against the host sim: Isaac physics → current
sense → serial telemetry → `python -m firmware.gui` (same as hardware, different port).

This path needs **`IsaacMcuHost`** / `run_gui_sim_demo`; plain `ThreeBoardMcuHost` does not
run physics readback, so **Cur** will not track load there.

## Prerequisites

- Host firmware library built: `make -C firmware host-actuator`
- Isaac Lab environment (GPU): **`krabby-isaacsim`** image or local Isaac install
- GUI deps: `python3-tk`, `pip install pyserial` (and repo `firmware` package on `PYTHONPATH`)

## Automated smoke (headless, no GUI)

Inside Isaac (Docker or local):

```bash
export RUN_FIRMWARE_ISAAC=1
pytest tests/integration/firmware/test_gui_current_sense_sdk.py -v
```

Covers idle **0**, moving **~2**, ground **~4**, and lift **~30** bands on the SDK path and
GUI **Cur** text (`str(jt.current)` in ``JointRow.update_from_telemetry``, headless in pytest).

## Manual demo checklist

Run the Isaac PTY host (headless sim + serial port). It prints the slave path and keeps
running until Ctrl+C.

```bash
# From repo root, in krabby-isaacsim or Isaac-enabled shell:
RUN_FIRMWARE_ISAAC=1 python -m firmware.sim.run_gui_sim_demo
# optional: --joint FLHL  (default) --device cuda:0
```

In a **second terminal** (same machine, X11/display for tkinter):

```bash
python -m firmware.gui --port /dev/pts/N   # use path printed by the demo host
```

| Step | Action | Expected **Cur** (joint under test, e.g. FLHL) |
|------|--------|--------------------------------------------------|
| 1 | Connect; all joints at rest after auto `H` | **0** (idle ADC) |
| 2 | Hold **Extend** or **Retract** ~1 s (leg swings, zero-g play scene) | rises toward **~2** (moving, no contact) |
| 3 | *(Optional, gravity)* Use demo with `--gravity` or Isaac contact scene; drive knee into floor | **~4** band (ground contact) |
| 4 | *(Optional, high load)* Tune in `wrench_tuning_viewer` or loaded stance until wrench exceeds lift threshold | approaches **~30** (lift load band) |

Notes:

- Cur shows firmware `(int)avgIS` from telemetry — EMA smoothing may lag a few ticks.
- Three-board sim uses `ThreeBoardMcuHost` without Isaac; **current sense requires**
  `IsaacMcuHost` / `run_gui_sim_demo` so `sync_sensor_reads` runs each step.
- Breakpoints: `firmware/sim/current_sense_mapping.py` (`CurrentSenseBreakpoints`).

# Virtual serial (sim USB)

Host sim exposes firmware USB `Serial` to pyserial clients (`KrabbyMCUSDK`, `python -m firmware.gui`, scripts). Point them at the bridge with **`KRABBY_MCU_PORT`** (detection order in `firmware/mcu_port.py`).

If you open the port yourself (not via `KrabbyMCUSDK.connect()`): **250000** baud, **DTR off** — same as hardware.

| OS | Start bridge | Set `KRABBY_MCU_PORT` to |
|----|--------------|---------------------------|
| Linux / macOS | `python -m firmware.sim.serial_pty_bridge` | printed path, e.g. `/dev/pts/N` |
| Windows | `python -m firmware.sim.serial_tcp_bridge` | printed URL, e.g. `socket://127.0.0.1:54321` |

Isaac / GUI sim demos call `open_sim_mcu_usb_bridge()` for you (PTY on Unix, TCP on Windows); use the path or URL they print.

Isaac HAL **`--firmware-transport serial_sdk`** (with **`--backend fullsim`**) uses the same
PTY pattern; **`in_process`** vs **`serial_sdk`**: [`hal/server/isaac/README.md`](../../hal/server/isaac/README.md).

**Windows, COM-only tools:** optional null-modem pair via [com0com](https://sourceforge.net/projects/com0com/) (or a signed fork). The repo does not ship drivers; wire your app to one `COMn` and a relay on the other, or prefer the TCP bridge above.

Only one process should hold the port or URL. The same env var overrides auto-detect for a physical Mega (`COMn` / `/dev/ttyACM*`).

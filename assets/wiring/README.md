# M16 wiring

The Schemdraw sheet records the leader Mega's I²C chain: Qwiic adapter →
LSM6DSO IMU → SSD1306 OLED. The OLED uses address `0x3D`.

Render and validate the documentation with:

```sh
make -C assets/wiring render
```

Open the HTML output under `generated/sheets/` for the interactive schematic.
Each diagram module's filename is its canonical name and the basename of every
generated format.

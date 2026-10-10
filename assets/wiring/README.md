# M16 wiring

The Schemdraw sheet records the leader Mega's I²C chain: Qwiic adapter →
LSM6DSO IMU → SSD1306 OLED. The OLED uses address `0x3D`.

Render and validate the documentation with:

```sh
make -C assets/wiring render
```

Open the SVG under `generated/sheets/` for a scalable diagram, or the PNG for a preview.
Each diagram module's filename is its canonical name and the basename of every
generated format.

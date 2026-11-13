# SEN0581 ToF setup (Jetson)

Depth-only USB-serial cameras (DFRobot SEN0581 / Sipeed Meta Sense Lite). Official
hardware docs: [DFRobot SEN0581 wiki](https://wiki.dfrobot.com/sen0581/),
[Sipeed MetaSense-A010 AT reference](http://wiki.sipeed.com/hardware/en/metasense/metasense-a010/at_command_en.html).

## How it fits together

```text
Physical SEN0581 (unique USB serial)
        │
        ▼
/etc/krabby/sen0581.json          ← per-Orin role map (not in git)
  belly_right / belly_left / rear
        │  krabby install
        ▼
/etc/udev/rules.d/99-krabby-sen0581.rules   ← generated on the robot
        │
        ▼
/dev/krabby-tof-belly-right|belly-left|rear ← stable names
        │
        ▼
JETSON_SENSOR_CATALOG sen0581_device        ← same paths on every Orin
```

- **Repo / every robot:** catalog ids and `/dev/krabby-tof-*` paths only.
- **Each Orin:** `/etc/krabby/sen0581.json` maps that robot’s camera USB serials to
  those roles. **`sudo krabby install`** creates/updates the JSON when needed and
  always regenerates the udev rule from it.

No robot-specific USB serials are committed in the codebase.

**HAL catalog (portable):**

| Mount | Catalog id | Device path (`sen0581_device`) |
|-------|------------|--------------------------------|
| Belly right | `belly_right_tof` | `/dev/krabby-tof-belly-right` |
| Belly left | `belly_left_tof` | `/dev/krabby-tof-belly-left` |
| Rear | `rear_tof` | `/dev/krabby-tof-rear` |

Primary front camera remains **`front_rgbd`** (ZED). Each SEN0581 enumerates as
FT2232 Dual UART (`0403:6010`) with **two** tty nodes; HAL uses **interface 00 only**.

---

## New Orin: `krabby install` owns the mapping

1. Plug the three SEN0581s (and ZED) into the robot USB hub.
2. Run:

   ```bash
   sudo krabby install
   ```

3. Install will:
   - Ensure `/etc/krabby/sen0581.json` exists (see seeding below).
   - Generate `/etc/udev/rules.d/99-krabby-sen0581.rules` from that JSON.
   - Reload udev so `/dev/krabby-tof-*` appear.

4. Confirm:

   ```bash
   cat /etc/krabby/sen0581.json
   ls -l /dev/krabby-tof-*
   ```

### How `/etc/krabby/sen0581.json` is seeded

| Situation | What `krabby install` does |
|-----------|----------------------------|
| File already exists | Uses it; regenerates udev from it |
| Missing, and exactly **3** SEN0581s are plugged | Writes a draft map (serials sorted → `belly_right`, `belly_left`, `rear`) and warns you to verify roles against the LCD mounts |
| Missing, and fewer/more than 3 plugged | Skips SEN0581 udev (rest of install continues). Plug cameras and re-run, or create the JSON by hand |

JSON shape (USB `ATTRS{serial}` form; `ID_SERIAL_SHORT` with an underscore is also accepted):

```json
{
  "belly_right": "YYYYMM RIGHT01",
  "belly_left": "YYYYMM LEFT01",
  "rear": "YYYYMM REAR01"
}
```

Template: [`hal/server/jetson/udev/sen0581.json.example`](../hal/server/jetson/udev/sen0581.json.example).

Read each unit’s serial with:

```bash
udevadm info -q property -n /dev/ttyUSB0 | grep ID_SERIAL
ls -l /dev/serial/by-id/    # use the …-if00-port0 link per camera
```

After editing roles (camera swap, spare, wrong auto-draft):

```bash
sudo nano /etc/krabby/sen0581.json   # set serials for each role
sudo krabby install                  # regenerates udev
ls -l /dev/krabby-tof-*
```

Reboot and confirm the `/dev/krabby-tof-*` symlinks return without replugging.

---

## Unboxing / single module (bench)

1. **USB-C** — Data cable. Device shows as `SIPEED Meta Sense Lite` in `lsusb`
   (`0403:6010`). LCD shows a live depth colormap after power-up.
2. **Serial** — Record `ID_SERIAL_SHORT` / `ATTRS{serial}` for
   `/etc/krabby/sen0581.json` (if00 only).
3. **Optional stream check** — open the unit’s `…-if00-port0` by-id path at 115200,
   send `AT+ISP=1`, `AT+BINN=1`, `AT+UNIT=1`, `AT+FPS=19`, `AT+DISP=3`, then read
   bytes. Default mux is LCD-only (`DISP=1`); USB frames need `DISP=2` or `3`.

---

## Aim check

| Mount | Nominal aim |
|-------|-------------|
| Belly (each side) | ~55° down, yawed ~15° toward the front legs, facing outward |
| Rear | Aimed rearward, ~45° down |

Confirm feet visibility on the LCD, then ensure `/etc/krabby/sen0581.json` maps that
physical serial to the matching role.

---

## HAL verification

Confirm ToFs (and ZED) on the observation bus:

```bash
python -m hal.tools.hal_dump
```

Expect `rgbd_by_catalog_id` keys `front_rgbd`, `belly_right_tof`, `belly_left_tof`,
and `rear_tof` (depth-only for the ToFs). For a live multi-camera view, use the
**teleop portal** and select `front_rgbd` plus the `*_gray16_depth` preview ids
(see [TELEOP.md](TELEOP.md)). `python -m hal.tools.multi_stream_display` only
opens the ZED front RGB+depth pair — it does not fan out to the three ToFs.

Depth-only rows publish `channels={"depth"}` at 100×100 float32 meters and
`{id}_gray16_depth` previews (0.15–1.5 m). Missing units warn-and-skip.

Locomotion Docker passes `--device` for each present `/dev/krabby-tof-*`
(`krabby/_docker.py`) plus the privileged `/dev` mount.

---

## Dependencies

`pyserial` is already in the locomotion requirement files. No vendor SDK.

# Connect a gamepad (Pro Controller / third-party) to Jetson Orin

Drive the robot with any Linux joystick that shows up as `/dev/input/js*`.
This page covers **USB** (most reliable for clones), **`krabby pair-pro`**
(official Nintendo Switch Pro over Bluetooth), and **manual Bluetooth** for
third-party pads.

## Prerequisites

`krabby install` must have been run at least once. It installs the
`hid_nintendo` DKMS module (helps official Nintendo pads), configures BlueZ
for userspace HID, and writes udev rules so joysticks appear under
`/dev/input/js*`.

## Choose a path

| Pad | Recommended | Notes |
|-----|-------------|--------|
| Any pad with a data USB-C cable | **USB** (§A) | Fastest bring-up; skips BT bond issues |
| Official Nintendo Switch Pro | **`krabby pair-pro`** (§B) | Needs Sync pinhole (by USB-C), not Home |
| Third-party (e.g. Global Hunter XB-324) | **USB**, or **manual `bluetoothctl`** (§C) | Often **no Sync pinhole**; `pair-pro` may find it if it advertises as `Pro Controller`, but BT bonds often stay `Paired: no` |

After `/dev/input/js*` exists, drive with:

```bash
krabby run --gamepad-only
```

(On a fleet-enrolled Orin, plain `krabby run` is portal teleop — use `--gamepad-only` for the pad.)

---

## A. USB (recommended for third-party / frequent bench testing)

1. Plug the pad into the Orin with a **data** USB cable (charge-only cables will not create `js*`).
2. Check:

```bash
ls /dev/input/js*
lsusb | grep -iE 'nintendo|057e|game|xbox|controller' || true
```

3. Optional stick test:

```bash
sudo apt install -y joystick
jstest /dev/input/js0   # or js1 if that is the pad
```

4. Start the gamepad stack:

```bash
krabby run --gamepad-only
# If sticks do nothing, try: krabby run --gamepad-only -- --device-id 1
```

USB does not need Sync, `pair-pro`, or a lasting Bluetooth bond.

---

## B. Official Nintendo Pro — `krabby pair-pro` (Bluetooth)

### B1. Pairing mode (Sync ≠ Home)

| Button | Role |
|--------|------|
| **Sync** (small pinhole next to USB-C) | Fresh Bluetooth pair — required for `pair-pro` |
| **Home** (house icon) | Wake / reconnect from cache — **not** pairing mode |

Hold **Sync** 3–5s until **all four LEDs flash rapidly**.

### B2. Run pairing

With the `krabby-launcher` venv activated (after `krabby install`):

```bash
sudo -E env PATH="$PATH" "$(which krabby)" pair-pro
```

(`krabby pair-pro` ships the script in `krabby-launcher`. From a clone you can
still run `sudo bash scripts/jetson/pair_pro_controller.sh`.)

The script discovers a device named **`Pro Controller`**, pairs it, captures
the link key (needed on L4T because BlueZ `store_hint=0` would drop it), and
writes it under `/var/lib/bluetooth/`.

**Success:** `Paired: yes`, `Connected: yes`, and `js device: /dev/input/js0`
(or `js1`).

### B3. Reconnecting later

Press **Home**. LED 1 should light; no re-pair needed if the bond stuck.
If reconnect fails, put the pad in **Sync** again and re-run `pair-pro`
(not a quick Home wake).

---

## C. Third-party Bluetooth (no Sync / `pair-pro` unreliable)

Clones (XB-324 class and similar) often:

- Have **no Sync pinhole**
- Use **long-press Home** (~3s, LEDs flash fast) or a **BT / pair** button as pairing mode
- Advertise as `Pro Controller`, `XB 324`, or another name
- Drop the bond (`Paired: no`) or power off in ~30–60s if pairing stalls

`pair-pro` only auto-discovers the name **`Pro Controller`**. If your pad uses
another name, or BT never reaches `Paired: yes`, use **USB (§A)** or manual
Bluetooth below.

### C1. Put the pad in its pairing mode

Check the maker’s sheet. Common patterns:

1. Fully power off (hold Home ~10s until LEDs dark).
2. Hold **Home ~3s** *or* a dedicated **BT/pair** button until LEDs flash quickly.
3. Keep them flashing while you pair (many pads sleep after ~30–60s).

### C2. Manual `bluetoothctl` (GUI “remove → pair” equivalent)

The desktop Bluetooth UI turns **pairable** on and runs a pairing agent. CLI
must do that explicitly. Enter **one command at a time** and wait for a reply
before the next (`pair` can take 10–30s; pasting `pair`/`trust`/`connect`/`quit`
together often leaves the bond stuck at `Paired: no`).

```bash
# Optional: clear a bad bond first (same as GUI remove / Paired off)
bluetoothctl remove AA:BB:CC:DD:EE:FF

# Put pad in pairing mode (LEDs flashing), then:
bluetoothctl
```

```text
power on
pairable on
agent on
default-agent
scan on
```

Wait for `[NEW] Device …` (name may be `Pro Controller`, `XB 324`, etc.). Then
**one line at a time**:

```text
pair AA:BB:CC:DD:EE:FF
```

Wait for **Pairing successful** (or a clear failure). Then:

```text
trust AA:BB:CC:DD:EE:FF
connect AA:BB:CC:DD:EE:FF
info AA:BB:CC:DD:EE:FF
```

You want `Paired: yes` and `Connected: yes`. Then `quit` and:

```bash
ls /dev/input/js*
```

If `pair` hangs or fails but the device is still in scan, try **`connect` alone**
(some clones bond on connect the way the GUI “Paired” toggle does):

```text
connect AA:BB:CC:DD:EE:FF
trust AA:BB:CC:DD:EE:FF
info AA:BB:CC:DD:EE:FF
```

If it still fails, or LEDs go out: `bluetoothctl remove …`, re-enter pairing
mode, retry — or use USB (§A). Confirm the adapter is pairable:

```bash
bluetoothctl show | grep -E 'Pairable|Powered'
# Pairable: yes
```

### C3. Optional: try `pair-pro` anyway

If `bluetoothctl devices` shows **`Pro Controller`**, you can run `pair-pro`.
Expect **Found too quickly** warns and frequent **`Paired: no`** on clones —
that is an incomplete bond, **not** a missing `hid_nintendo`. Prefer §A or §C2.

---

## Troubleshooting

### Pad turns off / LEDs go dark during pairing

Normal idle/power-save (~30–60s) when the bond does not finish. Power off fully,
re-enter pairing mode, and finish `pair` / `connect` (or `pair-pro`) while LEDs
are still flashing. Or use USB.

### `Found too quickly` (from `pair-pro`)

Often a **Home/cache** reconnect rather than Sync. The script **warns and
continues** (a pad already advertising can also appear in under 3s). If you then
get `Paired: no`, recover as below. On third-party pads without Sync, switch to
USB (§A) or manual BT (§C).

### CLI pair fails but GUI “Paired” toggle works

Usually the CLI skipped `pairable on` / `agent on`, or ran `pair`+`connect`+`quit`
in one paste. Follow §C2 one command at a time. Check `bluetoothctl show` shows
`Pairable: yes` while pairing.

### `Paired: no` (incomplete bond)

Not an `hid_nintendo` problem. Recover:

```bash
# 1. Power off pad (hold Home ~10s until LEDs dark)
# 2. Clear stale entry:
bluetoothctl remove <MAC>
# 3. Official Pro: hold Sync until ALL 4 LEDs flash rapidly
#    Third-party: long-press Home or BT/pair button per maker docs
# 4. Re-run pair-pro (official / name "Pro Controller") or bluetoothctl pair/trust/connect
# 5. If BT still fails: use USB (§A)
```

### No `/dev/input/js*` after `Paired: yes` (or over USB)

Then check drivers / udev:

```bash
lsmod | grep hid_nintendo
# Official Nintendo pads:
sudo modprobe hid_nintendo
# Any HID gamepad may still appear via hid-generic without hid_nintendo
ls /dev/input/js*
lsusb
```

After a kernel update, re-run host setup so DKMS rebuilds:

```bash
sudo -E env PATH="$PATH" "$(which krabby)" install
```

### `pair-pro` never finds the device

- Pad not advertising, or name ≠ `Pro Controller` → use §C (`scan on` / note real name) or USB.
- Fully off → pairing mode → retry within a few seconds.

### Sticks do nothing after `krabby run`

- Use `krabby run --gamepad-only` on enrolled hosts.
- Try `-- --device-id 1` if both `js0` and `js1` exist.
- After idle BT sleep (LEDs out) or a mid-session unplug: press **Home** (or replug USB) and wait up to ~10s — `InputController` reopens the pad without restarting the stack. Logs show disconnect then reconnect. Only if sticks stay dead after that: `docker rm -f krabby` then `krabby run`, or `sudo systemctl restart krabby-locomotion`.
- To exercise a **local** (not yet imaged) controller tree: from the clone,
  `krabby run --gamepad-only --mount "$PWD:/opt/krabby-research"`
  (launch sets `PYTHONPATH` when that mount is present). Smoke-test reopen only:
  `python3 controller/scripts/jetson/verify_joystick_reopen.py`

### Player 2 / multiple LEDs (official Pro)

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then disconnect and reconnect the controller.

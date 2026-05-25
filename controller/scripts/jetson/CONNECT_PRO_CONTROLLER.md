# Connect Pro Controller via Bluetooth to Jetson Orin

## Prerequisites

`krabby install` must have been run at least once. It installs the `hid_nintendo`
DKMS module, configures BlueZ for userspace HID, and writes the udev rules needed
for the Pro Controller to appear as a `/dev/input/js*` device.

## 1. Put the controller in pairing mode

Press and hold the **Sync** button (small circular hole near the USB-C port on
top) for **3–5 seconds** until the four LEDs start **flashing rapidly**.

## 2. Run the pairing script

```bash
sudo bash scripts/pair_pro_controller.sh
```

The script scans for a nearby Pro Controller, pairs it, captures the link key
(required on L4T because BlueZ's `store_hint=0` would otherwise discard it), and
writes the key to `/var/lib/bluetooth/`. When done, LED 1 lights up solid and
`/dev/input/js0` (or `js1`) appears.

Example output:

```
[1] Adapter: AA:BB:CC:DD:EE:FF
[2] btmon started (pid 12345)
[3] Discoverable on
[4] Scanning ...
[5] (no prior pairing found)
[6] Trusting + pairing ...
[7] Waiting for connection ...
[8] Waiting for Paired: yes ...
[9] Link key: AABBCCDDEEFF00112233445566778899
[10] Key written to /var/lib/bluetooth/...
Done. /dev/input/js0 is ready.
```

## 3. Verify

```bash
ls /dev/input/js*
```

You should see at least `js0`. The first LED on the controller should be lit solid.

To confirm input events are flowing:

```bash
sudo apt install -y joystick
jstest /dev/input/js0
```

## 4. Reconnecting later

Press the **Home** button. The controller will reconnect automatically; no
re-pairing is needed. LED 1 will light up when connected.

If it fails to reconnect, re-run `scripts/pair_pro_controller.sh`.

## Troubleshooting

**No `/dev/input/js*` after pairing**

Verify `hid_nintendo` is loaded:

```bash
lsmod | grep hid_nintendo
modinfo hid_nintendo
```

If not loaded, check DKMS:

```bash
dkms status
sudo modprobe hid_nintendo
```

`krabby install` installs DKMS automatically, but if it was run on a kernel that
was later updated you may need to re-run `sudo krabby install` so DKMS rebuilds
the module for the new kernel.

**Controller connects but shows as player 2 / multiple LEDs**

The udev rule installed by `krabby install` sets LED 1 on every BT connect. If
rules haven't taken effect yet, run:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then disconnect and reconnect the controller.

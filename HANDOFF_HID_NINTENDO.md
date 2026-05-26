# Handoff: Add hid_nintendo DKMS install to `krabby --install`

## What we're doing and why

**M14 (Krabby Installable Stack)** has an acceptance criterion that requires a non-technical
kit owner to go from boxed parts to a moving robot via `pip install krabby` + a few CLI
commands. One of those commands, `krabby --install`, sets up the host environment
(udev rules, dialout group) before pulling the locomotion Docker image.

During bench testing we discovered that on the **stock Jetson L4T kernel (5.15-tegra)**,
the `hid_nintendo` kernel module is **not present**. This module is required for the
Nintendo Switch Pro Controller to work as a USB HID device or Bluetooth HID device on
Linux. Without it, the controller shows up in `lsusb` and `/proc/bus/input/devices` but
sends no input events — buttons and sticks do nothing.

This affects **both USB and Bluetooth** connections. M14's acceptance criteria explicitly
requires driving at least one joint with a Bluetooth Pro Controller, so this is a hard
blocker.

## The fix

Extend `krabby --install` (specifically `krabby/_host.py`) to detect whether
`hid_nintendo` is available and, if not, install it via DKMS.

The DKMS source to use: https://github.com/nicman23/dkms-hid-nintendo
(or the DanielOgorchock backport — verify which builds cleanly on L4T 5.15-tegra)

The install logic should:
1. Check `modinfo hid_nintendo` — if found, skip (already available).
2. If not found, check that `dkms` and `git` are available (apt-install if not).
3. Clone the DKMS repo, `dkms add`, `dkms build`, `dkms install`, `modprobe hid_nintendo`.
4. Add a udev rule for the Pro Controller (VID `057e`, PID `2009`) while we're at it
   (see below).
5. Print clear `[ok]` / `[err]` status consistent with the existing style in `_host.py`.

Also add a udev rule for the Pro Controller so non-root can read `/dev/input/js*`:
```
SUBSYSTEM=="usb", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009", MODE="0666", TAG+="uaccess"
KERNEL=="hidraw*", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009", MODE="0666", TAG+="uaccess"
```

## Relevant files

- **`krabby/install.py`** — top-level `cmd_install()`, calls `run_host_setup()` then pulls image.
- **`krabby/_host.py`** — host setup logic (udev rule for Mega, dialout group). Add
  `_ensure_hid_nintendo()` here and call it from `run_host_setup()`.
- **`controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md`** — existing pairing docs;
  its troubleshooting section already mentions `hid_nintendo` as a known gap.
- **`patina-foundation-grants/grants/Krabby-Uno/Milestone14-Jetson-Orin-Installable-Stack/TASK-3-KRABBY-CLI-ECR-INSTALL.md`**
  — M14 Task 3 scope; "fix any device-passthrough or permissions gaps found during
  integration" covers this change.

## Environment confirmed during discovery

- Jetson: `uname -r` → `5.15.148-tegra`
- `modinfo hid_nintendo` → `ERROR: Module hid_nintendo not found.`
- `lsmod | grep hid` → (empty)
- Controller VID:PID: `057e:2009` (Nintendo Switch Pro Controller, USB)
- Device does appear in `/dev/input/` as `js0` + `event1` but produces no input events

## HAL wiring status (verified)

Fletcher's note — "you just need the default image with the Bluetooth controller hooked up to
HAL instead of model" — describes an existing code path. No HAL work is needed.

**What already exists:**

- `INPUT_CONTROLLER_KRABBY` mode in `controller/control_loop.py:195` wires the gamepad
  directly through `GamepadToKrabbyHALMapper` → `HalClient` → HAL server, bypassing the
  policy model entirely.
- `krabby run --gamepad-only` — starts `krabby-hal-server-jetson --control-source gamepad`
  in the locomotion container; no `--checkpoint` required.
- `controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md` — full runbook for the two-process
  gamepad-only flow.

**The complete AC verification path (once `hid_nintendo` is installed):**

```
Pro Controller (Bluetooth)
  → hid_nintendo kernel module  ← MISSING (this handoff)
  → /dev/input/js0
  → InputController (pygame SDL2)
  → GamepadToKrabbyHALMapper
  → HalClient (ZMQ TCP)
  → Jetson HAL server (krabby run --gamepad-only, no checkpoint needed)
  → KrabbyMCUSDK → firmware
```

Run with:
```bash
krabby run --gamepad-only
# second terminal:
krabby uno
```

The HAL layer is complete. `hid_nintendo` is the only remaining blocker.

## What was NOT tried yet

- The DKMS install itself on this Jetson (we stopped to write this handoff)
- Bluetooth pairing (blocked by missing module; same root cause)
- `joycond` userspace daemon (alternative approach, not needed if DKMS works)

"""One-time host setup: udev rule and dialout group membership."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UDEV_RULE_PATH = Path("/etc/udev/rules.d/99-krabby-mega.rules")
UDEV_RULE = (
    'SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042",'
    ' MODE="0666", GROUP="dialout"\n'
)

_PRO_CONTROLLER_UDEV_PATH = Path("/etc/udev/rules.d/99-krabby-pro-controller.rules")
_PRO_CONTROLLER_UDEV_RULE = (
    'SUBSYSTEM=="usb", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009",'
    ' MODE="0666", TAG+="uaccess"\n'
    'KERNEL=="hidraw*", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009",'
    ' MODE="0666", TAG+="uaccess"\n'
)

_HID_NINTENDO_DKMS_REPO = "https://github.com/nicman23/dkms-hid-nintendo"
_HID_NINTENDO_MODULES_LOAD = Path("/etc/modules-load.d/hid_nintendo.conf")

_PRO_CONTROLLER_LED_UDEV_PATH = Path("/etc/udev/rules.d/99-krabby-pro-controller-led.rules")
_PRO_CONTROLLER_LED_UDEV_RULE = (
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player1",'
    ' RUN+="/bin/sh -c \'echo 1 > /sys%p/brightness\'"\n'
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player2",'
    ' RUN+="/bin/sh -c \'echo 0 > /sys%p/brightness\'"\n'
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player3",'
    ' RUN+="/bin/sh -c \'echo 0 > /sys%p/brightness\'"\n'
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player4",'
    ' RUN+="/bin/sh -c \'echo 0 > /sys%p/brightness\'"\n'
)

_BT_INPUT_CONF = Path("/etc/bluetooth/input.conf")
# Required on L4T 5.15-tegra: kernel has no BTPROTO_HIDP socket, so BlueZ must
# use the userspace uhid path. ClassicBondedOnly=false is needed because the Pro
# Controller sends store_hint=0 (does not persist its link key), which would
# otherwise cause the HIDP connection to be rejected as non-bonded.
_BT_INPUT_CONF_SETTINGS = {
    "UserspaceHID": "true",
    "ClassicBondedOnly": "false",
}


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def _ensure_udev_rule() -> bool:
    if UDEV_RULE_PATH.exists() and UDEV_RULE_PATH.read_text() == UDEV_RULE:
        print(f"[ok]  udev rule already in place: {UDEV_RULE_PATH}")
        return True
    try:
        UDEV_RULE_PATH.write_text(UDEV_RULE)
        _run(["udevadm", "control", "--reload-rules"])
        _run(["udevadm", "trigger"])
        print(f"[+]   wrote udev rule: {UDEV_RULE_PATH}")
        return True
    except PermissionError:
        print(f"[err] cannot write {UDEV_RULE_PATH} — run with sudo", file=sys.stderr)
        return False


def _ensure_pro_controller_udev() -> bool:
    if _PRO_CONTROLLER_UDEV_PATH.exists() and _PRO_CONTROLLER_UDEV_PATH.read_text() == _PRO_CONTROLLER_UDEV_RULE:
        print(f"[ok]  Pro Controller udev rule already in place: {_PRO_CONTROLLER_UDEV_PATH}")
        return True
    try:
        _PRO_CONTROLLER_UDEV_PATH.write_text(_PRO_CONTROLLER_UDEV_RULE)
        _run(["udevadm", "control", "--reload-rules"])
        _run(["udevadm", "trigger"])
        print(f"[+]   wrote Pro Controller udev rule: {_PRO_CONTROLLER_UDEV_PATH}")
        return True
    except PermissionError:
        print(f"[err] cannot write {_PRO_CONTROLLER_UDEV_PATH} — run with sudo", file=sys.stderr)
        return False


def _dkms_state(pkg_name: str, version: str) -> str | None:
    """DKMS state for a module/version: 'installed', 'built', 'added', or None."""
    try:
        res = subprocess.run(["dkms", "status", "-m", pkg_name, "-v", version],
                             capture_output=True, text=True)
    except FileNotFoundError:
        return None
    out = res.stdout.lower()
    for st in ("installed", "built", "added"):
        if st in out:
            return st
    return None


def _dkms_ensure_installed(src: Path, pkg_name: str, version: str) -> bool:
    """Add/build/install a DKMS module, skipping any step already done.

    `dkms add` errors out if the module is already in the tree, so a prior run
    that added but never finished building/installing (e.g. it was interrupted)
    used to make every subsequent `krabby install` fail. Branch on the current
    dkms state instead so re-runs are idempotent.
    """
    state = _dkms_state(pkg_name, version)
    if state is None:
        if _run(["dkms", "add", str(src)]) != 0:
            print("[err] dkms add failed", file=sys.stderr)
            return False
        state = "added"
    else:
        print(f"[ok]  {pkg_name}/{version} already in DKMS tree ({state})")
    if state == "added":
        print(f"      building {pkg_name}/{version} (this may take a minute) ...")
        if _run(["dkms", "build", "-m", pkg_name, "-v", version]) != 0:
            print("[err] dkms build failed", file=sys.stderr)
            return False
        state = "built"
    if state == "built":
        if _run(["dkms", "install", "-m", pkg_name, "-v", version]) != 0:
            print("[err] dkms install failed", file=sys.stderr)
            return False
    return True


def _ensure_hid_nintendo() -> bool:
    try:
        if subprocess.run(["modinfo", "hid_nintendo"], capture_output=True).returncode == 0:
            print("[ok]  hid_nintendo kernel module already present")
            return True
    except FileNotFoundError:
        print("[skip] modinfo not found — skipping hid_nintendo setup (not Linux?)")
        return True

    print("      hid_nintendo not found — installing via DKMS ...")

    for pkg in ("dkms", "git"):
        if not shutil.which(pkg):
            print(f"      apt-installing {pkg} ...")
            if _run(["apt-get", "install", "-y", pkg]) != 0:
                print(f"[err] apt-get install {pkg} failed", file=sys.stderr)
                return False

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "dkms-hid-nintendo"
        print(f"      cloning {_HID_NINTENDO_DKMS_REPO} ...")
        if _run(["git", "clone", "--depth=1", _HID_NINTENDO_DKMS_REPO, str(src)]) != 0:
            print("[err] git clone dkms-hid-nintendo failed", file=sys.stderr)
            return False

        pkg_name = None
        version = None
        for line in (src / "dkms.conf").read_text().splitlines():
            if line.startswith("PACKAGE_NAME") and pkg_name is None:
                pkg_name = line.split("=", 1)[1].strip().strip('"')
            if line.startswith("PACKAGE_VERSION") and version is None:
                version = line.split("=", 1)[1].strip().strip('"')
        if not pkg_name or not version:
            print("[err] could not parse PACKAGE_NAME/VERSION from dkms.conf", file=sys.stderr)
            return False

        if not _dkms_ensure_installed(src, pkg_name, version):
            return False

    if _run(["modprobe", "hid_nintendo"]) != 0:
        print("[err] modprobe hid_nintendo failed", file=sys.stderr)
        return False

    try:
        _HID_NINTENDO_MODULES_LOAD.write_text("hid_nintendo\n")
        print(f"[+]   wrote {_HID_NINTENDO_MODULES_LOAD} (auto-load on boot)")
    except PermissionError:
        print(f"[err] cannot write {_HID_NINTENDO_MODULES_LOAD} — run with sudo", file=sys.stderr)
        return False

    print("[+]   hid_nintendo installed and loaded")
    return True


def _ensure_pro_controller_led_udev() -> bool:
    if _PRO_CONTROLLER_LED_UDEV_PATH.exists() and _PRO_CONTROLLER_LED_UDEV_PATH.read_text() == _PRO_CONTROLLER_LED_UDEV_RULE:
        print(f"[ok]  Pro Controller LED udev rule already in place: {_PRO_CONTROLLER_LED_UDEV_PATH}")
        return True
    try:
        _PRO_CONTROLLER_LED_UDEV_PATH.write_text(_PRO_CONTROLLER_LED_UDEV_RULE)
        _run(["udevadm", "control", "--reload-rules"])
        print(f"[+]   wrote Pro Controller LED udev rule: {_PRO_CONTROLLER_LED_UDEV_PATH}")
        return True
    except PermissionError:
        print(f"[err] cannot write {_PRO_CONTROLLER_LED_UDEV_PATH} — run with sudo", file=sys.stderr)
        return False


def _ensure_bt_input_conf() -> bool:
    if not _BT_INPUT_CONF.exists():
        print(f"[skip] {_BT_INPUT_CONF} not found — skipping Bluetooth HID config")
        return True

    try:
        text = _BT_INPUT_CONF.read_text()
    except PermissionError:
        print(f"[err] cannot read {_BT_INPUT_CONF} — run with sudo", file=sys.stderr)
        return False

    changed = False
    for key, value in _BT_INPUT_CONF_SETTINGS.items():
        active_pattern = re.compile(rf"^{key}={value}$", re.MULTILINE)
        if active_pattern.search(text):
            print(f"[ok]  {_BT_INPUT_CONF}: {key}={value} already set")
            continue
        # Replace commented or wrong-value line, or append if absent.
        any_line = re.compile(rf"^#?{key}=.*$", re.MULTILINE)
        if any_line.search(text):
            text = any_line.sub(f"{key}={value}", text)
        else:
            text = text.rstrip("\n") + f"\n{key}={value}\n"
        print(f"[+]   {_BT_INPUT_CONF}: set {key}={value}")
        changed = True

    if not changed:
        return True

    try:
        _BT_INPUT_CONF.write_text(text)
    except PermissionError:
        print(f"[err] cannot write {_BT_INPUT_CONF} — run with sudo", file=sys.stderr)
        return False

    ret = _run(["systemctl", "restart", "bluetooth"])
    if ret != 0:
        print("[err] failed to restart bluetooth service", file=sys.stderr)
        return False
    print("[+]   bluetooth service restarted")
    return True


def _ensure_dialout() -> None:
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or ""
    if not user:
        print("[skip] could not determine invoking user for dialout group")
        return
    result = subprocess.run(["groups", user], capture_output=True, text=True)
    if "dialout" in result.stdout:
        print(f"[ok]  {user} already in dialout group")
    else:
        ret = _run(["usermod", "-aG", "dialout", user])
        if ret == 0:
            print(f"[+]   added {user} to dialout group (re-login to take effect)")
        else:
            print(f"[err] usermod failed (exit {ret})", file=sys.stderr)


_BOOT_SERVICE_PATH = Path("/etc/systemd/system/krabby-locomotion.service")
_BOOT_SERVICE_NAME = "krabby-locomotion.service"


def _boot_service_unit(krabby_bin: str, user: str) -> str:
    """systemd unit that runs `krabby run` (the full gamepad stack) on boot."""
    return f"""\
[Unit]
Description=Krabby locomotion stack
Documentation=https://github.com/flliver/krabby-research
After=docker.service
Requires=docker.service
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
User={user}
# Clear any container left by an unclean shutdown so --name krabby is free.
ExecStartPre=-/usr/bin/docker rm -f krabby
ExecStart={krabby_bin} run
# The container runs under containerd, not this unit's cgroup; stop it explicitly.
ExecStop=/usr/bin/docker stop krabby
# The launcher exits if the MCU has not enumerated yet at boot; retry until it does.
Restart=on-failure
RestartSec=5
TimeoutStopSec=40

[Install]
WantedBy=multi-user.target
"""


def _ensure_boot_service(launch_on_startup: bool) -> bool:
    """Install (default) or remove the systemd unit that starts `krabby run` on boot.

    Enabled for boot but not started here; the stack comes up on the next boot (run
    `krabby run` to start immediately). `--no-launch-on-startup` tears down a prior unit.
    """
    if not shutil.which("systemctl"):
        print("[skip] systemctl not found — skipping boot autostart (not a systemd host?)")
        return True

    if not launch_on_startup:
        if _BOOT_SERVICE_PATH.exists():
            _run(["systemctl", "disable", "--now", _BOOT_SERVICE_NAME])
            try:
                _BOOT_SERVICE_PATH.unlink()
            except PermissionError:
                print(f"[err] cannot remove {_BOOT_SERVICE_PATH} — run with sudo", file=sys.stderr)
                return False
            _run(["systemctl", "daemon-reload"])
            print(f"[+]   boot autostart removed ({_BOOT_SERVICE_NAME})")
        else:
            print("[ok]  boot autostart not installed (--no-launch-on-startup)")
        return True

    # The service runs as the invoking user (not root) so it shares their install
    # state and docker/dialout access, mirroring the manual `krabby run`.
    krabby_bin = shutil.which("krabby") or "/usr/local/bin/krabby"
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
    unit = _boot_service_unit(krabby_bin, user)

    if _BOOT_SERVICE_PATH.exists() and _BOOT_SERVICE_PATH.read_text() == unit:
        print(f"[ok]  boot autostart unit already in place: {_BOOT_SERVICE_PATH}")
    else:
        try:
            _BOOT_SERVICE_PATH.write_text(unit)
        except PermissionError:
            print(f"[err] cannot write {_BOOT_SERVICE_PATH} — run with sudo", file=sys.stderr)
            return False
        _run(["systemctl", "daemon-reload"])
        print(f"[+]   wrote {_BOOT_SERVICE_PATH}")

    ret = _run(["systemctl", "enable", _BOOT_SERVICE_NAME])
    if ret != 0:
        print(f"[err] systemctl enable {_BOOT_SERVICE_NAME} failed (exit {ret})", file=sys.stderr)
        return False
    print(f"[ok]  boot autostart enabled (runs as {user}); starts on next boot. "
          f"Disable with `krabby install --no-launch-on-startup` or `sudo systemctl disable {_BOOT_SERVICE_NAME}`.")
    return True


def run_host_setup(launch_on_startup: bool = True) -> None:
    ok = _ensure_udev_rule()
    ok &= _ensure_pro_controller_udev()
    ok &= _ensure_pro_controller_led_udev()
    ok &= _ensure_hid_nintendo()
    ok &= _ensure_bt_input_conf()
    ok &= _ensure_boot_service(launch_on_startup)
    _ensure_dialout()
    if ok:
        print("\nHost setup complete. Replug your Mega boards.")
    else:
        print("\nHost setup incomplete — fix errors above and re-run.", file=sys.stderr)
        sys.exit(1)

"""krabby-firmware install: one-time host setup for flashing Mega 2560 without sudo."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

UDEV_RULE_PATH = Path("/etc/udev/rules.d/99-krabby-mega.rules")
UDEV_RULE = (
    'SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042",'
    ' MODE="0666", GROUP="dialout"\n'
)
PLATFORM_LOCAL_PATH = Path.home() / ".arduino15/packages/arduino/hardware/avr/1.8.7/platform.local.txt"
PLATFORM_LOCAL_CONTENT = "compiler.cpp.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256\ncompiler.c.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256\n"


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


def _ensure_tool(tool: str, install_cmd: list[str]) -> None:
    if shutil.which(tool):
        print(f"[ok]  {tool} already installed")
        return
    print(f"[+]   installing {tool} ...")
    ret = _run(install_cmd)
    if ret != 0:
        print(f"[err] failed to install {tool} (exit {ret})", file=sys.stderr)
    else:
        print(f"[ok]  {tool} installed")


def _ensure_platform_local() -> None:
    if PLATFORM_LOCAL_PATH.exists() and PLATFORM_LOCAL_PATH.read_text() == PLATFORM_LOCAL_CONTENT:
        print(f"[ok]  platform.local.txt already set: {PLATFORM_LOCAL_PATH}")
        return
    PLATFORM_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLATFORM_LOCAL_PATH.write_text(PLATFORM_LOCAL_CONTENT)
    print(f"[+]   wrote platform.local.txt: {PLATFORM_LOCAL_PATH}")


def run_install() -> None:
    ok = _ensure_udev_rule()
    _ensure_dialout()
    _ensure_tool("avrdude", ["apt-get", "install", "-y", "avrdude"])
    _ensure_tool("arduino-cli", ["snap", "install", "arduino-cli", "--classic"])
    _ensure_platform_local()
    if ok:
        print("\nHost setup complete. Replug your Mega boards before flashing.")
    else:
        print("\nHost setup incomplete — fix errors above and re-run.", file=sys.stderr)
        sys.exit(1)

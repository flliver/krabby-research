"""One-time host setup: udev rule and dialout group membership."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

UDEV_RULE_PATH = Path("/etc/udev/rules.d/99-krabby-mega.rules")
UDEV_RULE = (
    'SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042",'
    ' MODE="0666", GROUP="dialout"\n'
)


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


def run_host_setup() -> None:
    ok = _ensure_udev_rule()
    _ensure_dialout()
    if ok:
        print("\nHost setup complete. Replug your Mega boards.")
    else:
        print("\nHost setup incomplete — fix errors above and re-run.", file=sys.stderr)
        sys.exit(1)

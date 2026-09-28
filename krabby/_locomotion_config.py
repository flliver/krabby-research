"""Fleet locomotion kit config (/etc/krabby/locomotion.json) and HAL argv for enrolled hosts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from krabby._iot import IOT_DIR

LOCOMOTION_CONFIG_PATH = Path("/etc/krabby/locomotion.json")
LOCOMOTION_SERVICE_NAME = "krabby-locomotion.service"
DEFAULT_TELEOP_IP = "127.0.0.1"
_COLD_START_MIN_INTERVAL_S = 60.0
_last_cold_start_attempt = 0.0


def fleet_enrolled() -> bool:
    """True if this host completed ``krabby enroll``.

    ``krabby-locomotion`` and foreground ``krabby run`` execute as the kit user,
    while ``/etc/krabby/iot`` is root ``0700`` with a ``0600`` private key — do
    not ``stat`` those paths; ``locomotion.json`` is the non-root enrollment marker.
    """
    if LOCOMOTION_CONFIG_PATH.is_file():
        return True
    try:
        return IOT_DIR.is_dir() and (IOT_DIR / "config.json").is_file()
    except PermissionError:
        return IOT_DIR.is_dir()


def _config_home() -> Path:
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        try:
            import pwd

            return Path(pwd.getpwnam(sudo_user).pw_dir)
        except (KeyError, ImportError):
            pass
    return Path.home()


def default_config() -> dict[str, Any]:
    return {
        "control_source": "portal",
        "robot": "hex",
        "teleop_ip": DEFAULT_TELEOP_IP,
        "teleop_control_echo": False,
        "checkpoint": None,
        "checkpoint_host_dir": None,
        "zed_resources_host": None,
        "zed_settings_host": None,
    }


def load_config() -> dict[str, Any]:
    cfg = default_config()
    if not LOCOMOTION_CONFIG_PATH.is_file():
        return cfg
    try:
        on_disk = json.loads(LOCOMOTION_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return cfg
    if isinstance(on_disk, dict):
        cfg.update(on_disk)
    return cfg


def write_config(config: dict[str, Any]) -> None:
    LOCOMOTION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCOMOTION_CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def write_default_config(
    *,
    control_source: Optional[str] = None,
    robot: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_host_dir: Optional[str] = None,
    teleop_control_echo: Optional[bool] = None,
    overwrite: bool = False,
) -> None:
    if (
        LOCOMOTION_CONFIG_PATH.is_file()
        and not overwrite
        and control_source is None
        and robot is None
        and checkpoint is None
        and teleop_control_echo is None
    ):
        print(f"[ok]  fleet locomotion config already present: {LOCOMOTION_CONFIG_PATH}")
        return
    cfg = load_config() if LOCOMOTION_CONFIG_PATH.is_file() else default_config()
    if control_source is not None:
        cfg["control_source"] = control_source
    if robot is not None:
        cfg["robot"] = robot
    if checkpoint is not None:
        cfg["checkpoint"] = checkpoint
    if checkpoint_host_dir is not None:
        cfg["checkpoint_host_dir"] = checkpoint_host_dir
    if teleop_control_echo is not None:
        cfg["teleop_control_echo"] = teleop_control_echo
    if cfg["control_source"] == "inference" and not cfg.get("checkpoint"):
        print(
            "[err] locomotion control_source=inference requires --locomotion-checkpoint at enroll",
            file=sys.stderr,
        )
        sys.exit(1)
    write_config(cfg)
    print(f"[+]   wrote fleet locomotion config {LOCOMOTION_CONFIG_PATH}")


def _opt_value(args: list[str], name: str) -> str | None:
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _has_flag(args: list[str], name: str) -> bool:
    return name in args


def build_hal_argv(extra_args: list[str]) -> list[str]:
    """Container argv for hal.server.jetson.main on fleet-enrolled hosts."""
    cfg = load_config()
    control = _opt_value(extra_args, "--control-source") or str(cfg["control_source"])
    robot = _opt_value(extra_args, "--robot") or str(cfg["robot"])
    teleop_ip = _opt_value(extra_args, "--teleop-ip") or str(cfg.get("teleop_ip") or DEFAULT_TELEOP_IP)
    checkpoint = _opt_value(extra_args, "--checkpoint") or cfg.get("checkpoint")
    if control == "inference" and not checkpoint:
        print(
            "[err] inference locomotion requires a checkpoint in locomotion.json or --checkpoint",
            file=sys.stderr,
        )
        sys.exit(1)

    argv = [
        "--control-source",
        control,
        "--teleop-ip",
        teleop_ip,
        "--robot",
        robot,
    ]
    if checkpoint:
        argv.extend(["--checkpoint", str(checkpoint)])

    echo_flag = "--teleop-control-echo"
    if _has_flag(extra_args, echo_flag) or bool(cfg.get("teleop_control_echo")):
        argv.append(echo_flag)

    skip = {"--control-source", "--robot", "--checkpoint", "--teleop-ip", echo_flag}
    passthrough: list[str] = []
    i = 0
    while i < len(extra_args):
        a = extra_args[i]
        if a in skip:
            i += 2 if a != echo_flag else 1
            continue
        passthrough.append(a)
        i += 1
    return argv + passthrough


def fleet_volume_mounts(cfg: dict[str, Any] | None = None) -> list[str]:
    """Host paths to mount for fleet HAL (-v src:dst pairs as flat list)."""
    cfg = cfg or load_config()
    home = _config_home()
    mounts: list[str] = []

    zed_root = cfg.get("zed_resources_host")
    if zed_root:
        res_host = Path(str(zed_root)).expanduser()
    else:
        res_host = home / "zed-resources" / "resources"
    settings_host = cfg.get("zed_settings_host")
    if settings_host:
        set_host = Path(str(settings_host)).expanduser()
    else:
        set_host = home / "zed-resources" / "settings"
    mounts.extend(["-v", f"{res_host}:/usr/local/zed/resources"])
    mounts.extend(["-v", f"{set_host}:/usr/local/zed/settings"])

    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        ckpt_path = Path(str(checkpoint))
        host_dir = cfg.get("checkpoint_host_dir")
        if host_dir:
            ck_host = Path(str(host_dir)).expanduser()
        else:
            ck_host = ckpt_path.parent if ckpt_path.is_absolute() else home / "checkpoints"
        mounts.extend(["-v", f"{ck_host}:/workspace/checkpoints"])
    return mounts


def request_locomotion_start() -> None:
    """Rate-limited systemctl start when teleop signaling arrives without HAL connected."""
    global _last_cold_start_attempt
    if not fleet_enrolled():
        return
    if not shutil_which("systemctl"):
        return
    now = time.monotonic()
    if now - _last_cold_start_attempt < _COLD_START_MIN_INTERVAL_S:
        return
    _last_cold_start_attempt = now
    subprocess.run(
        ["systemctl", "start", LOCOMOTION_SERVICE_NAME],
        check=False,
        capture_output=True,
    )


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)

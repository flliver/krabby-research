"""Docker command builders for the krabby CLI."""
from __future__ import annotations

import glob
import platform
import subprocess
import sys


def gpu_flags() -> list[str]:
    if platform.machine() == "aarch64":
        return ["--runtime=nvidia"]
    return ["--gpus", "all"]


def serial_device_flags() -> list[str]:
    flags: list[str] = []
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        for dev in sorted(glob.glob(pattern)):
            flags += ["--device", dev]
    return flags


def pull(image_ref: str) -> str:
    """Pull image and return its digest. Raises SystemExit on failure."""
    result = subprocess.run(
        ["docker", "pull", image_ref],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[err] docker pull failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    digest_result = subprocess.run(
        ["docker", "inspect", "--format={{index .RepoDigests 0}}", image_ref],
        capture_output=True,
        text=True,
    )
    return digest_result.stdout.strip()


def run_cmd(image_ref: str, extra_args: list[str], entrypoint: str | None = None) -> list[str]:
    ep = ["--entrypoint", entrypoint] if entrypoint else []
    return [
        "docker", "run", "--rm",
        "--name", "krabby",
        "--privileged",
        *gpu_flags(),
        *ep,
        "-v", "/dev:/dev",
        "-p", "6001:6001",
        "-p", "6002:6002",
        image_ref,
        *extra_args,
    ]


def firmware_cmd(image_ref: str, firmware_args: list[str]) -> list[str]:
    cache_mount = f"{_home()}/.cache/krabby-firmware:/root/.cache/krabby-firmware"
    return [
        "docker", "run", "--rm",
        *serial_device_flags(),
        "-v", cache_mount,
        "-e", "LD_PRELOAD=",
        "--entrypoint", "krabby-firmware",
        image_ref,
        *firmware_args,
    ]


def _home() -> str:
    from pathlib import Path
    return str(Path.home())

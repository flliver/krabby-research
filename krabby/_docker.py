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


def host_network_flags() -> list[str]:
    """`--network host` on Jetson, nothing on x86.

    Tegra kernels don't ship the iptables `raw` table that Docker's default
    bridge driver needs, so *any* bridge container fails at start with "can't
    initialize iptables table `raw'" (exit 125) — even one that publishes no
    ports (Docker programs the raw table for every bridge endpoint). Host
    networking sidesteps the bridge entirely. On x86 (testing) the bridge
    works, so add nothing. See docs/JETSON_DEPLOYMENT.md "Docker iptables /
    networking errors".
    """
    if platform.machine() == "aarch64":
        return ["--network", "host"]
    return []


def network_flags() -> list[str]:
    """Networking for the HAL server's ZMQ endpoints (6001/6002).

    On Jetson, host networking exposes the endpoints on the host directly and
    `-p` would be redundant. On x86 (testing) the bridge works, so publish the
    ports explicitly.
    """
    if host := host_network_flags():
        return host
    return ["-p", "6001:6001", "-p", "6002:6002"]


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


def run_cmd(image_ref: str, extra_args: list[str], entrypoint: str | None = None, extra_mounts: list[str] | None = None) -> list[str]:
    ep = ["--entrypoint", entrypoint] if entrypoint else []
    mounts: list[str] = []
    for m in (extra_mounts or []):
        mounts += ["-v", m]
    return [
        "docker", "run", "--rm",
        "--name", "krabby",
        "--privileged",
        *gpu_flags(),
        *ep,
        "-v", "/dev:/dev",
        *mounts,
        *network_flags(),
        image_ref,
        *extra_args,
    ]


def firmware_cmd(image_ref: str, firmware_args: list[str], interactive: bool = False) -> list[str]:
    # `interactive` allocates a TTY so the standalone interactive menu (`krabby
    # firmware` with no subcommand) and paged `show` output behave the same through
    # the wrapper as they do for host-side `krabby-firmware`.
    tty_flags = ["-it"] if interactive else []
    cache_mount = f"{_home()}/.cache/krabby-firmware:/root/.cache/krabby-firmware"
    return [
        "docker", "run", "--rm",
        *tty_flags,
        *host_network_flags(),
        *serial_device_flags(),
        "-v", cache_mount,
        "-e", "LD_PRELOAD=",
        "--entrypoint", "krabby-firmware",
        image_ref,
        *firmware_args,
    ]


# Robot topologies the server and client must agree on (joint counts must match).
_GAMEPAD_ROBOTS = ("hex", "go2")


def _opt_value(args: list[str], name: str) -> str | None:
    """Return the value of `--name VALUE` or `--name=VALUE` in args, else None."""
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _gamepad_launch_script(server_robot: str) -> str:
    """bash that brings up the HAL server + krabby-uno client/controller together.

    Mirrors controller/scripts/jetson/helper/run_server_and_client_in_container.sh so
    a single `krabby run` starts the whole stack. The trap is installed before the
    2s server warm-up so a Ctrl-C / `docker stop` during startup still tears both
    down; `wait -n`'s status is captured and re-exited so a crashed child fails
    `krabby run` instead of looking like a clean shutdown.
    """
    return (
        'MCU_PORT="${KRABBY_MCU_PORT:-/dev/ttyACM0}"; '
        'if [ ! -e "$MCU_PORT" ]; then '
        'echo "error: MCU device not found at $MCU_PORT — connect the Krabby HAL board '
        '(or set KRABBY_MCU_PORT)." >&2; exit 1; fi; '
        'krabby-hal-server-jetson --control-source gamepad' + server_robot + ' '
        '--observation-bind "tcp://*:6001" --command-bind "tcp://*:6002" & '
        'server_pid=$!; '
        'trap "kill $server_pid ${client_pid:-} 2>/dev/null; wait 2>/dev/null; exit 130" TERM INT; '
        'sleep 2; '
        'krabby-uno --observation_endpoint tcp://localhost:6001 '
        '--command_endpoint tcp://localhost:6002 "$@" & '
        'client_pid=$!; '
        'wait -n; status=$?; '
        'kill $server_pid $client_pid 2>/dev/null; wait 2>/dev/null; '
        'exit $status'
    )


def gamepad_cmd(image_ref: str, extra_args: list[str], extra_mounts: list[str] | None = None) -> list[str]:
    """docker run that launches the HAL server + krabby-uno client/controller together.

    extra_args are forwarded to krabby-uno (e.g. --device-id, --rate, --robot). A
    --robot value is also injected into the server command so one flag keeps both
    halves on the same joint topology.
    """
    robot = _opt_value(extra_args, "--robot")
    server_robot = f" --robot {robot}" if robot in _GAMEPAD_ROBOTS else ""
    mounts: list[str] = []
    for m in (extra_mounts or []):
        mounts += ["-v", m]
    return [
        "docker", "run", "--rm",
        "--name", "krabby",
        "--privileged",
        *gpu_flags(),
        "-v", "/dev:/dev",
        *mounts,
        *network_flags(),
        "--entrypoint", "bash",
        image_ref,
        "-c", _gamepad_launch_script(server_robot),
        "bash",          # becomes $0; krabby-uno args follow as "$@"
        *extra_args,
    ]


def _home() -> str:
    from pathlib import Path
    return str(Path.home())

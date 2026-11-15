"""Linux SocketCAN: virtual CLI SET_DUTY visible on a real vcan interface."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import pytest

from firmware.stm import codec, ids
from firmware.stm.mcu_can import KrabbyMCUCANSDK

REPO_ROOT = Path(__file__).resolve().parents[3]
VCAN_LIFECYCLE = REPO_ROOT / "firmware" / "stm" / "scripts" / "vcan0-lifecycle.sh"
VCAN_IFACE = "vcan_stm_pytest"

pytestmark = pytest.mark.socketcan

SET_DUTY_LENGTH = 36
_SET_DUTY_PAYLOAD = (500).to_bytes(2, "little", signed=True) + b"\x00" * 34


def _lifecycle_env() -> dict[str, str]:
    env = os.environ.copy()
    env["KRABBY_CAN_IFACE"] = VCAN_IFACE
    return env


@pytest.fixture(scope="module")
def vcan_up():
    if platform.system() != "Linux":
        pytest.skip("SocketCAN tests require Linux")
    if not VCAN_LIFECYCLE.is_file():
        pytest.skip(f"missing {VCAN_LIFECYCLE}")
    env = _lifecycle_env()
    try:
        subprocess.run(
            [str(VCAN_LIFECYCLE), "up"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or "") + (exc.stdout or "")
        pytest.skip(f"cannot create {VCAN_IFACE} (need vcan + ip link; often sudo): {err.strip()}")
    yield VCAN_IFACE
    subprocess.run(
        [str(VCAN_LIFECYCLE), "down"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_sdk_set_duty_tx_on_socketcan_vcan(vcan_up: str) -> None:
    import can

    rx = can.interface.Bus(
        channel=vcan_up,
        interface="socketcan",
        fd=True,
        bitrate=1_000_000,
        data_bitrate=5_000_000,
        receive_own_messages=True,
    )
    received: list[can.Message] = []
    notifier = can.Notifier(rx, [received.append])
    try:
        sdk = KrabbyMCUCANSDK(virtual=True, interfaces=(vcan_up,))
        sdk.connect()
        sdk.set_duty(1, 0.5, 0.0, 0.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not received:
            time.sleep(0.02)
        sdk.close()
    finally:
        notifier.stop()
        rx.shutdown()

    assert received, f"no frames on {vcan_up}"
    msg = next(m for m in received if m.arbitration_id == ids.set_duty_can_id())
    assert msg.is_fd is True
    assert msg.bitrate_switch is True
    # SocketCAN FD may pad msg.data to the next DLC bucket (e.g. 48); krabby.dbc SET_DUTY is 36 bytes.
    payload = bytes(msg.data)[:SET_DUTY_LENGTH]
    assert len(payload) == SET_DUTY_LENGTH
    assert payload == _SET_DUTY_PAYLOAD
    _, encoded = codec.encode_set_duty(
        codec.set_duty_values_from_grid(((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    )
    assert payload == encoded
#!/usr/bin/env python3
"""Hardware smoke test: InputController disconnect → reopen without restart.

Usage (from krabby-research, pad connected so js* exists):
  python3 controller/scripts/jetson/verify_joystick_reopen.py
  python3 controller/scripts/jetson/verify_joystick_reopen.py --wait 120

Flow:
  1. Open InputController and confirm stick samples.
  2. Wait for disconnect (idle sleep / unplug / --simulate).
  3. Wait for reopen after Home-wake or USB replug.
  Exit 0 only if both disconnect and reconnect are observed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verify_joystick_reopen")


def _stick_live(state) -> bool:
    return abs(state.LX) > 0.15 or abs(state.LY) > 0.15 or abs(state.RX) > 0.15 or abs(state.RY) > 0.15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait",
        type=float,
        default=180.0,
        help="Seconds to wait for disconnect and for reconnect (each phase)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="InputController update rate Hz",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Force close the open SDL controller to exercise reopen without unplugging",
    )
    args = parser.parse_args()

    from controller.input.input_controller import InputController

    devices = InputController.list_devices()
    if not devices:
        logger.error(
            "No controller-capable device. Wake the pad (Home) or plug USB, "
            "then confirm: ls /dev/input/js*"
        )
        return 1

    ic = InputController.get_instance()
    device_id = devices[0]["device_id"]
    name = devices[0]["name"]
    logger.info("Opening %s (device_id=%s)", name, device_id)
    ic.start(device_id=device_id, update_rate_hz=args.rate)
    time.sleep(0.3)
    if not ic._running or ic._controller is None:
        logger.error("Failed to open controller at start")
        ic.stop()
        return 1

    logger.info("Phase 1: move a stick within %.0fs (optional proof of life)", min(20.0, args.wait))
    deadline = time.time() + min(20.0, args.wait)
    saw_live = False
    while time.time() < deadline:
        if _stick_live(ic.get_state()):
            saw_live = True
            break
        time.sleep(0.05)
    if saw_live:
        logger.info("Stick motion seen — controller is live")
    else:
        logger.warning("No stick motion yet; continuing (pad may be idle-centered)")

    if args.simulate:
        logger.info("Phase 2: --simulate closing SDL handle (expect auto-reopen if js* still present)")
        ic._mark_disconnected("simulate")
    else:
        logger.info(
            "Phase 2: disconnect the pad now — leave idle until LEDs out, or unplug USB "
            "(timeout %.0fs)",
            args.wait,
        )

    deadline = time.time() + args.wait
    disconnected = args.simulate
    while time.time() < deadline and not disconnected:
        if ic._controller is None or not ic._controller_still_attached():
            # Give the event loop a moment to process attached()→mark_disconnected
            time.sleep(0.2)
            if ic._controller is None:
                disconnected = True
                break
        time.sleep(0.05)

    if not disconnected:
        logger.error("Timeout waiting for disconnect")
        ic.stop()
        return 1
    logger.info("Disconnect observed (controller handle closed / not attached)")

    logger.info(
        "Phase 3: wake with Home or replug USB; waiting up to %.0fs for reopen",
        args.wait,
    )
    deadline = time.time() + args.wait
    reconnected = False
    while time.time() < deadline:
        if ic._controller is not None and ic._controller_still_attached():
            reconnected = True
            break
        time.sleep(0.05)

    if not reconnected:
        logger.error("Timeout waiting for reopen — FAILED")
        ic.stop()
        return 1

    name = "Unknown"
    try:
        import pygame._sdl2.controller as sdl2_controller

        if ic._device_id is not None:
            name = sdl2_controller.name_forindex(ic._device_id) or "Unknown"
    except Exception:
        pass
    logger.info("Reconnected: %s (device_id=%s) — PASSED", name, ic._device_id)
    ic.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

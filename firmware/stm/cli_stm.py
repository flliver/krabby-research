"""Subcommand handlers for krabby-firmware-stm."""

from __future__ import annotations

import argparse
import json
import sys
import time

from firmware.stm import ids
from firmware.stm.mcu_can import KrabbyMCUCANSDK


def _sdk_from_args(args: argparse.Namespace) -> KrabbyMCUCANSDK:
    return KrabbyMCUCANSDK(virtual=args.virtual)


def cmd_duty(args: argparse.Namespace) -> int:
    sdk = _sdk_from_args(args)
    try:
        sdk.connect()
        sdk.set_duty(args.leg_id, args.y, args.h, args.k)
    finally:
        sdk.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.joint not in (2, 3):
        print("joint must be 2 (hip) or 3 (knee)", file=sys.stderr)
        return 2
    sdk = _sdk_from_args(args)
    try:
        sdk.connect()
        sdk.send_calibration(args.leg_id, args.joint)
    finally:
        sdk.close()
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    sdk = _sdk_from_args(args)
    try:
        sdk.connect()
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = {
            "telemetry": sdk.telemetry,
            "board_info": sdk.board_info,
        }
        if args.json:
            print(json.dumps(snapshot, sort_keys=True, default=str))
        else:
            for leg_id in ids.LEG_IDS:
                fields = snapshot["telemetry"].get(leg_id) or {}
                if not fields:
                    continue
                j1_duty = fields.get("J1_DUTY", "")
                j2_duty = fields.get("J2_DUTY", "")
                j3_duty = fields.get("J3_DUTY", "")
                print(f"leg {leg_id}: J1_DUTY={j1_duty} J2_DUTY={j2_duty} J3_DUTY={j3_duty}")
            for leg_id in ids.LEG_IDS:
                info = snapshot["board_info"].get(leg_id) or {}
                if not info:
                    continue
                print(
                    f"leg {leg_id} info: "
                    f"{info.get('major')}.{info.get('minor')}.{info.get('patch')} "
                    f"commit={info.get('commit')}"
                )
    finally:
        sdk.close()
    return 0


_COMMANDS = {
    "duty": cmd_duty,
    "calibrate": cmd_calibrate,
    "telemetry": cmd_telemetry,
}
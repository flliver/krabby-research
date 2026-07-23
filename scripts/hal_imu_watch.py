#!/usr/bin/env python3
"""Watch live IMU state flowing through the HAL observation channel.

Subscribes to a HAL server's observation socket and prints, per sample, the
raw ``HardwareObservations.base_ang_vel_b`` / ``base_quat_w`` alongside the
parkour mapper's roll/pitch (``proprioceptive[3:5]``) — the exact values the
policy sees. Tilt the ZED and watch roll/pitch follow; pipe through ``tee``
to capture the bench log.

Run against a HAL server with a TCP observation bind, e.g. the locomotion
container started with:

    python3 -m hal.server.jetson.main --control-source portal \
        --observation-bind 'tcp://*:6001' --command-bind 'tcp://*:6002'

then (inside the container or anywhere that can reach the bind):

    python3 scripts/hal_imu_watch.py [--endpoint tcp://localhost:6001] \
        [--hz 5] [--samples 0]
"""

import argparse
import sys
import time

import numpy as np
import zmq

from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.client.data_structures.hardware import HardwareObservations
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

OBSERVATION_TOPIC = b"observation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--endpoint",
        type=str,
        default="tcp://localhost:6001",
        help="HAL observation endpoint (default: tcp://localhost:6001)",
    )
    parser.add_argument("--hz", type=float, default=5.0, help="print rate")
    parser.add_argument(
        "--samples", type=int, default=0, help="samples to print (0 = until Ctrl-C)"
    )
    args = parser.parse_args()

    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )
    mapper = HWObservationsToParkourMapper(observation_dimensions)

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, OBSERVATION_TOPIC)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(args.endpoint)
    print(f"Subscribed to {args.endpoint}; waiting for observations...")

    print(
        f"{'ang_vel_b rad/s':>28}{'base_quat_w xyzw (|q|~1)':>40}"
        f"{'mapper roll/pitch deg':>24}"
    )
    period = 1.0 / args.hz
    printed = 0
    try:
        while args.samples <= 0 or printed < args.samples:
            if not sub.poll(2000, zmq.POLLIN):
                print("  (no observation within 2 s — is the HAL server up?)")
                continue
            frame = sub.recv()
            if not frame.startswith(OBSERVATION_TOPIC):
                continue
            hw_obs = HardwareObservations.from_bytes(frame[len(OBSERVATION_TOPIC):])
            proprioceptive = mapper.map(hw_obs).to_array()
            ang = hw_obs.base_ang_vel_b
            quat = hw_obs.base_quat_w
            roll_deg = np.degrees(proprioceptive[3])
            pitch_deg = np.degrees(proprioceptive[4])
            print(
                f"  [{ang[0]:7.3f} {ang[1]:7.3f} {ang[2]:7.3f}]"
                f"   [{quat[0]:6.3f} {quat[1]:6.3f} {quat[2]:6.3f} {quat[3]:6.3f}]"
                f" |{np.linalg.norm(quat):5.3f}|"
                f"   {roll_deg:8.1f} {pitch_deg:8.1f}",
                flush=True,
            )
            printed += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        context.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())

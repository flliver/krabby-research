#!/usr/bin/env python3
"""One-off ZED IMU probe: verify pyzed API names and units on the Orin.

Run on the Jetson with the ZED 2i plugged in with a USBA-to-USBC USB3.0 cable.
This cable is required.

    python3 scripts/zed_imu_probe.py [--samples 20] [--hz 10]

What to check in the output:
- Every API-name check prints OK (get_sensors_data, get_imu_data,
  get_angular_velocity, get_linear_acceleration, get_pose().get_orientation(),
  timestamp.get_nanoseconds). Sensors are read with TIME_REFERENCE.CURRENT:
  IMAGE-referenced data freezes (non-unit quat) unless grab() runs, verified
  on SDK 5.4.0.
- Camera at rest: linear acceleration magnitude ~9.8 m/s² (gravity), angular
  velocity near zero. Rotate the camera slowly by hand: angular velocity
  should read in the tens of deg/s (pyzed reports deg/s — the HAL wrapper
  converts to rad/s).
- Quaternion is unit-norm and drifts smoothly while tilting.
- roll/pitch (computed from the quat exactly like the parkour mapper's
  proprioceptive[3:5], identity mount) track hand tilts — tilt ±45° and watch
  them follow (AC 5f). Pipe through `tee` to capture the bench log; output
  switches to one line per sample when stdout is not a terminal.
- Timestamps advance between samples.

If the ZedCamera HAL wrapper is importable, the probe also exercises
ZedCamera.get_imu() end to end and prints one converted (rad/s) sample.
"""

import argparse
import shutil
import sys
import time

import numpy as np


def roll_pitch_from_quat_xyzw(quat_xyzw: np.ndarray) -> tuple[float, float]:
    """Roll/pitch (rad) from an (x, y, z, w) quat — same XYZ-euler math as
    compute/parkour/utils/math.py:euler_xyz_from_quat, which the mapper feeds
    into proprioceptive[3:5]. Identity mount ⇒ camera quat == base_quat_w."""
    x, y, z, w = quat_xyzw
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def check(label: str, fn):
    """Run one API probe; print OK/FAIL and return the value (None on failure)."""
    try:
        value = fn()
        print(f"  OK   {label}")
        return value
    except AttributeError as e:
        print(f"  FAIL {label}: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=20, help="IMU samples to print")
    parser.add_argument("--hz", type=float, default=10.0, help="print rate")
    args = parser.parse_args()

    try:
        import pyzed.sl as sl
    except ImportError as e:
        print(f"pyzed import failed: {e}", file=sys.stderr)
        return 1

    sdk_version = getattr(sl.Camera, "get_sdk_version", lambda: "unknown")()
    print(f"pyzed imported; SDK version: {sdk_version}")

    camera = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"camera.open failed: {status}", file=sys.stderr)
        return 1
    print(f"Camera open: {camera.get_camera_information().camera_model}")

    print("\nAPI name checks (AC 5b):")
    sensors_data = check("sl.SensorsData()", sl.SensorsData)
    check("sl.TIME_REFERENCE.CURRENT", lambda: sl.TIME_REFERENCE.CURRENT)
    check(
        "camera.get_sensors_data(sensors_data, TIME_REFERENCE.CURRENT)",
        lambda: camera.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT),
    )
    imu = check("sensors_data.get_imu_data()", sensors_data.get_imu_data)
    if imu is None:
        print("cannot continue without imu data", file=sys.stderr)
        camera.close()
        return 1
    check("imu.get_angular_velocity()", imu.get_angular_velocity)
    check("imu.get_linear_acceleration()", imu.get_linear_acceleration)
    check("imu.get_pose().get_orientation().get()", lambda: imu.get_pose().get_orientation().get())
    check("imu.timestamp.get_nanoseconds()", lambda: imu.timestamp.get_nanoseconds())

    print(f"\n{args.samples} samples (rest the camera, then tilt it by hand):")
    # Field widths match the body line segments below so headers sit over
    # their columns.
    print(
        f"{'ang_vel deg/s':>27}{'lin_acc m/s² (|g|~9.8)':>33}{'quat xyzw (|q|~1)':>40}"
        f"{'roll/pitch deg':>18}{'ts advance':>12}"
    )
    is_tty = sys.stdout.isatty()
    last_ts = None
    period = 1.0 / args.hz
    for _ in range(args.samples):
        if camera.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) != sl.ERROR_CODE.SUCCESS:
            print("\n  get_sensors_data returned non-SUCCESS")
            time.sleep(period)
            continue
        imu = sensors_data.get_imu_data()
        ang = np.asarray(imu.get_angular_velocity())
        acc = np.asarray(imu.get_linear_acceleration())
        quat = np.asarray(imu.get_pose().get_orientation().get())
        roll, pitch = roll_pitch_from_quat_xyzw(quat)
        try:
            ts = imu.timestamp.get_nanoseconds()
        except AttributeError:
            ts = None
        advanced = "n/a" if ts is None or last_ts is None else ("yes" if ts > last_ts else "NO")
        last_ts = ts
        line = (
            f"  [{ang[0]:7.2f} {ang[1]:7.2f} {ang[2]:7.2f}]"
            f"   [{acc[0]:6.2f} {acc[1]:6.2f} {acc[2]:6.2f}] |{np.linalg.norm(acc):5.2f}|"
            f"   [{quat[0]:6.3f} {quat[1]:6.3f} {quat[2]:6.3f} {quat[3]:6.3f}] |{np.linalg.norm(quat):5.3f}|"
            f"   {np.degrees(roll):7.1f} {np.degrees(pitch):7.1f}"
            f"   {advanced:>9}"
        )
        if is_tty:
            # If the line wraps, \r only returns to the start of the last wrapped
            # row and the header scrolls away — truncate to the terminal width.
            columns = shutil.get_terminal_size().columns
            print(f"\r{line[:columns - 1]}\x1b[K", end="", flush=True)
        else:
            # Piped/teed (bench-log capture): one full line per sample.
            print(line, flush=True)
        time.sleep(period)
    print()

    camera.close()

    print("\nHAL wrapper path (ZedCamera.get_imu):")
    try:
        from hal.server.jetson.zed_camera import ZedCamera
    except ImportError as e:
        print(f"  skipped (hal.server.jetson not importable: {e})")
        return 0
    cam = ZedCamera(depth_mode="NEURAL", resolution=(640, 480), fps=30)
    try:
        sample = cam.get_imu()
        if sample is None:
            print("  get_imu() returned None (fetch failed)")
            return 1
        print(f"  ang_vel_rad_s        = {sample.ang_vel_rad_s}")
        print(f"  lin_acc_m_s2         = {sample.lin_acc_m_s2}")
        print(f"  orientation_quat_xyzw= {sample.orientation_quat_xyzw}")
        print(f"  timestamp_ns         = {sample.timestamp_ns}")
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Host leader telemetry includes a parseable IMU segment (TelemetryFrame / imu_telemetry)."""

from firmware.interfaces.telemetry_frame import TelemetryFrame

# Matches default SimImuBodyState + appendImuMeasurement formatting on host FRONT line.
HOST_STATIONARY_IMU_SUFFIX = (
    ";IMU 0.000 0.000 9.806 0.0000 0.0000 0.0000 25.0 1"
)


def test_host_stationary_imu_segment_parses():
    line = (
        "FRONT; FLHY 0.500 500 12 0 0 0 0 0 0"
        + HOST_STATIONARY_IMU_SUFFIX
    )
    frame = TelemetryFrame.parse_line(line)
    assert frame.imu is not None
    assert frame.imu.valid == 1
    assert frame.imu.accel[2] == 9.806
    assert frame.imu.temp_c == 25.0


def test_host_imu_segment_joint_telemetry_unchanged():
    line = (
        "FRONT; FLHY 0.500 500 12 0 0 0 0 0 0"
        + HOST_STATIONARY_IMU_SUFFIX
    )
    frame = TelemetryFrame.parse_line(line)
    assert len(frame.joints) == 1
    assert frame.joints[0].name == "FLHY"

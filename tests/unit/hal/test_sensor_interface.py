"""Unit tests for ``SensorInterface``, Jetson catalog, and ``build_pipeline`` string shape."""

import pytest

from hal.server.isaac.sensor_backend_isaac import IsaacSensorInterface
from hal.server.jetson.front_camera_factory import FRONT_RGB_DEPTH_CAMERA_FACTORIES
from hal.server.jetson.sensor_backend_jetson import (
    JETSON_SENSOR_CATALOG,
    JetsonSensorInterface,
    front_observation_camera_catalog_entry,
)
from hal.server.sensor_interface import GStreamerHandle, SensorInfo, SensorPose

from tests.unit.hal.isaac_sensor_fixtures import ISAAC_CONFIGURED_SENSORS_FIXTURE


def test_jetson_catalog_single_primary_front_rgbd():
    primaries = [e for e in JETSON_SENSOR_CATALOG if e.is_primary]
    assert len(primaries) == 1
    entry = primaries[0]
    assert entry.id == "front_rgbd"
    assert entry.camera_driver == "zed"
    assert entry.camera_driver in FRONT_RGB_DEPTH_CAMERA_FACTORIES


def test_front_observation_camera_catalog_entry():
    e = front_observation_camera_catalog_entry()
    assert e.id == "front_rgbd"
    assert e.camera_driver == "zed"
    assert e.camera_driver in FRONT_RGB_DEPTH_CAMERA_FACTORIES


def test_jetson_catalog_has_front_and_side_rgbd_only():
    ids = {e.id for e in JETSON_SENSOR_CATALOG}
    assert ids == {"front_rgbd", "side_rgbd"}
    assert all(e.type == "rgbd" for e in JETSON_SENSOR_CATALOG)
    assert not any(e.type == "radar" for e in JETSON_SENSOR_CATALOG)


def test_jetson_list_sensors_matches_catalog():
    iface = JetsonSensorInterface()
    sensors = iface.list_sensors()
    n_depth = sum(1 for e in JETSON_SENSOR_CATALOG if e.gst_depth_quant_range_m is not None)
    assert len(sensors) == len(JETSON_SENSOR_CATALOG) + n_depth
    ids = [s.id for s in sensors]
    for e in JETSON_SENSOR_CATALOG:
        assert e.id in ids
        if e.gst_depth_quant_range_m is not None:
            assert f"{e.id}_gray16_depth" in ids


def test_jetson_get_gstreamer_handle_is_pure_transform():
    iface = JetsonSensorInterface()
    listed = iface.list_sensors()[0]

    h0 = iface.get_gstreamer_handle(listed)
    assert h0.resolution == listed.resolution

    custom = SensorInfo(
        id="front_rgbd",
        type="rgbd",
        modality="rgbd",
        resolution=(320, 240),
        fps=listed.fps,
        pose=listed.pose,
        camera_driver=listed.camera_driver,
    )
    h = iface.get_gstreamer_handle(custom)
    assert h.resolution == (320, 240)

    arbitrary = SensorInfo(
        id="nonexistent",
        type="rgbd",
        modality="rgbd",
        resolution=(640, 480),
        fps=30,
        pose=SensorPose(x=0, y=0, z=0, qx=0, qy=0, qz=0, qw=1),
        camera_driver=None,
    )
    h2 = iface.get_gstreamer_handle(arbitrary)
    assert h2.sensor_id == "nonexistent"
    assert h2.appsrc_pixel_format == "RGB"


def test_jetson_build_pipeline_h264_fakesink_nvenc_shape():
    iface = JetsonSensorInterface(use_nvenc=True)
    sensor = iface.list_sensors()[0]
    handle = iface.get_gstreamer_handle(sensor)
    assert handle.appsrc_pixel_format == "RGB"

    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")

    assert "appsrc name=src" in pipe
    w, h = handle.resolution
    assert f"video/x-raw,format=RGB,width={w},height={h},framerate={handle.fps}/1" in pipe
    assert "nvvidconv" in pipe
    assert "nvv4l2h264enc" in pipe
    assert "h264parse" in pipe
    assert pipe.rstrip().endswith("fakesink")


def test_jetson_build_pipeline_h264_fakesink_software_encode_shape():
    iface = JetsonSensorInterface(use_nvenc=False)
    sensor = iface.list_sensors()[0]
    handle = iface.get_gstreamer_handle(sensor)

    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")

    assert "appsrc name=src" in pipe
    w, h = handle.resolution
    assert f"video/x-raw,format=RGB,width={w},height={h},framerate={handle.fps}/1" in pipe
    assert "x264enc" in pipe
    assert "h264parse" in pipe
    assert pipe.rstrip().endswith("fakesink")


def test_jetson_build_pipeline_raw_shape():
    iface = JetsonSensorInterface()
    sensor = iface.list_sensors()[0]
    handle = iface.get_gstreamer_handle(sensor)
    pipe = iface.build_pipeline(handle, encoding="raw", output_element="fakesink")

    assert "appsrc name=src" in pipe
    assert "videoconvert" in pipe
    assert pipe.rstrip().endswith("fakesink")


def test_jetson_depth_gray16_handle_and_sw_pipeline_shape():
    iface = JetsonSensorInterface(use_nvenc=False)
    depth = next(s for s in iface.list_sensors() if s.id == "front_rgbd_gray16_depth")
    handle = iface.get_gstreamer_handle(depth)
    assert handle.appsrc_pixel_format == "GRAY16_LE"
    assert handle.depth_range_m == (0.2, 25.0)

    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")
    assert "appsrc name=src" in pipe
    w, h = handle.resolution
    assert (
        f"video/x-raw,format=GRAY16_LE,width={w},height={h},framerate={handle.fps}/1"
        in pipe
    )
    assert "videoconvert" in pipe
    assert "x264enc" in pipe
    assert pipe.rstrip().endswith("fakesink")


def test_jetson_depth_gray16_nvenc_pipeline_shape():
    iface = JetsonSensorInterface(use_nvenc=True)
    depth = next(s for s in iface.list_sensors() if s.id == "side_rgbd_gray16_depth")
    handle = iface.get_gstreamer_handle(depth)
    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")
    assert "format=GRAY16_LE" in pipe
    assert "videoconvert" in pipe
    assert "nvvidconv" in pipe
    assert "nvv4l2h264enc" in pipe


def test_isaac_list_and_handle_and_pipeline_shape():
    iface = IsaacSensorInterface(configured_sensors=ISAAC_CONFIGURED_SENSORS_FIXTURE)
    sensors = iface.list_sensors()
    assert len(sensors) == len(ISAAC_CONFIGURED_SENSORS_FIXTURE)

    front = next(s for s in sensors if s.id == "front_rgbd")
    handle = iface.get_gstreamer_handle(front)
    assert handle.appsrc_pixel_format == "RGB"

    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")
    assert "appsrc name=src" in pipe
    assert "video/x-raw,format=RGB" in pipe
    assert "x264enc" in pipe
    assert "h264parse" in pipe
    assert pipe.rstrip().endswith("fakesink")


def test_isaac_radar_handle_uses_gray8_in_pipeline():
    iface = IsaacSensorInterface(configured_sensors=ISAAC_CONFIGURED_SENSORS_FIXTURE)
    radar = next(s for s in iface.list_sensors() if s.id == "radar_front")
    handle = iface.get_gstreamer_handle(radar)
    assert handle.appsrc_pixel_format == "GRAY8"

    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")
    assert "format=GRAY8" in pipe


def test_isaac_depth_gray16_handle_and_pipeline_shape():
    iface = IsaacSensorInterface(configured_sensors=ISAAC_CONFIGURED_SENSORS_FIXTURE)
    depth = next(s for s in iface.list_sensors() if s.id == "front_rgbd_gray16_depth")
    handle = iface.get_gstreamer_handle(depth)
    assert handle.appsrc_pixel_format == "GRAY16_LE"
    assert handle.depth_range_m == (0.2, 2.0)
    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink")
    assert "format=GRAY16_LE" in pipe
    assert "x264enc" in pipe


def test_gstreamer_handle_manual_build_pipeline_jetson():
    """``build_pipeline`` uses handle fields only (no id/catalog relookup)."""
    iface = JetsonSensorInterface(use_nvenc=False)
    sensor = iface.list_sensors()[0]
    canon = iface.get_gstreamer_handle(sensor)

    spoof_handle = GStreamerHandle(
        sensor_id=sensor.id,
        sensor_type=sensor.type,
        modality=sensor.modality,
        resolution=(320, 240),
        fps=15,
        camera_driver=sensor.camera_driver,
        appsrc_pixel_format="RGB",
        backend_data={"use_nvenc": False},
    )
    pipe = iface.build_pipeline(spoof_handle, encoding="h264", output_element="fakesink")
    assert "width=320,height=240,framerate=15/1" in pipe

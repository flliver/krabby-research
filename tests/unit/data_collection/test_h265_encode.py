"""Unit tests for H.265 encoder pipeline strings (no GStreamer required)."""

from __future__ import annotations

from data_collection.h265_encode import build_h265_appsrc_appsink_pipeline


def test_sw_pipeline_uses_x265() -> None:
    desc = build_h265_appsrc_appsink_pipeline(
        100, 100, 20, format_caps="GRAY8", use_nvenc=False
    )
    assert "appsrc name=src" in desc
    assert "format=GRAY8" in desc
    assert "width=100" in desc
    assert "x265enc" in desc
    assert "h265parse" in desc
    assert "appsink name=sink" in desc
    assert "nvv4l2h265enc" not in desc
    assert "nvvidconv" not in desc


def test_nvenc_pipeline_forces_system_nv12_before_nvvidconv() -> None:
    desc = build_h265_appsrc_appsink_pipeline(
        672, 376, 10, format_caps="GRAY8", use_nvenc=True
    )
    assert "format=GRAY8" in desc
    assert "videoconvert ! video/x-raw,format=NV12 ! nvvidconv" in desc
    assert "memory:NVMM),format=NV12" in desc
    assert "nvv4l2h265enc" in desc
    assert "h265parse" in desc
    assert "x265enc" not in desc


def test_nvenc_rgb_pipeline_same_nv12_bridge() -> None:
    desc = build_h265_appsrc_appsink_pipeline(
        640, 360, 10, format_caps="RGB", use_nvenc=True
    )
    assert "format=RGB" in desc
    assert "videoconvert ! video/x-raw,format=NV12 ! nvvidconv" in desc
    assert "nvv4l2h265enc" in desc

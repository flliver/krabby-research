"""Front RGB-D driver registry and ``create_front_rgb_depth_camera``."""

import pytest

from hal.server.jetson.front_camera_factory import (
    FRONT_RGB_DEPTH_CAMERA_FACTORIES,
    create_front_rgb_depth_camera,
)


def test_front_rgb_depth_factories_include_zed_maixsense_sen0581():
    assert set(FRONT_RGB_DEPTH_CAMERA_FACTORIES) >= {
        "zed",
        "maixsense_a075v",
        "sen0581",
    }


def test_create_front_unknown_driver_raises():
    with pytest.raises(ValueError, match="Unknown front"):
        create_front_rgb_depth_camera(
            "nope",
            resolution=(64, 48),
            fps=30,
            depth_mode="NEURAL",
        )


def test_create_front_maixsense_requires_literal_host():
    with pytest.raises(ValueError, match="maixsense_host"):
        create_front_rgb_depth_camera(
            "maixsense_a075v",
            resolution=(64, 48),
            fps=30,
            depth_mode="NEURAL",
        )
    with pytest.raises(ValueError, match="maixsense_host"):
        create_front_rgb_depth_camera(
            "maixsense_a075v",
            resolution=(64, 48),
            fps=30,
            depth_mode="NEURAL",
            maixsense_host="   ",
        )


def test_create_front_sen0581_requires_device():
    with pytest.raises(ValueError, match="sen0581_device"):
        create_front_rgb_depth_camera(
            "sen0581",
            resolution=(100, 100),
            fps=20,
            depth_mode="TOF",
        )
    with pytest.raises(ValueError, match="sen0581_device"):
        create_front_rgb_depth_camera(
            "sen0581",
            resolution=(100, 100),
            fps=20,
            depth_mode="TOF",
            sen0581_device="   ",
        )

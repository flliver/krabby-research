"""Front RGB-D driver registry and ``create_front_rgb_depth_camera``."""

import pytest

from hal.server.jetson.front_camera_factory import (
    FRONT_RGB_DEPTH_CAMERA_FACTORIES,
    create_front_rgb_depth_camera,
)


def test_front_rgb_depth_factories_include_zed_and_maixsense():
    assert set(FRONT_RGB_DEPTH_CAMERA_FACTORIES) >= {"zed", "maixsense_a075v"}


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

"""Unit tests for HardwareObservations serialization/deserialization."""

import numpy as np
import pytest

from hal.client.data_structures.hardware import HardwareObservations, RgbdCatalogObservation


def test_serialize_deserialize_with_privileged_latent():
    """Test that privileged_latent is correctly serialized and deserialized."""
    # Create test data
    privileged_latent = np.array([6.921, 1.234, 0.567] + [0.0] * 26, dtype=np.float32)  # 29 values
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    
    # Create HardwareObservations with privileged_latent
    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        scan_features=scan_features,
        privileged_latent=privileged_latent,
    )
    
    blob = hw_obs.to_bytes()
    hw_obs_deserialized = HardwareObservations.from_bytes(blob)
    
    # Verify privileged_latent is preserved
    assert hw_obs_deserialized.privileged_latent is not None, "privileged_latent is None after deserialization"
    assert hw_obs_deserialized.privileged_latent.shape == (29,), f"Expected shape (29,), got {hw_obs_deserialized.privileged_latent.shape}"
    assert np.allclose(
        hw_obs_deserialized.privileged_latent, privileged_latent
    ), f"privileged_latent mismatch: expected {privileged_latent}, got {hw_obs_deserialized.privileged_latent}"
    
    # Verify first value matches
    assert np.isclose(
        hw_obs_deserialized.privileged_latent[0], 6.921
    ), f"First value mismatch: expected 6.921, got {hw_obs_deserialized.privileged_latent[0]}"


def test_serialize_deserialize_without_privileged_latent():
    """Test that serialization works when privileged_latent is None (hardware case)."""
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    
    # Create HardwareObservations without privileged_latent
    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        scan_features=scan_features,
        privileged_latent=None,  # Not available on hardware
    )
    
    blob = hw_obs.to_bytes()
    hw_obs_deserialized = HardwareObservations.from_bytes(blob)

    # Verify privileged_latent is None
    assert hw_obs_deserialized.privileged_latent is None, "privileged_latent should be None after deserialization"


def test_serialize_deserialize_roundtrip():
    """Test full roundtrip: serialize -> deserialize -> serialize -> deserialize."""
    privileged_latent = np.array([6.921, 1.234, 0.567] + [0.0] * 26, dtype=np.float32)
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    
    # Create original
    hw_obs_original = HardwareObservations(
        joint_positions=np.random.randn(12).astype(np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.random.randn(3).astype(np.float32),
        base_lin_vel_b=np.random.randn(3).astype(np.float32),
        base_quat_w=np.random.randn(4).astype(np.float32),
        joint_velocities=np.random.randn(12).astype(np.float32),
        contact_forces=np.random.randn(5).astype(np.float32),
        previous_action=np.random.randn(12).astype(np.float32),
        scan_features=scan_features,
        privileged_latent=privileged_latent,
    )
    
    blob1 = hw_obs_original.to_bytes()
    hw_obs_round1 = HardwareObservations.from_bytes(blob1)

    blob2 = hw_obs_round1.to_bytes()
    hw_obs_round2 = HardwareObservations.from_bytes(blob2)
    
    # Verify privileged_latent is preserved through both roundtrips
    assert hw_obs_round2.privileged_latent is not None, "privileged_latent lost in roundtrip"
    assert np.allclose(
        hw_obs_round2.privileged_latent, privileged_latent
    ), f"privileged_latent mismatch after roundtrip: expected {privileged_latent}, got {hw_obs_round2.privileged_latent}"


def test_serialize_deserialize_with_camera_rgb_depth_only():
    """Test that camera_rgb and camera_depth are serialized/deserialized in single blob."""
    camera_height, camera_width = 480, 640
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0

    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
    )

    blob = hw_obs.to_bytes()
    hw_obs_deserialized = HardwareObservations.from_bytes(blob)
    assert hw_obs_deserialized.camera_rgb is not None
    assert hw_obs_deserialized.camera_depth is not None
    assert np.array_equal(hw_obs_deserialized.camera_rgb, camera_rgb)
    assert np.allclose(hw_obs_deserialized.camera_depth, camera_depth)


def test_serialize_deserialize_with_scan_and_camera():
    """Test blob: 8 base + scan_features + camera_rgb + camera_depth (no privileged_latent)."""
    camera_height, camera_width = 480, 640
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0

    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        scan_features=scan_features,
        privileged_latent=None,
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
    )

    blob = hw_obs.to_bytes()
    hw_obs_deserialized = HardwareObservations.from_bytes(blob)
    assert hw_obs_deserialized.scan_features is not None
    assert hw_obs_deserialized.privileged_latent is None
    assert np.array_equal(hw_obs_deserialized.camera_rgb, camera_rgb)
    assert np.allclose(hw_obs_deserialized.camera_depth, camera_depth)


def test_serialize_deserialize_with_scan_priv_and_camera():
    """Test blob: 8 base + scan_features + privileged_latent + camera_rgb + camera_depth."""
    camera_height, camera_width = 480, 640
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    privileged_latent = np.array([1.0, 2.0, 3.0] + [0.0] * 26, dtype=np.float32)
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0

    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        scan_features=scan_features,
        privileged_latent=privileged_latent,
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
    )

    blob = hw_obs.to_bytes()
    hw_obs_deserialized = HardwareObservations.from_bytes(blob)
    assert hw_obs_deserialized.privileged_latent is not None
    assert np.array_equal(hw_obs_deserialized.camera_rgb, camera_rgb)
    assert np.allclose(hw_obs_deserialized.camera_depth, camera_depth)


def test_roundtrip_with_camera_data():
    """Test full roundtrip with camera_rgb and camera_depth."""
    camera_height, camera_width = 480, 640
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0

    hw_obs_original = HardwareObservations(
        joint_positions=np.random.randn(12).astype(np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
    )

    blob = hw_obs_original.to_bytes()
    hw_obs_round = HardwareObservations.from_bytes(blob)
    assert np.array_equal(hw_obs_round.camera_rgb, camera_rgb)
    assert np.allclose(hw_obs_round.camera_depth, camera_depth)


def test_camera_rgb_depth_both_or_none():
    """Test that camera_rgb and camera_depth must be both set or both None."""
    camera_height, camera_width = 480, 640
    camera_rgb = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="camera_rgb and camera_depth must be both set or both None"):
        HardwareObservations(
            joint_positions=np.zeros(12, dtype=np.float32),
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=0,
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
            joint_velocities=np.zeros(12, dtype=np.float32),
            contact_forces=np.zeros(5, dtype=np.float32),
            previous_action=np.zeros(12, dtype=np.float32),
            camera_rgb=camera_rgb,
            camera_depth=None,
        )


def test_side_scan_and_rgb_depth_roundtrip():
    """Optional legacy side_* fields serialize after front camera (depth features + RGB-D)."""
    camera_height, camera_width = 480, 640
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0
    side_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    side_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0
    scan_f = np.linspace(-0.5, 0.5, 132, dtype=np.float32)
    side_scan = np.linspace(0.1, 0.9, 132, dtype=np.float32)

    hw = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=1,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
        scan_features=scan_f,
        side_scan_features=side_scan,
        side_camera_rgb=side_rgb,
        side_camera_depth=side_depth,
    )
    out = HardwareObservations.from_bytes(hw.to_bytes())
    assert np.allclose(out.scan_features, scan_f)
    assert np.allclose(out.side_scan_features, side_scan)
    assert np.array_equal(out.side_camera_rgb, side_rgb)
    assert np.allclose(out.side_camera_depth, side_depth)
    assert out.rgbd_by_catalog_id is None


def test_side_camera_different_resolution_than_front_roundtrip():
    """Legacy side_camera_* may use Hs×Ws different from front camera_height×camera_width."""
    camera_height, camera_width = 480, 640
    sh, sw = 240, 320
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0
    side_rgb = np.random.randint(0, 255, (sh, sw, 3), dtype=np.uint8)
    side_depth = np.random.rand(sh, sw).astype(np.float32) * 5.0
    scan_f = np.linspace(-0.5, 0.5, 132, dtype=np.float32)
    side_scan = np.linspace(0.1, 0.9, 132, dtype=np.float32)

    hw = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=1,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
        scan_features=scan_f,
        side_scan_features=side_scan,
        side_camera_rgb=side_rgb,
        side_camera_depth=side_depth,
    )
    out = HardwareObservations.from_bytes(hw.to_bytes())
    assert out.camera_height == camera_height and out.camera_width == camera_width
    assert np.array_equal(out.side_camera_rgb, side_rgb)
    assert out.side_camera_rgb.shape == (sh, sw, 3)
    assert np.allclose(out.side_camera_depth, side_depth)


def test_rgbd_by_catalog_id_roundtrip():
    """Per-catalog-id RGB-D chunks serialize after legacy side camera blobs."""
    camera_height, camera_width = 480, 640
    camera_rgb = np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8)
    camera_depth = np.random.rand(camera_height, camera_width).astype(np.float32) * 5.0
    scan_f = np.linspace(-0.5, 0.5, 132, dtype=np.float32)

    rgbd_by_catalog_id = {
        "front_rgbd": RgbdCatalogObservation(
            rgb=camera_rgb.copy(),
            depth=camera_depth.copy(),
            scan_features=scan_f.copy(),
        ),
        "side_right_rgbd": RgbdCatalogObservation(
            rgb=np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8),
            depth=np.random.rand(camera_height, camera_width).astype(np.float32),
            scan_features=np.linspace(0.0, 1.0, 64, dtype=np.float32),
        ),
        "side_left_rgbd": RgbdCatalogObservation(
            rgb=np.random.randint(0, 255, (camera_height, camera_width, 3), dtype=np.uint8),
            depth=np.random.rand(camera_height, camera_width).astype(np.float32),
        ),
    }

    hw = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=2,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        camera_rgb=camera_rgb,
        camera_depth=camera_depth,
        scan_features=scan_f,
        rgbd_by_catalog_id=rgbd_by_catalog_id,
    )
    out = HardwareObservations.from_bytes(hw.to_bytes())
    assert out.rgbd_by_catalog_id is not None
    assert set(out.rgbd_by_catalog_id) == {"front_rgbd", "side_right_rgbd", "side_left_rgbd"}
    for cid in out.rgbd_by_catalog_id:
        a = rgbd_by_catalog_id[cid]
        b = out.rgbd_by_catalog_id[cid]
        assert np.array_equal(b.rgb, a.rgb)
        assert np.allclose(b.depth, a.depth)
        if a.scan_features is None:
            assert b.scan_features is None
        else:
            assert np.allclose(b.scan_features, a.scan_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


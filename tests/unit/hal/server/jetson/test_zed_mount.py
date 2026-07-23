"""Camera→body mount rotation: config loading and frame math (AC 5d)."""

from pathlib import Path

import numpy as np
import pytest

from hal.server.jetson.zed_mount import (
    DEFAULT_MOUNT_YAML,
    ZED_MOUNT_YAML_ENV,
    apply_camera_to_body,
    load_camera_to_body_rotation,
    quat_xyzw_to_rot_matrix,
    rot_matrix_to_quat_xyzw,
)

# Rotate vectors +90° about z: camera x → body y
_R_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestLoadRotation:
    def test_missing_file_yields_identity(self, tmp_path):
        rot = load_camera_to_body_rotation(tmp_path / "nope.yaml")
        np.testing.assert_array_equal(rot, np.eye(3, dtype=np.float32))

    def test_packaged_default_is_identity(self):
        assert DEFAULT_MOUNT_YAML.exists()
        rot = load_camera_to_body_rotation()
        np.testing.assert_array_equal(rot, np.eye(3, dtype=np.float32))

    def test_loads_matrix_from_yaml(self, tmp_path):
        p = tmp_path / "mount.yaml"
        p.write_text(
            "r_camera_to_body:\n  - [0, -1, 0]\n  - [1, 0, 0]\n  - [0, 0, 1]\n"
        )
        rot = load_camera_to_body_rotation(p)
        np.testing.assert_allclose(rot, _R_Z90)
        assert rot.dtype == np.float32

    def test_env_var_override(self, tmp_path, monkeypatch):
        p = tmp_path / "mount.yaml"
        p.write_text(
            "r_camera_to_body:\n  - [0, 0, 1]\n  - [-1, 0, 0]\n  - [0, -1, 0]\n"
        )
        monkeypatch.setenv(ZED_MOUNT_YAML_ENV, str(p))
        rot = load_camera_to_body_rotation()
        np.testing.assert_allclose(
            rot, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        )

    def test_missing_key_raises(self, tmp_path):
        p = tmp_path / "mount.yaml"
        p.write_text("something_else: 1\n")
        with pytest.raises(RuntimeError, match="missing 'r_camera_to_body'"):
            load_camera_to_body_rotation(p)

    def test_non_orthonormal_raises(self, tmp_path):
        p = tmp_path / "mount.yaml"
        p.write_text(
            "r_camera_to_body:\n  - [1, 0.5, 0]\n  - [0, 1, 0]\n  - [0, 0, 1]\n"
        )
        with pytest.raises(RuntimeError, match="not orthonormal"):
            load_camera_to_body_rotation(p)

    def test_reflection_raises(self, tmp_path):
        p = tmp_path / "mount.yaml"
        p.write_text(
            "r_camera_to_body:\n  - [-1, 0, 0]\n  - [0, 1, 0]\n  - [0, 0, 1]\n"
        )
        with pytest.raises(RuntimeError, match="det"):
            load_camera_to_body_rotation(p)

    def test_wrong_shape_raises(self, tmp_path):
        p = tmp_path / "mount.yaml"
        p.write_text("r_camera_to_body:\n  - [1, 0]\n  - [0, 1]\n")
        with pytest.raises(RuntimeError, match="3x3"):
            load_camera_to_body_rotation(p)


class TestQuatMatrixConversions:
    # xyzw quats for: identity, 90° about x, 120° about (1,1,1), arbitrary
    QUATS = [
        [0.0, 0.0, 0.0, 1.0],
        [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)],
        [0.5, 0.5, 0.5, 0.5],
        [0.1, -0.3, 0.2, 0.9],  # normalized below
    ]

    @pytest.mark.parametrize("quat", QUATS)
    def test_round_trip(self, quat):
        q = np.asarray(quat, dtype=np.float64)
        q = q / np.linalg.norm(q)
        q2 = rot_matrix_to_quat_xyzw(quat_xyzw_to_rot_matrix(q))
        if np.dot(q, q2) < 0:
            q2 = -q2  # q and -q encode the same rotation
        np.testing.assert_allclose(q2, q, atol=1e-6)

    def test_known_matrix(self):
        # 90° about z in xyzw is (0, 0, sin45°, cos45°) and must map to _R_Z90
        q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        np.testing.assert_allclose(quat_xyzw_to_rot_matrix(q), _R_Z90, atol=1e-12)


class TestApplyCameraToBody:
    def test_identity_mount_passthrough(self):
        ang_vel = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        av_b, q_b = apply_camera_to_body(np.eye(3), ang_vel, quat)
        np.testing.assert_allclose(av_b, ang_vel, atol=1e-6)
        if np.dot(q_b, quat) < 0:
            q_b = -q_b
        np.testing.assert_allclose(q_b, quat, atol=1e-6)

    def test_ang_vel_rotates_with_mount(self):
        av_b, _ = apply_camera_to_body(
            _R_Z90, np.array([1.0, 0.0, 0.0]), np.array([0, 0, 0, 1.0])
        )
        np.testing.assert_allclose(av_b, [0.0, 1.0, 0.0], atol=1e-12)

    def test_quat_composition_matches_matrix_composition(self):
        # R_world_body must equal R_world_camera @ R_camera_to_body.T
        quat_c = np.array([0.1, -0.3, 0.2, 0.9])
        quat_c = quat_c / np.linalg.norm(quat_c)
        _, q_b = apply_camera_to_body(_R_Z90, np.zeros(3), quat_c)
        expected = quat_xyzw_to_rot_matrix(quat_c) @ _R_Z90.T
        np.testing.assert_allclose(quat_xyzw_to_rot_matrix(q_b), expected, atol=1e-6)

    def test_outputs_are_float32(self):
        av_b, q_b = apply_camera_to_body(
            np.eye(3), np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0])
        )
        assert av_b.dtype == np.float32
        assert q_b.dtype == np.float32

"""Unit tests for the krabby CLI package (AC8)."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _state: image-ref resolution and state roundtrip
# ---------------------------------------------------------------------------

from krabby._state import (
    DEFAULT_TAG,
    ECR_REPO,
    resolve_image_ref,
    load_state,
    save_state,
    installed_image,
)


class TestResolveImageRef:
    def test_none_returns_default(self):
        assert resolve_image_ref(None) == f"{ECR_REPO}:{DEFAULT_TAG}"

    def test_bare_tag_is_prefixed(self):
        assert resolve_image_ref("v1.2.3") == f"{ECR_REPO}:v1.2.3"

    def test_fully_qualified_uri_returned_as_is(self):
        uri = "ghcr.io/org/krabby-locomotion:latest"
        assert resolve_image_ref(uri) == uri

    def test_ecr_uri_with_tag_returned_as_is(self):
        uri = f"{ECR_REPO}:some-tag"
        assert resolve_image_ref(uri) == uri


class TestStateRoundtrip:
    def test_load_returns_empty_dict_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("krabby._state.STATE_PATH", tmp_path / "state.json")
        assert load_state() == {}

    def test_save_and_load(self, tmp_path, monkeypatch):
        path = tmp_path / "krabby" / "state.json"
        monkeypatch.setattr("krabby._state.STATE_PATH", path)
        save_state("myrepo:mytag", "sha256:abc")
        assert load_state() == {"image_ref": "myrepo:mytag", "digest": "sha256:abc"}

    def test_installed_image_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("krabby._state.STATE_PATH", tmp_path / "state.json")
        assert installed_image() is None

    def test_installed_image_returns_saved_ref(self, tmp_path, monkeypatch):
        path = tmp_path / "krabby" / "state.json"
        monkeypatch.setattr("krabby._state.STATE_PATH", path)
        save_state("myrepo:tag", "sha256:xyz")
        assert installed_image() == "myrepo:tag"

    def test_corrupt_state_file_returns_empty(self, tmp_path, monkeypatch):
        path = tmp_path / "state.json"
        path.write_text("not-json")
        monkeypatch.setattr("krabby._state.STATE_PATH", path)
        assert load_state() == {}


# ---------------------------------------------------------------------------
# _docker: command construction
# ---------------------------------------------------------------------------

from krabby._docker import gpu_flags, serial_device_flags, run_cmd, firmware_cmd, uno_cmd


class TestGpuFlags:
    def test_aarch64_returns_runtime_nvidia(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        assert gpu_flags() == ["--runtime=nvidia"]

    def test_x86_64_returns_gpus_all(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        assert gpu_flags() == ["--gpus", "all"]


class TestSerialDeviceFlags:
    def test_returns_device_flags_for_each_port(self, monkeypatch):
        monkeypatch.setattr(
            "krabby._docker.glob.glob",
            lambda pattern: {
                "/dev/ttyACM*": ["/dev/ttyACM0"],
                "/dev/ttyUSB*": ["/dev/ttyUSB0", "/dev/ttyUSB1"],
            }.get(pattern, []),
        )
        flags = serial_device_flags()
        assert flags == [
            "--device", "/dev/ttyACM0",
            "--device", "/dev/ttyUSB0",
            "--device", "/dev/ttyUSB1",
        ]

    def test_returns_empty_when_no_devices(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        assert serial_device_flags() == []


class TestRunCmd:
    def test_contains_privileged_and_dev_mount(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = run_cmd("myimage:tag", [])
        assert "--privileged" in cmd
        assert "-v" in cmd
        assert "/dev:/dev" in cmd

    def test_contains_gpu_flags(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = run_cmd("myimage:tag", [])
        assert "--runtime=nvidia" in cmd

    def test_contains_zmq_ports(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = run_cmd("myimage:tag", [])
        assert "-p" in cmd
        assert "6001:6001" in cmd
        assert "6002:6002" in cmd

    def test_extra_args_appended(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = run_cmd("myimage:tag", ["--checkpoint", "/path/to/ckpt.pt"])
        assert cmd[-2:] == ["--checkpoint", "/path/to/ckpt.pt"]

    def test_image_in_cmd(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = run_cmd("myrepo:mytag", [])
        assert "myrepo:mytag" in cmd

    def test_dev_mount_covers_input_and_serial_devices(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = run_cmd("myimage:tag", [])
        # /dev:/dev exposes /dev/ttyACM*, /dev/ttyUSB*, /dev/input/js*, /dev/input/event*
        assert "/dev:/dev" in cmd

    def test_extra_mounts_inserted_before_image(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = run_cmd("myimage:tag", [], extra_mounts=["/tmp/repo:/workspace"])
        img_idx = cmd.index("myimage:tag")
        assert "-v" in cmd[:img_idx]
        assert "/tmp/repo:/workspace" in cmd[:img_idx]

    def test_multiple_extra_mounts(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = run_cmd("myimage:tag", [], extra_mounts=["/a:/a", "/b:/b"])
        img_idx = cmd.index("myimage:tag")
        docker_flags = cmd[:img_idx]
        v_indices = [i for i, x in enumerate(docker_flags) if x == "-v"]
        mounts = [docker_flags[i + 1] for i in v_indices]
        assert "/a:/a" in mounts
        assert "/b:/b" in mounts

    def test_no_extra_mounts_when_not_provided(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd_no_mounts = run_cmd("myimage:tag", [])
        cmd_empty_mounts = run_cmd("myimage:tag", [], extra_mounts=[])
        assert cmd_no_mounts == cmd_empty_mounts


class TestUnoCmd:
    def test_entrypoint_is_krabby_uno(self):
        cmd = uno_cmd("myimage:tag", [])
        idx = cmd.index("--entrypoint")
        assert cmd[idx + 1] == "krabby-uno"

    def test_shares_krabby_container_network(self):
        cmd = uno_cmd("myimage:tag", [])
        assert "--network=container:krabby" in cmd

    def test_dev_mount_for_gamepad_access(self):
        cmd = uno_cmd("myimage:tag", [])
        assert "/dev:/dev" in cmd

    def test_extra_args_appended(self):
        cmd = uno_cmd("myimage:tag", ["--device-id", "1"])
        assert cmd[-2:] == ["--device-id", "1"]

    def test_image_in_cmd(self):
        cmd = uno_cmd("myimage:tag", [])
        assert "myimage:tag" in cmd


from krabby.run import cmd_run, _GAMEPAD_ONLY_ENTRYPOINT, _GAMEPAD_ONLY_ARGS


class TestCmdRunGamepadOnly:
    def _run(self, monkeypatch, **kwargs):
        """Call cmd_run with subprocess and sys.exit mocked; return captured run_cmd kwargs."""
        captured = {}

        def fake_run_cmd(ref, extra_args, entrypoint=None, extra_mounts=None):
            captured.update(ref=ref, extra_args=extra_args, entrypoint=entrypoint)
            return ["docker", "run", ref]

        monkeypatch.setattr("krabby.run.run_cmd", fake_run_cmd)
        monkeypatch.setattr("krabby.run.subprocess.run", lambda cmd: type("R", (), {"returncode": 0})())
        monkeypatch.setattr("krabby.run.sys.exit", lambda _code: None)
        cmd_run(**kwargs)
        return captured

    def test_entrypoint_is_hal_server_jetson(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", gamepad_only=True)
        assert captured["entrypoint"] == "krabby-hal-server-jetson"

    def test_args_contain_control_source_gamepad(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", gamepad_only=True)
        assert captured["extra_args"] == ["--control-source", "gamepad"]

    def test_extra_args_appended_after_gamepad_args(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", gamepad_only=True, extra_args=["--robot", "go2"])
        assert captured["extra_args"] == ["--control-source", "gamepad", "--robot", "go2"]

    def test_no_args_no_entrypoint_exits_1(self):
        with pytest.raises(SystemExit) as exc_info:
            cmd_run(image_ref="img:tag")
        assert exc_info.value.code == 1

    def test_no_args_error_message_mentions_gamepad_only(self, capsys):
        with pytest.raises(SystemExit):
            cmd_run(image_ref="img:tag")
        assert "gamepad-only" in capsys.readouterr().err

    def test_gamepad_only_entrypoint_constant(self):
        assert _GAMEPAD_ONLY_ENTRYPOINT == "krabby-hal-server-jetson"

    def test_gamepad_only_args_constant(self):
        assert _GAMEPAD_ONLY_ARGS == ["--control-source", "gamepad"]


class TestFirmwareCmd:
    def test_entrypoint_is_krabby_firmware(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        cmd = firmware_cmd("myimage:tag", ["show"])
        assert "--entrypoint" in cmd
        idx = cmd.index("--entrypoint")
        assert cmd[idx + 1] == "krabby-firmware"

    def test_firmware_args_appended(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        cmd = firmware_cmd("myimage:tag", ["update", "--device", "/dev/ttyACM0"])
        assert cmd[-3:] == ["update", "--device", "/dev/ttyACM0"]

    def test_cache_volume_mounted(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        cmd = firmware_cmd("myimage:tag", [])
        cache_entries = [a for a in cmd if "krabby-firmware" in a and "cache" in a]
        assert len(cache_entries) == 1
        assert ":/root/.cache/krabby-firmware" in cache_entries[0]

    def test_device_flags_included(self, monkeypatch):
        monkeypatch.setattr(
            "krabby._docker.glob.glob",
            lambda pattern: {
                "/dev/ttyACM*": ["/dev/ttyACM0"],
                "/dev/ttyUSB*": [],
            }.get(pattern, []),
        )
        cmd = firmware_cmd("myimage:tag", ["show"])
        assert "--device" in cmd
        assert "/dev/ttyACM0" in cmd

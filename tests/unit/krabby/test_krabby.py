"""Unit tests for the krabby CLI package (AC8)."""
from __future__ import annotations

import sys
import types
from pathlib import Path


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

    def test_default_tag_is_release_latest(self):
        # Task E: kit owners track the stable release line by default.
        assert DEFAULT_TAG == "release-latest"

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


class TestStateHome:
    """State must live in the invoking user's home even under `sudo krabby install`,
    so the user's later unprivileged run/firmware commands read the same file."""

    def test_uses_sudo_user_home(self, monkeypatch):
        import sys, types
        import krabby._state as st
        monkeypatch.setenv("SUDO_USER", "krabby")
        monkeypatch.setitem(sys.modules, "pwd",
                            types.SimpleNamespace(getpwnam=lambda u: types.SimpleNamespace(pw_dir=f"/home/{u}")))
        assert str(st._state_home()) == "/home/krabby"

    def test_no_sudo_uses_real_home(self, monkeypatch):
        import krabby._state as st
        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setattr("krabby._state.Path.home", lambda: st.Path("/home/me"))
        assert str(st._state_home()) == "/home/me"

    def test_unknown_sudo_user_falls_back(self, monkeypatch):
        import sys, types
        import krabby._state as st
        monkeypatch.setenv("SUDO_USER", "ghost")
        def _boom(_u):
            raise KeyError(_u)
        monkeypatch.setitem(sys.modules, "pwd", types.SimpleNamespace(getpwnam=_boom))
        monkeypatch.setattr("krabby._state.Path.home", lambda: st.Path("/home/me"))
        assert str(st._state_home()) == "/home/me"


class TestVersion:
    """`krabby --version` is read from package metadata (set from the release tag),
    so it can never drift from the published version — nothing to bump by hand."""

    def test_reads_installed_metadata(self, monkeypatch):
        import krabby.__main__ as m
        monkeypatch.setattr(m, "_pkg_version", lambda name: "9.9.9")
        assert m._version() == "9.9.9"

    def test_fallback_when_not_installed(self, monkeypatch):
        import krabby.__main__ as m
        from importlib.metadata import PackageNotFoundError
        def _missing(name):
            raise PackageNotFoundError(name)
        monkeypatch.setattr(m, "_pkg_version", _missing)
        assert m._version() == "0+unknown"

    def test_version_flag_prints_metadata_version(self, monkeypatch, capsys):
        import krabby.__main__ as m
        monkeypatch.setattr(m, "_pkg_version", lambda name: "1.2.3")
        monkeypatch.setattr(sys, "argv", ["krabby", "--version"])
        try:
            m.main()
            raise AssertionError("expected SystemExit from --version")
        except SystemExit as e:
            assert e.code == 0
        assert "krabby 1.2.3" in capsys.readouterr().out


class TestBootService:
    """`krabby install` installs a systemd unit to start `krabby run` on boot
    (default on); `--no-launch-on-startup` tears it down."""

    def test_unit_content(self):
        import krabby._host as h
        u = h._boot_service_unit("/usr/local/bin/krabby", "krabby")
        assert "ExecStart=/usr/local/bin/krabby run" in u
        assert "User=krabby" in u
        assert "ExecStop=/usr/bin/docker stop krabby" in u
        assert "WantedBy=multi-user.target" in u
        assert "Requires=docker.service" in u

    def test_enable_writes_unit_and_enables(self, tmp_path, monkeypatch):
        import krabby._host as h
        calls = []
        monkeypatch.setattr(h, "_BOOT_SERVICE_PATH", tmp_path / "krabby-locomotion.service")
        monkeypatch.setattr(h.shutil, "which",
                            lambda n: "/usr/bin/systemctl" if n == "systemctl" else "/usr/local/bin/krabby")
        monkeypatch.setattr(h, "_run", lambda cmd: calls.append(cmd) or 0)
        monkeypatch.setenv("SUDO_USER", "krabby")
        assert h._ensure_boot_service(True) is True
        unit = (tmp_path / "krabby-locomotion.service").read_text()
        assert "ExecStart=/usr/local/bin/krabby run" in unit
        assert "User=krabby" in unit
        assert ["systemctl", "enable", "krabby-locomotion.service"] in calls

    def test_idempotent_when_unit_unchanged(self, tmp_path, monkeypatch):
        import krabby._host as h
        path = tmp_path / "krabby-locomotion.service"
        monkeypatch.setattr(h, "_BOOT_SERVICE_PATH", path)
        monkeypatch.setattr(h.shutil, "which",
                            lambda n: "/usr/bin/systemctl" if n == "systemctl" else "/usr/local/bin/krabby")
        monkeypatch.setattr(h, "_run", lambda cmd: 0)
        monkeypatch.setenv("SUDO_USER", "krabby")
        path.write_text(h._boot_service_unit("/usr/local/bin/krabby", "krabby"))
        mtime = path.stat().st_mtime_ns
        assert h._ensure_boot_service(True) is True
        assert path.stat().st_mtime_ns == mtime  # not rewritten

    def test_disable_removes_unit(self, tmp_path, monkeypatch):
        import krabby._host as h
        path = tmp_path / "krabby-locomotion.service"
        path.write_text("stale")
        calls = []
        monkeypatch.setattr(h, "_BOOT_SERVICE_PATH", path)
        monkeypatch.setattr(h.shutil, "which", lambda n: "/usr/bin/systemctl")
        monkeypatch.setattr(h, "_run", lambda cmd: calls.append(cmd) or 0)
        assert h._ensure_boot_service(False) is True
        assert not path.exists()
        assert ["systemctl", "disable", "--now", "krabby-locomotion.service"] in calls

    def test_skips_without_systemctl(self, monkeypatch):
        import krabby._host as h
        monkeypatch.setattr(h.shutil, "which", lambda n: None)
        assert h._ensure_boot_service(True) is True  # graceful no-op on non-systemd hosts


class TestDkmsEnsureInstalled:
    """`krabby install`'s DKMS step must be idempotent: a module left in the
    'added' or 'built' state by an interrupted run should resume, not fail on
    `dkms add` (regression — that aborted every re-run)."""

    def _run_recorder(self, monkeypatch):
        import krabby._host as h
        calls = []
        monkeypatch.setattr(h, "_run", lambda cmd: calls.append(cmd) or 0)
        return h, calls

    def _verbs(self, calls):
        return [c[1] for c in calls if c[:1] == ["dkms"]]

    def test_fresh_module_adds_builds_installs(self, monkeypatch):
        h, calls = self._run_recorder(monkeypatch)
        monkeypatch.setattr(h, "_dkms_state", lambda p, v: None)
        assert h._dkms_ensure_installed(Path("/src"), "nintendo", "3.2") is True
        assert self._verbs(calls) == ["add", "build", "install"]

    def test_already_added_skips_add(self, monkeypatch):
        h, calls = self._run_recorder(monkeypatch)
        monkeypatch.setattr(h, "_dkms_state", lambda p, v: "added")
        assert h._dkms_ensure_installed(Path("/src"), "nintendo", "3.2") is True
        assert self._verbs(calls) == ["build", "install"]  # no add

    def test_already_built_only_installs(self, monkeypatch):
        h, calls = self._run_recorder(monkeypatch)
        monkeypatch.setattr(h, "_dkms_state", lambda p, v: "built")
        assert h._dkms_ensure_installed(Path("/src"), "nintendo", "3.2") is True
        assert self._verbs(calls) == ["install"]

    def test_already_installed_does_nothing(self, monkeypatch):
        h, calls = self._run_recorder(monkeypatch)
        monkeypatch.setattr(h, "_dkms_state", lambda p, v: "installed")
        assert h._dkms_ensure_installed(Path("/src"), "nintendo", "3.2") is True
        assert self._verbs(calls) == []

    def test_add_failure_returns_false(self, monkeypatch):
        import krabby._host as h
        monkeypatch.setattr(h, "_dkms_state", lambda p, v: None)
        monkeypatch.setattr(h, "_run", lambda cmd: 1)  # add fails
        assert h._dkms_ensure_installed(Path("/src"), "nintendo", "3.2") is False

    def test_dkms_state_parses_status(self, monkeypatch):
        import krabby._host as h
        monkeypatch.setattr(
            h.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(
                stdout="nintendo/3.2, 5.15.148-tegra, aarch64: installed\n", returncode=0),
        )
        assert h._dkms_state("nintendo", "3.2") == "installed"

    def test_dkms_state_added(self, monkeypatch):
        import krabby._host as h
        monkeypatch.setattr(
            h.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(stdout="nintendo/3.2: added\n", returncode=0),
        )
        assert h._dkms_state("nintendo", "3.2") == "added"

    def test_dkms_state_absent(self, monkeypatch):
        import krabby._host as h
        monkeypatch.setattr(
            h.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(stdout="", returncode=0),
        )
        assert h._dkms_state("nintendo", "3.2") is None


class TestInstallLaunchFlag:
    """`--launch-on-startup` defaults to true; `--no-launch-on-startup` opts out."""

    def _run_install(self, monkeypatch, argv):
        cap = {}
        monkeypatch.setattr(
            "krabby.install.cmd_install",
            lambda image_ref=None, launch_on_startup=True: cap.update(image=image_ref, launch=launch_on_startup),
        )
        monkeypatch.setattr(sys, "argv", argv)
        krabby_main.main()
        return cap

    def test_default_is_true(self, monkeypatch):
        assert self._run_install(monkeypatch, ["krabby", "install"])["launch"] is True

    def test_no_launch_on_startup_disables(self, monkeypatch):
        assert self._run_install(monkeypatch, ["krabby", "install", "--no-launch-on-startup"])["launch"] is False

    def test_explicit_launch_on_startup(self, monkeypatch):
        assert self._run_install(monkeypatch, ["krabby", "install", "--launch-on-startup"])["launch"] is True


# ---------------------------------------------------------------------------
# _docker: command construction
# ---------------------------------------------------------------------------

from krabby._docker import gpu_flags, host_network_flags, network_flags, serial_device_flags, run_cmd, firmware_cmd, gamepad_cmd


class TestGpuFlags:
    def test_aarch64_returns_runtime_nvidia(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        assert gpu_flags() == ["--runtime=nvidia"]

    def test_x86_64_returns_gpus_all(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        assert gpu_flags() == ["--gpus", "all"]


class TestHostNetworkFlags:
    def test_aarch64_uses_host_networking(self, monkeypatch):
        # Tegra kernels lack the iptables `raw` table Docker's bridge needs, so
        # *any* bridge container fails — the Jetson path must use host net.
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        assert host_network_flags() == ["--network", "host"]

    def test_x86_64_adds_nothing(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        assert host_network_flags() == []


class TestNetworkFlags:
    def test_aarch64_uses_host_networking(self, monkeypatch):
        # Host networking exposes the ZMQ endpoints; `-p` would be redundant.
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        assert network_flags() == ["--network", "host"]

    def test_x86_64_publishes_ports(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        assert network_flags() == ["-p", "6001:6001", "-p", "6002:6002"]


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

    def test_aarch64_uses_host_network_not_published_ports(self, monkeypatch):
        # Regression: on Jetson the bridge driver fails to publish ports
        # (kernel lacks iptables `raw`), so the run command must use host net.
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = run_cmd("myimage:tag", [])
        assert "--network" in cmd and "host" in cmd
        assert "-p" not in cmd

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


class TestGamepadCmd:
    def test_entrypoint_is_bash(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = gamepad_cmd("myimage:tag", [])
        idx = cmd.index("--entrypoint")
        assert cmd[idx + 1] == "bash"

    def test_launches_server_and_client(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        script = " ".join(gamepad_cmd("myimage:tag", []))
        # The single container brings up both the HAL server and the krabby-uno client.
        assert "krabby-hal-server-jetson" in script
        assert "--control-source gamepad" in script
        assert "krabby-uno" in script

    def test_forwards_signals_for_clean_shutdown(self):
        script = " ".join(gamepad_cmd("myimage:tag", []))
        assert "trap" in script

    def test_exposes_zmq_ports(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = gamepad_cmd("myimage:tag", [])
        assert "6001:6001" in cmd
        assert "6002:6002" in cmd

    def test_dev_mount_and_privileged(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = gamepad_cmd("myimage:tag", [])
        assert "--privileged" in cmd
        assert "/dev:/dev" in cmd

    def test_gpu_flags_included(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = gamepad_cmd("myimage:tag", [])
        assert "--runtime=nvidia" in cmd

    def test_extra_args_forwarded_to_uno_after_script(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        cmd = gamepad_cmd("myimage:tag", ["--device-id", "1", "--rate", "50"])
        # client args trail the script's $0 placeholder so they arrive as "$@" for krabby-uno
        assert cmd[-4:] == ["--device-id", "1", "--rate", "50"]
        c_idx = cmd.index("-c")
        bash_dollar0 = cmd.index("bash", c_idx + 2)  # the $0 placeholder after `-c <script>`
        assert bash_dollar0 < cmd.index("--device-id")

    def test_image_in_cmd(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        assert "myimage:tag" in gamepad_cmd("myimage:tag", [])

    def test_propagates_child_exit_status(self, monkeypatch):
        # A crashed server/client must fail `krabby run`, not look like a clean exit.
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        script = " ".join(gamepad_cmd("myimage:tag", []))
        assert "wait -n" in script
        assert "status=$?" in script
        assert "exit $status" in script

    def test_robot_value_injected_into_server(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = gamepad_cmd("myimage:tag", ["--robot", "go2"])
        script = " ".join(cmd)
        assert "--control-source gamepad --robot go2" in script  # server matched to client
        assert cmd[-2:] == ["--robot", "go2"]                    # still forwarded to client via $@

    def test_no_robot_leaves_server_default(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        assert "--robot" not in " ".join(gamepad_cmd("myimage:tag", []))

    def test_unknown_robot_not_injected_into_server(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        # only known topologies are injected server-side (guards against arg injection)
        script = " ".join(gamepad_cmd("myimage:tag", ["--robot", "bogus"]))
        assert "--control-source gamepad --observation-bind" in script


from krabby.run import cmd_run
from krabby import __main__ as krabby_main


class TestCmdRun:
    def _run(self, monkeypatch, **kwargs):
        """Call cmd_run with the docker builders, subprocess, and sys.exit mocked.

        Returns ("gamepad"|"inference", captured_kwargs) so tests can assert which
        path ran and with what args.
        """
        captured = {}

        def fake_gamepad_cmd(ref, extra_args, extra_mounts=None):
            captured.update(mode="gamepad", ref=ref, extra_args=extra_args, extra_mounts=extra_mounts)
            return ["docker", "run", ref]

        def fake_run_cmd(ref, extra_args, entrypoint=None, extra_mounts=None):
            captured.update(mode="inference", ref=ref, extra_args=extra_args, entrypoint=entrypoint, extra_mounts=extra_mounts)
            return ["docker", "run", ref]

        monkeypatch.setattr("krabby.run.gamepad_cmd", fake_gamepad_cmd)
        monkeypatch.setattr("krabby.run.run_cmd", fake_run_cmd)
        monkeypatch.setattr("krabby.run.subprocess.run", lambda cmd: type("R", (), {"returncode": 0})())
        monkeypatch.setattr("krabby.run.sys.exit", lambda _code: None)
        cmd_run(**kwargs)
        return captured

    def test_base_call_launches_gamepad_stack(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag")
        assert captured["mode"] == "gamepad"
        assert captured["extra_args"] == []

    def test_gamepad_only_flag_launches_gamepad_stack(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", gamepad_only=True)
        assert captured["mode"] == "gamepad"

    def test_gamepad_client_args_forwarded(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", gamepad_only=True, extra_args=["--device-id", "1"])
        assert captured["mode"] == "gamepad"
        assert captured["extra_args"] == ["--device-id", "1"]

    def test_checkpoint_args_select_inference(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", extra_args=["--checkpoint", "/ckpt.pt"])
        assert captured["mode"] == "inference"
        assert captured["extra_args"] == ["--checkpoint", "/ckpt.pt"]

    def test_explicit_entrypoint_selects_inference(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", entrypoint="krabby-hal-server-jetson")
        assert captured["mode"] == "inference"
        assert captured["entrypoint"] == "krabby-hal-server-jetson"

    def test_extra_mounts_passed_through_gamepad(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", extra_mounts=["/a:/a"])
        assert captured["mode"] == "gamepad"
        assert captured["extra_mounts"] == ["/a:/a"]

    def test_leading_double_dash_stripped_then_inference(self, monkeypatch):
        # argparse REMAINDER yields ["--", "--checkpoint", ...]; the `--` must not reach the container.
        captured = self._run(monkeypatch, image_ref="img:tag", extra_args=["--", "--checkpoint", "/ckpt.pt"])
        assert captured["mode"] == "inference"
        assert captured["extra_args"] == ["--checkpoint", "/ckpt.pt"]

    def test_gamepad_client_args_without_flag_stay_gamepad(self, monkeypatch):
        # client args like --device-id must NOT be misrouted to the inference server
        captured = self._run(monkeypatch, image_ref="img:tag", extra_args=["--device-id", "1"])
        assert captured["mode"] == "gamepad"
        assert captured["extra_args"] == ["--device-id", "1"]

    def test_gamepad_only_overrides_checkpoint(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", gamepad_only=True, extra_args=["--checkpoint", "/c"])
        assert captured["mode"] == "gamepad"

    def test_inference_passes_extra_mounts(self, monkeypatch):
        captured = self._run(monkeypatch, image_ref="img:tag", extra_args=["--checkpoint", "/c"], extra_mounts=["/a:/a"])
        assert captured["mode"] == "inference"
        assert captured["extra_mounts"] == ["/a:/a"]


class TestRunArgvEndToEnd:
    """Drive the real argparse path via main() — guards the `--`/REMAINDER handling
    that the direct-cmd_run tests can't see (they get pre-stripped lists)."""

    def _capture(self, monkeypatch, argv):
        captured = {}

        def fake_gamepad_cmd(ref, extra_args, extra_mounts=None):
            captured.update(mode="gamepad", extra_args=extra_args)
            return ["docker"]

        def fake_run_cmd(ref, extra_args, entrypoint=None, extra_mounts=None):
            captured.update(mode="inference", extra_args=extra_args, entrypoint=entrypoint)
            return ["docker"]

        monkeypatch.setattr("krabby.run.gamepad_cmd", fake_gamepad_cmd)
        monkeypatch.setattr("krabby.run.run_cmd", fake_run_cmd)
        monkeypatch.setattr("krabby.run.installed_image", lambda: "img:tag")
        monkeypatch.setattr("krabby.run.subprocess.run", lambda cmd: type("R", (), {"returncode": 0})())
        monkeypatch.setattr("krabby.run.sys.exit", lambda _code: None)
        monkeypatch.setattr(sys, "argv", argv)
        krabby_main.main()
        return captured

    def test_checkpoint_via_double_dash_strips_and_infers(self, monkeypatch):
        captured = self._capture(monkeypatch, ["krabby", "run", "--", "--checkpoint", "/ckpt.pt"])
        assert captured["mode"] == "inference"
        assert captured["extra_args"] == ["--checkpoint", "/ckpt.pt"]  # no leading --

    def test_device_id_via_double_dash_is_gamepad(self, monkeypatch):
        captured = self._capture(monkeypatch, ["krabby", "run", "--gamepad-only", "--", "--device-id", "1"])
        assert captured["mode"] == "gamepad"
        assert captured["extra_args"] == ["--device-id", "1"]

    def test_bare_double_dash_is_gamepad(self, monkeypatch):
        captured = self._capture(monkeypatch, ["krabby", "run", "--"])
        assert captured["mode"] == "gamepad"
        assert captured["extra_args"] == []

    def test_no_args_is_gamepad(self, monkeypatch):
        captured = self._capture(monkeypatch, ["krabby", "run"])
        assert captured["mode"] == "gamepad"


class TestCmdFirmwareInteractive:
    def _capture(self, monkeypatch, stdin_tty, stdout_tty):
        captured = {}

        def fake_firmware_cmd(ref, args, interactive=False):
            captured["interactive"] = interactive
            return ["docker"]

        monkeypatch.setattr("krabby.firmware.firmware_cmd", fake_firmware_cmd)
        monkeypatch.setattr("krabby.firmware.installed_image", lambda: "img:tag")
        monkeypatch.setattr("krabby.firmware.subprocess.run", lambda cmd: type("R", (), {"returncode": 0})())
        monkeypatch.setattr("krabby.firmware.sys.exit", lambda _code: None)
        monkeypatch.setattr("krabby.firmware.sys.stdin", type("S", (), {"isatty": lambda self: stdin_tty})())
        monkeypatch.setattr("krabby.firmware.sys.stdout", type("S", (), {"isatty": lambda self: stdout_tty})())
        from krabby.firmware import cmd_firmware
        cmd_firmware(["show"], image_ref="img:tag")
        return captured

    def test_interactive_when_both_streams_tty(self, monkeypatch):
        assert self._capture(monkeypatch, True, True)["interactive"] is True

    def test_not_interactive_when_stdout_piped(self, monkeypatch):
        assert self._capture(monkeypatch, True, False)["interactive"] is False

    def test_not_interactive_when_stdin_piped(self, monkeypatch):
        assert self._capture(monkeypatch, False, True)["interactive"] is False


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

    def test_no_tty_by_default(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        assert "-it" not in firmware_cmd("myimage:tag", ["show"])

    def test_interactive_allocates_tty(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        assert "-it" in firmware_cmd("myimage:tag", [], interactive=True)

    def test_aarch64_uses_host_network(self, monkeypatch):
        # Regression: `krabby firmware ...` also runs a container, so it hits
        # the same Tegra bridge failure and must use host networking.
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "aarch64")
        cmd = firmware_cmd("myimage:tag", ["show"])
        assert "--network" in cmd and "host" in cmd

    def test_x86_64_no_host_network(self, monkeypatch):
        monkeypatch.setattr("krabby._docker.glob.glob", lambda _: [])
        monkeypatch.setattr("krabby._docker.platform.machine", lambda: "x86_64")
        assert "--network" not in firmware_cmd("myimage:tag", ["show"])

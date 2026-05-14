"""Unit tests for firmware.install (--install command)."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call


import firmware.install as install_mod


class TestEnsureUdevRule:
    def test_noop_when_rule_already_correct(self, tmp_path, monkeypatch):
        rule_path = tmp_path / "99-krabby-mega.rules"
        rule_path.write_text(install_mod.UDEV_RULE)
        monkeypatch.setattr(install_mod, "UDEV_RULE_PATH", rule_path)
        with patch.object(install_mod, "_run") as mock_run:
            result = install_mod._ensure_udev_rule()
        assert result is True
        mock_run.assert_not_called()

    def test_writes_rule_when_missing(self, tmp_path, monkeypatch):
        rule_path = tmp_path / "99-krabby-mega.rules"
        monkeypatch.setattr(install_mod, "UDEV_RULE_PATH", rule_path)
        with patch.object(install_mod, "_run", return_value=0):
            result = install_mod._ensure_udev_rule()
        assert result is True
        assert rule_path.read_text() == install_mod.UDEV_RULE

    def test_writes_rule_when_stale(self, tmp_path, monkeypatch):
        rule_path = tmp_path / "99-krabby-mega.rules"
        rule_path.write_text("old content\n")
        monkeypatch.setattr(install_mod, "UDEV_RULE_PATH", rule_path)
        with patch.object(install_mod, "_run", return_value=0):
            result = install_mod._ensure_udev_rule()
        assert result is True
        assert rule_path.read_text() == install_mod.UDEV_RULE

    def test_returns_false_on_permission_error(self, tmp_path, monkeypatch):
        rule_path = tmp_path / "no-write" / "rule.rules"
        monkeypatch.setattr(install_mod, "UDEV_RULE_PATH", rule_path)

        def _raise(*_):
            raise PermissionError

        monkeypatch.setattr(rule_path.__class__, "write_text", lambda *_: (_ for _ in ()).throw(PermissionError()))
        with patch.object(install_mod, "_run", return_value=0):
            result = install_mod._ensure_udev_rule()
        assert result is False

    def test_reloads_udev_after_write(self, tmp_path, monkeypatch):
        rule_path = tmp_path / "99-krabby-mega.rules"
        monkeypatch.setattr(install_mod, "UDEV_RULE_PATH", rule_path)
        calls = []
        with patch.object(install_mod, "_run", side_effect=lambda cmd: calls.append(cmd) or 0):
            install_mod._ensure_udev_rule()
        assert any("reload-rules" in " ".join(c) for c in calls)


class TestEnsureDialout:
    def test_skips_when_already_in_group(self, monkeypatch):
        monkeypatch.setenv("USER", "alice")
        import subprocess
        with patch("subprocess.run", return_value=MagicMock(stdout="alice dialout sudo\n")) as mock_run:
            with patch.object(install_mod, "_run") as mock_internal:
                install_mod._ensure_dialout()
        mock_internal.assert_not_called()

    def test_adds_user_when_not_in_group(self, monkeypatch):
        monkeypatch.setenv("USER", "alice")
        with patch("subprocess.run", return_value=MagicMock(stdout="alice sudo\n")):
            with patch.object(install_mod, "_run", return_value=0) as mock_run:
                install_mod._ensure_dialout()
        mock_run.assert_called_once_with(["usermod", "-aG", "dialout", "alice"])


class TestEnsureTool:
    def test_noop_when_tool_present(self):
        with patch("shutil.which", return_value="/usr/bin/avrdude"):
            with patch.object(install_mod, "_run") as mock_run:
                install_mod._ensure_tool("avrdude", ["apt-get", "install", "-y", "avrdude"])
        mock_run.assert_not_called()

    def test_installs_when_tool_missing(self):
        with patch("shutil.which", return_value=None):
            with patch.object(install_mod, "_run", return_value=0) as mock_run:
                install_mod._ensure_tool("avrdude", ["apt-get", "install", "-y", "avrdude"])
        mock_run.assert_called_once_with(["apt-get", "install", "-y", "avrdude"])


class TestEnsurePlatformLocal:
    def test_noop_when_file_already_correct(self, tmp_path, monkeypatch):
        p = tmp_path / "platform.local.txt"
        p.write_text(install_mod.PLATFORM_LOCAL_CONTENT)
        monkeypatch.setattr(install_mod, "PLATFORM_LOCAL_PATH", p)
        install_mod._ensure_platform_local()
        assert p.read_text() == install_mod.PLATFORM_LOCAL_CONTENT

    def test_writes_file_when_missing(self, tmp_path, monkeypatch):
        p = tmp_path / "platform.local.txt"
        monkeypatch.setattr(install_mod, "PLATFORM_LOCAL_PATH", p)
        install_mod._ensure_platform_local()
        assert p.read_text() == install_mod.PLATFORM_LOCAL_CONTENT

    def test_overwrites_stale_file(self, tmp_path, monkeypatch):
        p = tmp_path / "platform.local.txt"
        p.write_text("old=stuff\n")
        monkeypatch.setattr(install_mod, "PLATFORM_LOCAL_PATH", p)
        install_mod._ensure_platform_local()
        assert p.read_text() == install_mod.PLATFORM_LOCAL_CONTENT

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        p = tmp_path / "deep" / "nested" / "platform.local.txt"
        monkeypatch.setattr(install_mod, "PLATFORM_LOCAL_PATH", p)
        install_mod._ensure_platform_local()
        assert p.exists()

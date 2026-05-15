"""Unit tests for firmware.cli (--show / --update)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import firmware.cli as cli_mod
from firmware.manifest import BranchEntry, FirmwareIndex


# --- fixtures ---

def _make_index(branches: dict) -> FirmwareIndex:
    return FirmwareIndex(
        schema_version=1,
        updated="2025-01-01T00:00:00",
        branches={
            name: BranchEntry(
                branch=name,
                build_key=bk,
                hex_url=f"https://example.com/{name}/firmware.hex",
                manifest_url=f"https://example.com/{name}/manifest.json",
            )
            for name, bk in branches.items()
        },
    )


# --- _is_port ---

class TestIsPort:
    def test_dev_tty_is_port(self):
        assert cli_mod._is_port("/dev/ttyACM0") is True

    def test_com_is_port(self):
        assert cli_mod._is_port("COM3") is True

    def test_branch_name_is_not_port(self):
        assert cli_mod._is_port("release/0.2.0") is False

    def test_mainline_is_not_port(self):
        assert cli_mod._is_port("mainline") is False


# --- cmd_show ---

class TestCmdShow:
    def test_no_boards_no_crash(self, capsys):
        index = _make_index({"mainline": "20250101-120000-abc1234"})
        with patch.object(cli_mod, "_all_mega_ports", return_value=[]):
            with patch.object(cli_mod, "_fetch_index", return_value=index):
                cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "No attached Mega boards" in out
        assert "mainline" in out

    def test_shows_version_for_board(self, capsys):
        index = _make_index({"release/0.2.0": "20250101-120000-abc1234"})
        with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0"]):
            with patch.object(cli_mod, "_probe_version", return_value="VER 0.2.0 release/0.2.0 abc1234"):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "/dev/ttyACM0" in out
        assert "0.2.0" in out

    def test_shows_no_version_response_when_probe_fails(self, capsys):
        index = _make_index({"mainline": "20250101-120000-abc1234"})
        with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0"]):
            with patch.object(cli_mod, "_probe_version", return_value=None):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "no version response" in out

    def test_shows_multiple_branches(self, capsys):
        index = _make_index({
            "mainline": "20250101-120000-abc1234",
            "release/0.2.0": "20250201-120000-def5678",
        })
        with patch.object(cli_mod, "_all_mega_ports", return_value=[]):
            with patch.object(cli_mod, "_fetch_index", return_value=index):
                cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "mainline" in out
        assert "release/0.2.0" in out

    def test_s3_error_printed_to_stderr(self, capsys):
        with patch.object(cli_mod, "_all_mega_ports", return_value=[]):
            with patch.object(cli_mod, "_fetch_index", side_effect=Exception("timeout")):
                cli_mod.cmd_show()
        err = capsys.readouterr().err
        assert "timeout" in err


# --- cmd_update ---

class TestCmdUpdate:
    def _default_index(self):
        return _make_index({
            "mainline": "20250101-120000-aaa0000",
            "release/0.2.0": "20250201-120000-bbb1111",
        })

    def test_uses_latest_release_when_no_branch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with patch.object(cli_mod, "_download_hex") as mock_dl:
                with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0"]):
                    with patch.object(cli_mod, "_flash") as mock_flash:
                        cli_mod.cmd_update()
        # release/0.2.0 is the latest release branch
        dl_dest = mock_dl.call_args[0][1]
        assert "release" in str(dl_dest)
        mock_flash.assert_called_once()

    def test_explicit_branch_used(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with patch.object(cli_mod, "_download_hex") as mock_dl:
                with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0"]):
                    with patch.object(cli_mod, "_flash"):
                        cli_mod.cmd_update("mainline")
        dl_dest = mock_dl.call_args[0][1]
        assert "mainline" in str(dl_dest)

    def test_port_arg_used_when_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with patch.object(cli_mod, "_download_hex"):
                with patch.object(cli_mod, "_all_mega_ports") as mock_ports:
                    with patch.object(cli_mod, "_flash") as mock_flash:
                        cli_mod.cmd_update("/dev/ttyACM1")
        mock_ports.assert_not_called()
        assert mock_flash.call_args[0][1] == "/dev/ttyACM1"

    def test_cached_hex_skips_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        # pre-populate cache
        cached = tmp_path / "release" / "0.2.0" / "bbb1111" / "firmware.hex"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"\x00" * 100)
        # commit for release/0.2.0 is bbb1111 (last segment of build_key)
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with patch.object(cli_mod, "_download_hex") as mock_dl:
                with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0"]):
                    with patch.object(cli_mod, "_flash"):
                        cli_mod.cmd_update("release/0.2.0")
        mock_dl.assert_not_called()

    def test_unknown_branch_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with pytest.raises(SystemExit):
                cli_mod.cmd_update("release/9.9.9")

    def test_no_boards_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with patch.object(cli_mod, "_download_hex"):
                with patch.object(cli_mod, "_all_mega_ports", return_value=[]):
                    with pytest.raises(SystemExit):
                        cli_mod.cmd_update()

    def test_multiple_boards_without_port_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = self._default_index()
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with patch.object(cli_mod, "_download_hex"):
                with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0", "/dev/ttyACM1"]):
                    with pytest.raises(SystemExit):
                        cli_mod.cmd_update()

    def test_no_release_branch_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        index = _make_index({"mainline": "20250101-120000-aaa0000"})
        with patch.object(cli_mod, "_fetch_index", return_value=index):
            with pytest.raises(SystemExit):
                cli_mod.cmd_update()


# --- _cached_hex path ---

class TestCachedHex:
    def test_cache_path_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli_mod, "CACHE_DIR", tmp_path)
        result = cli_mod._cached_hex("release/0.2.0", "abc1234", "firmware.hex")
        assert result == tmp_path / "release/0.2.0" / "abc1234" / "firmware.hex"


# --- _probe_version ---

class TestProbeVersion:
    def _make_ser(self, lines: list[bytes]) -> MagicMock:
        """Return a mock Serial instance that yields lines on readline()."""
        ser = MagicMock()
        ser.readline.side_effect = lines + [b""] * 100
        return ser

    def _patch_serial(self, ser: MagicMock):
        mock_serial_cls = MagicMock(return_value=ser)
        serial_mod = MagicMock()
        serial_mod.Serial = mock_serial_cls
        return patch.dict("sys.modules", {"serial": serial_mod})

    def test_returns_ver_line_when_present(self):
        ser = self._make_ser([b"VER 0.2.0 release/0.2.0 abc1234\n"])
        with self._patch_serial(ser):
            with patch("time.sleep"), patch("time.time", side_effect=[0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result == "VER 0.2.0 release/0.2.0 abc1234"

    def test_returns_none_when_no_ver_line(self):
        ser = self._make_ser([b"Krabby booted\n", b"FRONT; data\n"])
        with self._patch_serial(ser):
            with patch("time.sleep"):
                with patch("time.time", return_value=999):
                    result = cli_mod._probe_version("/dev/ttyACM0", timeout=0.0)
        assert result is None

    def test_skips_non_ver_lines(self):
        ser = self._make_ser([
            b"Krabby booted\n",
            b"FRONT; FLHY 0.5 512\n",
            b"VER 0.2.0 release/0.2.0 abc1234\n",
        ])
        with self._patch_serial(ser):
            with patch("time.sleep"), patch("time.time", side_effect=[0, 0, 0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result == "VER 0.2.0 release/0.2.0 abc1234"

    def test_sets_dtr_and_rts_false(self):
        ser = self._make_ser([b"VER dev-local dev-local dev-local\n"])
        with self._patch_serial(ser) as patched:
            with patch("time.sleep"), patch("time.time", side_effect=[0, 0, 0, 0, 1]):
                cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert ser.dtr is False
        assert ser.rts is False

    def test_opens_and_closes_port(self):
        ser = self._make_ser([b"VER 0.2.0 release/0.2.0 abc1234\n"])
        with self._patch_serial(ser):
            with patch("time.sleep"), patch("time.time", side_effect=[0, 0, 0, 0, 1]):
                cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        ser.open.assert_called_once()
        ser.close.assert_called_once()

    def test_closes_port_on_exception(self):
        ser = self._make_ser([])
        ser.readline.side_effect = OSError("device disconnected")
        with self._patch_serial(ser):
            with patch("time.sleep"), patch("time.time", side_effect=[0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result is None
        ser.close.assert_called_once()

    def test_returns_none_when_serial_import_missing(self):
        with patch.dict("sys.modules", {"serial": None}):
            result = cli_mod._probe_version("/dev/ttyACM0")
        assert result is None

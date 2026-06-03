"""Unit tests for firmware.cli (--show / --update)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import urllib.error

import firmware.cli as cli_mod
from firmware.manifest import BranchBuild, BranchEntry, FirmwareIndex


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
            with patch.object(cli_mod, "_probe_version", return_value=("VER 0.2.0 release/0.2.0 abc1234", None)):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "/dev/ttyACM0" in out
        assert "0.2.0" in out

    def test_shows_no_version_response_when_probe_fails(self, capsys):
        index = _make_index({"mainline": "20250101-120000-abc1234"})
        with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyACM0"]):
            with patch.object(cli_mod, "_probe_version", return_value=(None, None)):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "no version response" in out

    def test_role_hint_labels_follower_as_left(self, capsys):
        index = _make_index({"release/0.2.8": "20250101-120000-abc1234"})
        with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyUSB0"]):
            with patch.object(cli_mod, "_probe_version",
                              return_value=("VER 0.2.8|-|- release/0.2.8|-|- abc1234|-|-", "left")):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "left: 0.2.8" in out
        assert "primary" not in out

    def test_role_hint_labels_follower_as_right(self, capsys):
        index = _make_index({"release/0.2.8": "20250101-120000-abc1234"})
        with patch.object(cli_mod, "_all_mega_ports", return_value=["/dev/ttyUSB1"]):
            with patch.object(cli_mod, "_probe_version",
                              return_value=("VER 0.2.8|-|- release/0.2.8|-|- abc1234|-|-", "right")):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "right: 0.2.8" in out
        assert "primary" not in out

    def test_combined_ver_fills_all_three_ports(self, capsys):
        """Leader's combined VER shows all three role slots; port annotation when ROLE_HINT present."""
        index = _make_index({"release/0.2.9": "20250101-120000-abc1234"})
        combined = "VER 0.2.9|0.2.9|0.2.9 release/0.2.9|release/0.2.9|release/0.2.9 abc1234|abc1234|abc1234"

        def fake_probe(port):
            if port == "/dev/ttyACM0":
                return (combined, "front")
            elif port == "/dev/ttyUSB0":
                return (None, "left")
            else:
                return (None, "right")

        with patch.object(cli_mod, "_all_mega_ports",
                          return_value=["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyUSB1"]):
            with patch.object(cli_mod, "_probe_version", side_effect=fake_probe):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "primary (/dev/ttyACM0): 0.2.9" in out
        assert "left (/dev/ttyUSB0): 0.2.9" in out
        assert "right (/dev/ttyUSB1): 0.2.9" in out
        assert "no version response" not in out

    def test_combined_ver_no_role_hints_shows_correct_slots(self, capsys):
        """Old firmware without ROLE_HINT: combined VER slots shown by role, not port mapping."""
        index = _make_index({"release/0.2.9": "20250101-120000-abc1234"})
        combined = "VER 0.2.9|0.2.8|0.2.8 release/0.2.9|release/0.2.8|release/0.2.8 abc1234|def5678|def5678"

        def fake_probe(port):
            # Leader returns combined VER; followers time out. No ROLE_HINT on any board.
            if port == "/dev/ttyACM0":
                return (combined, None)
            return (None, None)

        with patch.object(cli_mod, "_all_mega_ports",
                          return_value=["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyUSB1"]):
            with patch.object(cli_mod, "_probe_version", side_effect=fake_probe):
                with patch.object(cli_mod, "_fetch_index", return_value=index):
                    cli_mod.cmd_show()
        out = capsys.readouterr().out
        assert "primary: 0.2.9" in out
        assert "left: 0.2.8" in out
        assert "right: 0.2.8" in out
        # Old bug: all three ports mapped to slot 0, showing 0.2.9 for left/right too
        assert "left: 0.2.9" not in out
        assert "right: 0.2.9" not in out

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


# --- cmd_show <branch> (per-branch build history) ---

class TestCmdShowBranch:
    def _builds(self):
        # Intentionally out of order to prove the command sorts newest-first.
        return [
            BranchBuild(build_key="20250101-120000-aaa0000", commit="aaa0000",
                        commit_date="2025-01-01", ver_string="0.2.9 release/0.2.9 aaa0000",
                        hex_url="h", manifest_url="m"),
            BranchBuild(build_key="20250301-120000-ccc2222", commit="ccc2222",
                        commit_date="2025-03-01", ver_string="0.2.9 release/0.2.9 ccc2222",
                        hex_url="h", manifest_url="m"),
        ]

    def test_lists_builds_newest_first(self, capsys):
        # Feed an UNSORTED builds.json through the real _fetch_builds/parse_builds so
        # this exercises the actual newest-first sort, not a pre-sorted fixture.
        unsorted_doc = {
            "schema_version": 1, "branch": "release/0.2.9", "builds": [
                {"build_key": "20250101-120000-aaa0000", "commit": "aaa0000",
                 "commit_date": "2025-01-01", "ver_string": "0.2.9 release/0.2.9 aaa0000",
                 "hex_url": "h", "manifest_url": "m"},
                {"build_key": "20250301-120000-ccc2222", "commit": "ccc2222",
                 "commit_date": "2025-03-01", "ver_string": "0.2.9 release/0.2.9 ccc2222",
                 "hex_url": "h", "manifest_url": "m"},
            ],
        }
        with patch.object(cli_mod, "_fetch_json", return_value=unsorted_doc):
            cli_mod.cmd_show("release/0.2.9")
        out = capsys.readouterr().out
        assert out.index("ccc2222") < out.index("aaa0000")  # sorted by code, not fixture
        assert "release/0.2.9" in out

    def test_branch_listing_does_not_probe_boards(self):
        builds = sorted(self._builds(), key=lambda b: b.build_key, reverse=True)
        with patch.object(cli_mod, "_all_mega_ports") as mock_ports:
            with patch.object(cli_mod, "_fetch_builds", return_value=builds):
                cli_mod.cmd_show("release/0.2.9")
        mock_ports.assert_not_called()

    def test_no_builds_message(self, capsys):
        with patch.object(cli_mod, "_fetch_builds", return_value=[]):
            cli_mod.cmd_show("release/0.2.9")
        assert "No builds found" in capsys.readouterr().out

    @pytest.mark.parametrize("code", [403, 404])
    def test_missing_branch_exits_with_hint(self, code):
        # Real public bucket returns 403 for an absent key (no ListBucket); 404 if listable.
        err = urllib.error.HTTPError("u", code, "no", None, None)
        with patch.object(cli_mod, "_fetch_builds", side_effect=err):
            with pytest.raises(SystemExit) as ei:
                cli_mod.cmd_show("release/9.9.9")
        assert "No build history" in str(ei.value)  # the friendly hint, not a raw HTTP error

    def test_real_error_uses_generic_message(self):
        err = urllib.error.HTTPError("u", 500, "boom", None, None)
        with patch.object(cli_mod, "_fetch_builds", side_effect=err):
            with pytest.raises(SystemExit) as ei:
                cli_mod.cmd_show("release/0.2.9")
        assert "Could not fetch builds" in str(ei.value)


class TestPage:
    class _FakeOut:
        def __init__(self, tty):
            self._tty = tty
            self.written = ""

        def isatty(self):
            return self._tty

        def write(self, s):
            self.written += s

    def test_non_tty_writes_plainly(self, monkeypatch):
        out = self._FakeOut(tty=False)
        monkeypatch.setattr(cli_mod.sys, "stdout", out)
        cli_mod._page("hello")
        assert out.written == "hello\n"

    def test_tty_routes_through_pager(self, monkeypatch):
        sent = {}

        class FakeProc:
            returncode = 0

            def communicate(self, text):
                sent["text"] = text

        monkeypatch.setattr(cli_mod.sys, "stdout", self._FakeOut(tty=True))
        monkeypatch.setattr(cli_mod.subprocess, "Popen", lambda *a, **k: FakeProc())
        cli_mod._page("rows\n")
        assert sent["text"] == "rows\n"

    def test_pager_nonzero_exit_falls_back_to_stdout(self, monkeypatch):
        out = self._FakeOut(tty=True)

        class FakeProc:
            returncode = 127  # e.g. PAGER/less missing → shell "command not found"

            def communicate(self, text):
                pass

        monkeypatch.setattr(cli_mod.sys, "stdout", out)
        monkeypatch.setattr(cli_mod.subprocess, "Popen", lambda *a, **k: FakeProc())
        cli_mod._page("rows")
        assert out.written == "rows\n"  # not silently swallowed


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
        # Force `_flash`'s "no flash tool" branch so this test can NEVER shell out to a
        # real avrdude/arduino-cli against a connected board. Without this, on a bench
        # machine that has avrdude installed the test would reset/poke /dev/ttyACM* and
        # hang. This matches the deterministic CI path (clean runner, no flash tools).
        monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: None)
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
        """Return a mock Serial context manager that yields lines on readline()."""
        ser = MagicMock()
        ser.__enter__ = MagicMock(return_value=ser)
        ser.__exit__ = MagicMock(return_value=False)
        ser.readline.side_effect = lines + [b""] * 100
        return ser

    def _patch_serial(self, ser: MagicMock):
        serial_mod = MagicMock()
        serial_mod.Serial = MagicMock(return_value=ser)
        return patch.dict("sys.modules", {"serial": serial_mod})

    def test_returns_ver_line_after_krabby_ready(self):
        ser = self._make_ser([
            b"Krabby Ready PINS_REV3.\r\n",
            b"VER 0.2.0 release/0.2.0 abc1234\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result[0] == "VER 0.2.0 release/0.2.0 abc1234"
        assert result[1] is None

    def test_sends_v_only_after_krabby_ready(self):
        ser = self._make_ser([
            b"--- SYNC ---\r\n",
            b"Krabby Ready PINS_REV3.\r\n",
            b"VER 0.2.0 release/0.2.0 abc1234\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 0, 0, 1]):
                cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        # V should have been sent exactly once (after Krabby Ready)
        assert ser.write.call_count == 1
        ser.write.assert_called_with(b"V\n")

    def test_retries_v_on_empty_readline_when_ready(self):
        ser = self._make_ser([
            b"Krabby Ready PINS_REV3.\r\n",
            b"",  # empty — triggers retry V send
            b"VER 0.2.0 release/0.2.0 abc1234\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result[0] == "VER 0.2.0 release/0.2.0 abc1234"
        assert ser.write.call_count == 2  # once on Ready, once on empty

    def test_returns_none_when_no_krabby_ready(self):
        ser = self._make_ser([b"--- SYNC ---\r\n"])
        with self._patch_serial(ser):
            with patch("time.time", return_value=999):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=0.0)
        assert result == (None, None)

    def test_returns_none_when_no_ver_after_ready(self):
        ser = self._make_ser([
            b"Krabby Ready PINS_REV3.\r\n",
            b"FRONT; data\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 999]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=0.0)
        assert result == (None, None)

    def test_handles_real_ver_format(self):
        ser = self._make_ser([
            b"Krabby Ready PINS_REV2_UNO_V01. FLHY\r\n",
            b"VER 0.2.0|-|- release/0.2.0|-|- ac66d5e|-|-\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result[0] == "VER 0.2.0|-|- release/0.2.0|-|- ac66d5e|-|-"

    def test_returns_none_on_serial_exception(self):
        ser = self._make_ser([])
        ser.readline.side_effect = OSError("device disconnected")
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result == (None, None)

    def test_returns_none_when_serial_import_missing(self):
        with patch.dict("sys.modules", {"serial": None}):
            result = cli_mod._probe_version("/dev/ttyACM0")
        assert result == (None, None)

    def test_captures_role_hint_before_krabby_ready(self):
        ser = self._make_ser([
            b"--- SYNC ---\r\n",
            b"ROLE_HINT: LEFT\r\n",
            b"Krabby Ready PINS_REV3.\r\n",
            b"VER 0.2.8|-|- release/0.2.8|-|- abc1234|-|-\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result[0].startswith("VER ")
        assert result[1] == "left"

    def test_role_hint_right_captured(self):
        ser = self._make_ser([
            b"ROLE_HINT: RIGHT\r\n",
            b"Krabby Ready PINS_REV3.\r\n",
            b"VER 0.2.8|-|- release/0.2.8|-|- abc1234|-|-\r\n",
        ])
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0, 0, 0, 0, 0, 1]):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=1.0)
        assert result[1] == "right"

    def test_early_exit_after_v_retry_limit_with_no_ver(self):
        # Follower board: "Krabby Ready" arrives but VER never comes because the
        # follower responds to V on its UART uplink, not USB.
        ser = self._make_ser([
            b"ROLE_HINT: LEFT\r\n",
            b"Krabby Ready PINS_REV3.\r\n",
        ] + [b""] * (cli_mod._PROBE_V_RETRY_LIMIT + 1))
        with self._patch_serial(ser):
            with patch("time.time", side_effect=[0] + [0.1] * 30):
                result = cli_mod._probe_version("/dev/ttyACM0", timeout=10.0)
        assert result == (None, "left")
        # Initial V sent on Krabby Ready + one V per retry before cutoff
        assert ser.write.call_count == cli_mod._PROBE_V_RETRY_LIMIT + 1

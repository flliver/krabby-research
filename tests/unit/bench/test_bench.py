"""Unit tests for krabby-bench (AC10)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parents[3] / "bench"))

from krabby_bench._config import Config, AlertConfig, SmtpConfig, GithubConfig, load_config
from krabby_bench._state import load_state, save_state
from krabby_bench._smoke import SmokeResult, _parse_ports, _parse_versions, run_smoke
from krabby_bench._alert import should_alert, send_alert
from krabby_bench.watchdog import poll_once


# ---------------------------------------------------------------------------
# _state: roundtrip
# ---------------------------------------------------------------------------

class TestState:
    def test_load_returns_empty_when_missing(self, tmp_path):
        assert load_state(tmp_path / "state.json") == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        save_state(path, {"last_tested_digest": "sha256:abc", "last_alert_key": "k"})
        assert load_state(path)["last_tested_digest"] == "sha256:abc"

    def test_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("not-json")
        assert load_state(path) == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "state.json"
        save_state(path, {"x": 1})
        assert path.exists()


# ---------------------------------------------------------------------------
# _config: load
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults_when_no_file(self, tmp_path):
        cfg = load_config(tmp_path / "config.toml")
        assert cfg.ecr.tag == "mainline-latest"
        assert cfg.alert.mode == "email"

    def test_toml_overrides_ecr_tag(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text('[ecr]\ntag = "release-latest"\n')
        cfg = load_config(p)
        assert cfg.ecr.tag == "release-latest"
        assert cfg.ecr.repo.startswith("public.ecr.aws")  # other fields keep default

    def test_toml_overrides_alert_mode(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text('[alert]\nmode = "both"\n')
        cfg = load_config(p)
        assert cfg.alert.mode == "both"

    def test_state_path_override(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(f'state_path = "{tmp_path}/mystate.json"\n')
        cfg = load_config(p)
        assert cfg.state_path == tmp_path / "mystate.json"


# ---------------------------------------------------------------------------
# _ecr: digest polling + dedup
# ---------------------------------------------------------------------------

class TestEcrDigestPoll:
    def test_get_digest_fetches_via_registry_api(self):
        import hashlib

        class _Resp:
            def __init__(self, body):
                self._body = body if isinstance(body, bytes) else body.encode()
            def read(self):
                return self._body
            def __enter__(self): return self
            def __exit__(self, *_): pass

        manifest_body = b'{"fake": "manifest"}'
        expected_digest = "sha256:" + hashlib.sha256(manifest_body).hexdigest()

        token_resp = _Resp('{"token": "test-token"}')
        manifest_resp = _Resp(manifest_body)

        with patch("krabby_bench._ecr.urllib.request.urlopen", side_effect=[token_resp, manifest_resp]):
            from krabby_bench._ecr import get_digest
            digest = get_digest("public.ecr.aws/t7t7b3i3/krabby-locomotion", "mainline-latest")

        assert digest == expected_digest


# ---------------------------------------------------------------------------
# _smoke: parsing and orchestration
# ---------------------------------------------------------------------------

class TestParseVersions:
    def test_extracts_three_versions(self):
        output = (
            "  /dev/ttyACM0  primary: 0.2.0 (mainline abc1234)\n"
            "  /dev/ttyUSB0  left: 0.2.0 (mainline abc1234)\n"
            "  /dev/ttyUSB1  right: 0.2.0 (mainline abc1234)\n"
        )
        assert _parse_versions(output) == ["0.2.0", "0.2.0", "0.2.0"]

    def test_returns_empty_for_no_match(self):
        assert _parse_versions("no boards here") == []

    def test_ignores_no_version_response(self):
        output = (
            "  /dev/ttyUSB0  (no version response)\n"
            "  /dev/ttyACM0  primary: 0.2.9 (release abc) | left: 0.2.9 (release abc) | right: 0.2.9 (release abc)\n"
        )
        assert _parse_versions(output) == ["0.2.9", "0.2.9", "0.2.9"]


class TestParsePorts:
    def test_extracts_three_ports(self):
        output = (
            "  /dev/ttyACM0  primary: 0.2.0 (mainline abc)\n"
            "  /dev/ttyUSB0  left: 0.2.0 (mainline abc)\n"
            "  /dev/ttyUSB1  right: 0.2.0 (mainline abc)\n"
        )
        assert _parse_ports(output) == ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyUSB1"]

    def test_returns_empty_for_no_match(self):
        assert _parse_ports("no boards here") == []

    def test_shared_port_counted_once(self):
        # primary+left share ttyACM0 on one line, right on ttyUSB0
        output = (
            "  /dev/ttyUSB0  right: 0.2.9 (release/0.2.9 abc)\n"
            "  /dev/ttyACM0  primary: 0.2.9 (release/0.2.9 abc) | left: 0.2.9 (release/0.2.9 abc)\n"
        )
        assert _parse_ports(output) == ["/dev/ttyUSB0", "/dev/ttyACM0"]


class TestRunSmoke:
    # New flow: show(ports) → update×N → show(versions)
    _SHOW_3 = (
        "  /dev/ttyACM0  primary: 0.2.0 (mainline abc)\n"
        "  /dev/ttyUSB0  left: 0.2.0 (mainline abc)\n"
        "  /dev/ttyUSB1  right: 0.2.0 (mainline abc)\n"
    )
    _SHOW_1 = "  /dev/ttyACM0  primary: 0.2.0 (mainline abc)\n"

    def _mock_firmware(self, side_effects):
        """side_effects: list of (rc, stdout, stderr) per _run_firmware call."""
        return patch(
            "krabby_bench._smoke._run_firmware",
            side_effect=side_effects,
        )

    def _mock_s3(self, ver_string="0.2.0"):
        return patch("krabby_bench._smoke._fetch_expected_ver", return_value=ver_string)

    def test_passes_when_all_boards_match_s3(self):
        # show(ports=3) + update×3 + show(versions=3)
        calls = [(0, self._SHOW_3, ""), (0, "", ""), (0, "", ""), (0, "", ""), (0, self._SHOW_3, "")]
        with self._mock_firmware(calls):
            with self._mock_s3("0.2.0"):
                result = run_smoke("mainline", "myimage:tag")
        assert result.ok
        assert result.ver_observed == ["0.2.0", "0.2.0", "0.2.0"]
        assert result.ver_expected == "0.2.0"

    def test_fails_on_firmware_update_nonzero(self):
        # show(ports=3) + update fails on first port
        calls = [(0, self._SHOW_3, ""), (1, "", "flash error")]
        with self._mock_firmware(calls):
            result = run_smoke("mainline", "myimage:tag")
        assert not result.ok
        assert result.step == "firmware_update"

    def test_fails_when_fewer_than_three_ports(self):
        # show finds only 1 port — fail before any updates
        calls = [(0, self._SHOW_1, "")]
        with self._mock_firmware(calls):
            result = run_smoke("mainline", "myimage:tag")
        assert not result.ok
        assert result.step == "firmware_show_ports"

    def test_fails_daisy_chain_masks_missing_usb(self):
        # show finds 1 port but daisy chain reports 3 versions — should still fail
        show_3_versions_1_port = (
            "  /dev/ttyACM0  primary: 0.2.0 (mainline abc) | "
            "left: 0.2.0 (mainline abc) | right: 0.2.0 (mainline abc)\n"
        )
        calls = [(0, show_3_versions_1_port, "")]
        with self._mock_firmware(calls):
            result = run_smoke("mainline", "myimage:tag")
        assert not result.ok
        assert result.step == "firmware_show_ports"

    def test_fails_when_boards_disagree(self):
        show_disagree = (
            "  /dev/ttyACM0  primary: 0.2.0 (mainline abc)\n"
            "  /dev/ttyUSB0  left: 0.1.9 (mainline abc)\n"
            "  /dev/ttyUSB1  right: 0.2.0 (mainline abc)\n"
        )
        calls = [(0, self._SHOW_3, ""), (0, "", ""), (0, "", ""), (0, "", ""), (0, show_disagree, "")]
        with self._mock_firmware(calls):
            result = run_smoke("mainline", "myimage:tag")
        assert not result.ok
        assert result.step == "ver_mismatch"

    def test_fails_when_ver_differs_from_s3(self):
        calls = [(0, self._SHOW_3, ""), (0, "", ""), (0, "", ""), (0, "", ""), (0, self._SHOW_3, "")]
        with self._mock_firmware(calls):
            with self._mock_s3("0.3.0"):
                result = run_smoke("mainline", "myimage:tag")
        assert not result.ok
        assert result.step == "ver_mismatch_s3"


# ---------------------------------------------------------------------------
# _alert: dedup + dispatch
# ---------------------------------------------------------------------------

class TestShouldAlert:
    def test_alerts_when_state_empty(self):
        assert should_alert({}, "sha256:abc:firmware_update", 3600)

    def test_alerts_when_key_changes(self):
        state = {
            "last_alert_key": "sha256:old:firmware_show",
            "last_alert_at": datetime.now(timezone.utc).isoformat(),
        }
        assert should_alert(state, "sha256:new:firmware_update", 3600)

    def test_suppresses_within_dedup_window(self):
        alert_key = "sha256:abc:firmware_update"
        state = {
            "last_alert_key": alert_key,
            "last_alert_at": datetime.now(timezone.utc).isoformat(),
        }
        assert not should_alert(state, alert_key, 3600)

    def test_allows_after_dedup_window_expires(self):
        alert_key = "sha256:abc:firmware_update"
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=3601)).isoformat()
        state = {"last_alert_key": alert_key, "last_alert_at": old_time}
        assert should_alert(state, alert_key, 3600)


class TestSendAlert:
    def _make_result(self):
        return SmokeResult(
            ok=False, step="firmware_update", detail="exit 1",
            stdout="out", stderr="err",
        )

    def test_smtp_called_for_email_mode(self):
        cfg_alert = AlertConfig(mode="email")
        cfg_smtp = SmtpConfig(host="smtp.example.com", port=587, user="u",
                              password="p", from_addr="f@e.com", to_addr="t@e.com")
        cfg_github = GithubConfig()

        with patch("krabby_bench._alert.smtplib.SMTP") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            send_alert(cfg_alert, cfg_smtp, cfg_github, "sha256:abc", self._make_result())

        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("u", "p")
        mock_smtp.send_message.assert_called_once()

    def test_github_called_for_github_mode(self):
        cfg_alert = AlertConfig(mode="github")
        cfg_smtp = SmtpConfig()
        cfg_github = GithubConfig(repo="owner/repo", token="tok123")

        with patch("krabby_bench._alert.requests.post") as mock_post:
            mock_post.return_value.raise_for_status = MagicMock()
            send_alert(cfg_alert, cfg_smtp, cfg_github, "sha256:abc", self._make_result())

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "owner/repo" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["labels"] == ["bench-alarm"]

    def test_both_mode_calls_smtp_and_github(self):
        cfg_alert = AlertConfig(mode="both")
        cfg_smtp = SmtpConfig(host="h", port=587, user="u", password="p",
                              from_addr="f@e", to_addr="t@e")
        cfg_github = GithubConfig(repo="owner/repo", token="tok")

        with patch("krabby_bench._alert.smtplib.SMTP") as mock_smtp_cls, \
             patch("krabby_bench._alert.requests.post") as mock_post:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_post.return_value.raise_for_status = MagicMock()
            send_alert(cfg_alert, cfg_smtp, cfg_github, "sha256:abc", self._make_result())

        mock_smtp.send_message.assert_called_once()
        mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# watchdog: poll_once orchestration
# ---------------------------------------------------------------------------

class TestPollOnce:
    def _config(self, tmp_path) -> Config:
        cfg = Config()
        cfg.state_path = tmp_path / "state.json"
        return cfg

    def test_skips_when_digest_unchanged(self, tmp_path):
        cfg = self._config(tmp_path)
        state = {"last_tested_digest": "sha256:same"}
        with patch("krabby_bench.watchdog.get_digest", return_value="sha256:same") as mock_get:
            new_state = poll_once(cfg, state)
        assert new_state == state

    def test_runs_smoke_on_new_digest(self, tmp_path):
        cfg = self._config(tmp_path)
        state = {"last_tested_digest": "sha256:old"}
        smoke_ok = SmokeResult(ok=True, ver_observed=["0.2.0", "0.2.0", "0.2.0"], ver_expected="0.2.0")

        with patch("krabby_bench.watchdog.get_digest", return_value="sha256:new"), \
             patch("krabby_bench.watchdog.subprocess.run"), \
             patch("krabby_bench.watchdog.run_smoke", return_value=smoke_ok), \
             patch("krabby_bench.watchdog._get_image_ref", return_value="myrepo:tag"):
            new_state = poll_once(cfg, state)

        assert new_state["last_tested_digest"] == "sha256:new"
        assert "last_alert_at" not in new_state

    def test_sends_alert_on_smoke_failure(self, tmp_path):
        cfg = self._config(tmp_path)
        state = {}
        smoke_fail = SmokeResult(ok=False, step="firmware_update", detail="exit 1")

        with patch("krabby_bench.watchdog.get_digest", return_value="sha256:new"), \
             patch("krabby_bench.watchdog.subprocess.run"), \
             patch("krabby_bench.watchdog.run_smoke", return_value=smoke_fail), \
             patch("krabby_bench.watchdog._get_image_ref", return_value="myrepo:tag"), \
             patch("krabby_bench.watchdog.send_alert") as mock_alert:
            new_state = poll_once(cfg, state)

        mock_alert.assert_called_once()
        assert "last_alert_at" in new_state

    def test_dedup_suppresses_repeat_alert(self, tmp_path):
        cfg = self._config(tmp_path)
        alert_key = "sha256:new:firmware_update"
        state = {
            "last_tested_digest": "sha256:old",
            "last_alert_key": alert_key,
            "last_alert_at": datetime.now(timezone.utc).isoformat(),
        }
        smoke_fail = SmokeResult(ok=False, step="firmware_update", detail="exit 1")

        with patch("krabby_bench.watchdog.get_digest", return_value="sha256:new"), \
             patch("krabby_bench.watchdog.subprocess.run"), \
             patch("krabby_bench.watchdog.run_smoke", return_value=smoke_fail), \
             patch("krabby_bench.watchdog._get_image_ref", return_value="myrepo:tag"), \
             patch("krabby_bench.watchdog.send_alert") as mock_alert:
            new_state = poll_once(cfg, state)

        mock_alert.assert_not_called()

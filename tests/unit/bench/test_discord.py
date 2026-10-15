"""Unit tests for Discord CI notification formatting."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[3] / "bench"))

from krabby_bench._discord import (
    default_commit_subject,
    format_discord_payload,
    format_failed_steps,
    format_job_results,
    list_failed_steps_from_run,
    notify_ci_result,
    previous_workflow_run_conclusion,
    should_collapse_consecutive_pass,
)


class TestFormatDiscordPayload:
    def test_pass_includes_branch_and_jobs(self):
        payload = format_discord_payload(
            ok=True,
            title="artifact-health",
            commit="abcdef0123456789",
            branch="main",
            event="workflow_run",
            job_results={
                "packages": "success",
                "locomotion-image": "success",
                "firmware-artifact": "success",
            },
            run_url="https://example.com/run/1",
        )
        desc = payload["embeds"][0]["description"]
        assert "**PASS** — artifact-health" in desc
        assert "Commit: `abcdef0`" in desc
        assert "Branch: `main` (workflow_run)" in desc
        assert "• packages: **OK**" in desc
        assert "[View run](https://example.com/run/1)" in desc
        assert payload["content"] == "PASS: artifact-health"

    def test_fail_includes_stage_and_steps(self):
        payload = format_discord_payload(
            ok=False,
            title="publish-firmware",
            commit="deadbeef",
            subject="fix: overflow path prints ERR",
            branch="main",
            event="push",
            failed_stage="build-and-publish",
            failed_steps=["build-and-publish › Compile firmware"],
            detail="arduino-cli exited 1",
            help_url="https://github.com/flliver/krabby-research/blob/main/docs/CI-FAILURES.md",
        )
        desc = payload["embeds"][0]["description"]
        assert "**FAIL** — publish-firmware" in desc
        assert "Commit: `deadbee` fix: overflow path prints ERR" in desc
        assert "Failed stage: **build-and-publish**" in desc
        assert "Failed steps: **build-and-publish › Compile firmware**" in desc
        assert "[CI failures playbook](https://github.com/flliver/krabby-research/blob/main/docs/CI-FAILURES.md)" in desc
        assert "arduino-cli exited 1" in desc
        assert payload["content"] == "FAIL: publish-firmware (build-and-publish)"


class TestDefaultCommitSubject:
    def test_clips_long_subject(self, monkeypatch):
        long = "x" * 200

        def fake_check_output(*_a, **_k):
            return long + "\n"

        monkeypatch.setattr(
            "subprocess.check_output",
            fake_check_output,
        )
        # Import path used inside default_commit_subject
        import subprocess as sp

        monkeypatch.setattr(sp, "check_output", fake_check_output)
        out = default_commit_subject(max_len=40)
        assert len(out) == 40
        assert out.endswith("…")


class TestFormatHelpers:
    def test_format_job_results_empty(self):
        assert format_job_results(None) == ""
        assert format_job_results({}) == ""

    def test_format_failed_steps(self):
        assert format_failed_steps(None) == ""
        assert "A › B" in format_failed_steps(["A › B"])


class TestListFailedSteps:
    def test_returns_empty_without_env(self, monkeypatch):
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert list_failed_steps_from_run() == []

    def test_parses_failed_steps(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
        monkeypatch.setenv("GITHUB_RUN_ID", "99")
        monkeypatch.setenv("GITHUB_TOKEN", "tok")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "jobs": [
                {
                    "name": "build-and-publish",
                    "conclusion": "failure",
                    "steps": [
                        {"name": "Checkout", "conclusion": "success"},
                        {"name": "Compile firmware", "conclusion": "failure"},
                        {"name": "Discord on failure", "conclusion": "skipped"},
                    ],
                }
            ]
        }

        with patch("krabby_bench._discord.requests.get", return_value=mock_resp) as get:
            steps = list_failed_steps_from_run()
            assert steps == ["build-and-publish › Compile firmware"]
            get.assert_called_once()


class TestCollapseConsecutivePass:
    def test_fail_never_collapses(self):
        assert (
            should_collapse_consecutive_pass(
                ok=False,
                collapse_consecutive_pass=True,
                previous_conclusion="success",
            )
            is False
        )

    def test_pass_collapses_after_success(self):
        assert (
            should_collapse_consecutive_pass(
                ok=True,
                collapse_consecutive_pass=True,
                previous_conclusion="success",
            )
            is True
        )

    def test_pass_posts_after_failure(self):
        assert (
            should_collapse_consecutive_pass(
                ok=True,
                collapse_consecutive_pass=True,
                previous_conclusion="failure",
            )
            is False
        )

    def test_pass_posts_when_no_prior(self):
        assert (
            should_collapse_consecutive_pass(
                ok=True,
                collapse_consecutive_pass=True,
                previous_conclusion=None,
            )
            is False
        )


class TestPreviousWorkflowRunConclusion:
    def test_returns_empty_without_env(self, monkeypatch):
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert previous_workflow_run_conclusion() is None

    def test_skips_current_run(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
        monkeypatch.setenv("GITHUB_RUN_ID", "100")
        monkeypatch.setenv("GITHUB_TOKEN", "tok")

        run_resp = MagicMock()
        run_resp.raise_for_status = MagicMock()
        run_resp.json.return_value = {"workflow_id": 42}

        list_resp = MagicMock()
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = {
            "workflow_runs": [
                {"id": 100, "conclusion": "success"},
                {"id": 99, "conclusion": "failure"},
            ]
        }

        with patch(
            "krabby_bench._discord.requests.get",
            side_effect=[run_resp, list_resp],
        ):
            assert previous_workflow_run_conclusion() == "failure"


class TestNotifyCiResult:
    def test_fetches_steps_on_failure(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/hook")
        with patch(
            "krabby_bench._discord.list_failed_steps_from_run",
            return_value=["build › Smoke"],
        ), patch(
            "krabby_bench._discord.default_commit_subject",
            return_value="fix: smoke teleop-ip",
        ), patch("krabby_bench._discord.post_discord", return_value=True) as post:
            assert notify_ci_result(ok=False, title="publish-locomotion") is True
            kwargs = post.call_args.kwargs
            assert kwargs["failed_steps"] == ["build › Smoke"]
            assert kwargs["failed_stage"] == "build"
            assert kwargs["subject"] == "fix: smoke teleop-ip"
            assert "CI-FAILURES.md" in kwargs["help_url"]

    def test_skips_consecutive_pass(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/hook")
        with patch(
            "krabby_bench._discord.previous_workflow_run_conclusion",
            return_value="success",
        ), patch("krabby_bench._discord.post_discord", return_value=True) as post:
            assert (
                notify_ci_result(
                    ok=True,
                    title="artifact-health",
                    collapse_consecutive_pass=True,
                )
                is False
            )
            post.assert_not_called()

    def test_posts_pass_after_prior_failure(self, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/hook")
        with patch(
            "krabby_bench._discord.previous_workflow_run_conclusion",
            return_value="failure",
        ), patch(
            "krabby_bench._discord.default_commit_subject",
            return_value="chore: bump",
        ), patch("krabby_bench._discord.post_discord", return_value=True) as post:
            assert (
                notify_ci_result(
                    ok=True,
                    title="artifact-health",
                    collapse_consecutive_pass=True,
                )
                is True
            )
            post.assert_called_once()

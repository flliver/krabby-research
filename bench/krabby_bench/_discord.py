"""Discord webhook notifications for harness / CI.

Posts on success and failure. Skips cleanly when DISCORD_WEBHOOK_URL is unset
so local and stubbed-CI runs do not require a secret.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)


def resolve_webhook_url(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get("DISCORD_WEBHOOK_URL") or "").strip()


def default_run_url() -> str:
    """Build Actions run URL from GITHUB_* env when present."""
    server = (os.environ.get("GITHUB_SERVER_URL") or "").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    run_id = os.environ.get("GITHUB_RUN_ID") or ""
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return ""


def default_commit_sha() -> str:
    return (os.environ.get("GITHUB_SHA") or "").strip()


def format_discord_payload(
    *,
    ok: bool,
    title: str = "krabby-bench",
    commit: str = "",
    subject: str = "",
    failed_stage: Optional[str] = None,
    run_url: str = "",
    detail: str = "",
) -> dict:
    """Build a Discord webhook JSON body (content + embed)."""
    status = "PASS" if ok else "FAIL"
    color = 0x2ECC71 if ok else 0xE74C3C
    short_sha = commit[:7] if commit else ""
    lines = [f"**{status}** — {title}"]
    if short_sha or subject:
        commit_line = short_sha
        if subject:
            commit_line = f"{short_sha} {subject}".strip() if short_sha else subject
        lines.append(f"Commit: `{commit_line}`")
    if not ok and failed_stage:
        lines.append(f"Failed stage: **{failed_stage}**")
    if run_url:
        lines.append(f"[View run]({run_url})")
    if detail:
        # Discord embed description soft limit ~4096; keep actionable, not a full log.
        clipped = detail if len(detail) <= 1500 else detail[:1500] + "…"
        lines.append(clipped)

    description = "\n".join(lines)
    return {
        "content": f"{status}: {title}" + (f" ({failed_stage})" if failed_stage else ""),
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color,
            }
        ],
    }


def post_discord(
    webhook_url: Optional[str] = None,
    *,
    ok: bool,
    title: str = "krabby-bench",
    commit: str = "",
    subject: str = "",
    failed_stage: Optional[str] = None,
    run_url: str = "",
    detail: str = "",
) -> bool:
    """POST a result to Discord.

    Returns True if a message was sent, False if skipped (no webhook).
    Network/HTTP errors are logged; returns False (does not raise) so harness
    exit codes stay driven by the test result, not notification delivery.
    """
    url = resolve_webhook_url(webhook_url)
    if not url:
        log.info("Discord skipped: no webhook (set DISCORD_WEBHOOK_URL)")
        return False

    payload = format_discord_payload(
        ok=ok,
        title=title,
        commit=commit or default_commit_sha(),
        subject=subject,
        failed_stage=failed_stage,
        run_url=run_url or default_run_url(),
        detail=detail,
    )
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        log.info("Discord notification sent (%s)", "PASS" if ok else "FAIL")
        return True
    except Exception:
        log.error("Discord notification failed", exc_info=True)
        return False


def notify_harness_result(
    *,
    ok: bool,
    failed_stage: Optional[str] = None,
    detail: str = "",
    run_url: str = "",
    commit: str = "",
    subject: str = "",
    webhook_url: Optional[str] = None,
) -> bool:
    """Notify Discord about a four-stage harness outcome."""
    return post_discord(
        webhook_url,
        ok=ok,
        title="krabby-bench harness",
        commit=commit,
        subject=subject,
        failed_stage=failed_stage,
        run_url=run_url,
        detail=detail,
    )

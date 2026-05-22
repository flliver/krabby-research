"""Alert dispatch: SMTP email and/or GitHub Issue."""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

import requests

from krabby_bench._config import AlertConfig, GithubConfig, SmtpConfig
from krabby_bench._smoke import SmokeResult

log = logging.getLogger(__name__)


def should_alert(state: dict, alert_key: str, dedup_window: int) -> bool:
    if state.get("last_alert_key") != alert_key:
        return True
    last_at = state.get("last_alert_at")
    if not last_at:
        return True
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_at)).total_seconds()
    return elapsed >= dedup_window


def send_alert(
    config_alert: AlertConfig,
    config_smtp: SmtpConfig,
    config_github: GithubConfig,
    digest: str,
    result: SmokeResult,
) -> None:
    body = _format_body(digest, result)
    title = f"[krabby-bench] Smoke failure: {result.step} ({digest[:16]})"

    if config_alert.mode in ("email", "both"):
        if config_smtp.host:
            try:
                _send_smtp(config_smtp, title, body)
                log.info("Alert email sent to %s", config_smtp.to_addr)
            except Exception:
                log.error("SMTP alert failed", exc_info=True)
    if config_alert.mode in ("github", "both"):
        try:
            _open_github_issue(config_github, title, body)
            log.info("GitHub issue opened in %s", config_github.repo)
        except Exception:
            log.error("GitHub alert failed", exc_info=True)


def _format_body(digest: str, result: SmokeResult) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        f"Timestamp:    {ts}",
        f"Digest:       {digest}",
        f"Failed step:  {result.step}",
        f"Detail:       {result.detail}",
        f"Versions observed: {result.ver_observed}",
        f"Version expected:  {result.ver_expected}",
        "",
        "--- stdout ---",
        result.stdout[-2000:] or "(empty)",
        "",
        "--- stderr ---",
        result.stderr[-2000:] or "(empty)",
    ]
    return "\n".join(lines)


def _send_smtp(cfg: SmtpConfig, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.from_addr
    msg["To"] = cfg.to_addr
    msg.set_content(body)
    if cfg.port == 465:
        with smtplib.SMTP_SSL(cfg.host, cfg.port) as smtp:
            smtp.login(cfg.user, cfg.password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(cfg.host, cfg.port) as smtp:
            smtp.starttls()
            smtp.login(cfg.user, cfg.password)
            smtp.send_message(msg)


def _open_github_issue(cfg: GithubConfig, title: str, body: str) -> None:
    resp = requests.post(
        f"https://api.github.com/repos/{cfg.repo}/issues",
        headers={
            "Authorization": f"Bearer {cfg.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"title": title, "body": body, "labels": ["bench-alarm"]},
        timeout=30,
    )
    resp.raise_for_status()

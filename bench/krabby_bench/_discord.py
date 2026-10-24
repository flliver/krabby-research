"""Discord webhook notifications for harness / CI.

Posts on success and failure. Skips cleanly when DISCORD_WEBHOOK_URL is unset
so local and stubbed-CI runs do not require a secret.

CI workflows collapse consecutive PASSes (FAIL always; PASS only after a
non-green prior run) so green schedules / pushes do not spam Discord.
"""
from __future__ import annotations

import logging
import os
from typing import Mapping, Optional, Sequence

import requests

log = logging.getLogger(__name__)

_DETAIL_SOFT_LIMIT = 1500


def _github_api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


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


def default_commit_subject(*, max_len: int = 120) -> str:
    """First line of HEAD commit message (git subject), clipped for Discord."""
    try:
        import subprocess

        subject = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""
    if max_len > 0 and len(subject) > max_len:
        return subject[: max_len - 1] + "…"
    return subject


def default_branch() -> str:
    """Best-effort branch/ref from GITHUB_REF_NAME / GITHUB_HEAD_REF."""
    # pull_request: GITHUB_HEAD_REF is the source branch; GITHUB_REF_NAME is the PR merge ref.
    head = (os.environ.get("GITHUB_HEAD_REF") or "").strip()
    if head:
        return head
    return (os.environ.get("GITHUB_REF_NAME") or "").strip()


def default_event() -> str:
    return (os.environ.get("GITHUB_EVENT_NAME") or "").strip()


def default_ci_failures_doc_url() -> str:
    """Permalink to the CI failures playbook on the default branch."""
    server = (os.environ.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY") or "flliver/krabby-research"
    return f"{server}/{repo}/blob/main/docs/CI-FAILURES.md"


def format_job_results(job_results: Optional[Mapping[str, str]]) -> str:
    """Render per-job/stage results as a compact multi-line block."""
    if not job_results:
        return ""
    lines = ["Jobs:"]
    for name, result in job_results.items():
        mark = "OK" if result == "success" else result.upper()
        lines.append(f"• {name}: **{mark}**")
    return "\n".join(lines)


def format_failed_steps(failed_steps: Optional[Sequence[str]]) -> str:
    if not failed_steps:
        return ""
    return "Failed steps: " + "; ".join(f"**{s}**" for s in failed_steps)


def list_failed_steps_from_run(
    *,
    repository: Optional[str] = None,
    run_id: Optional[str] = None,
    token: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list[str]:
    """Return ``job › step`` labels for failed steps in an Actions run.

    Uses the GitHub Actions jobs API. Returns [] if env/token/API is unavailable
    so callers can still post a useful Discord message without step detail.
    """
    repo = (repository or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    rid = (run_id or os.environ.get("GITHUB_RUN_ID") or "").strip()
    tok = (token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    base = (api_url or os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")
    if not (repo and rid and tok):
        return []

    url = f"{base}/repos/{repo}/actions/runs/{rid}/jobs"
    try:
        resp = requests.get(
            url,
            headers=_github_api_headers(tok),
            params={"per_page": 100},
            timeout=30,
        )
        resp.raise_for_status()
        jobs = resp.json().get("jobs") or []
    except Exception:
        log.warning("Could not list Actions jobs for Discord detail", exc_info=True)
        return []

    out: list[str] = []
    for job in jobs:
        job_name = job.get("name") or "job"
        for step in job.get("steps") or []:
            if step.get("conclusion") in ("failure", "timed_out"):
                step_name = step.get("name") or "step"
                out.append(f"{job_name} › {step_name}")
    return out


def previous_workflow_run_conclusion(
    *,
    repository: Optional[str] = None,
    run_id: Optional[str] = None,
    token: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Optional[str]:
    """Return the conclusion of the prior completed run of this workflow.

    Looks up the current run's ``workflow_id``, then the newest completed run
    other than ``run_id``. Returns e.g. ``\"success\"`` / ``\"failure\"``, or
    ``None`` if env/token/API is unavailable or no prior run exists.
    """
    repo = (repository or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    rid = (run_id or os.environ.get("GITHUB_RUN_ID") or "").strip()
    tok = (token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    base = (api_url or os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")
    if not (repo and rid and tok):
        return None

    headers = _github_api_headers(tok)
    try:
        run_resp = requests.get(
            f"{base}/repos/{repo}/actions/runs/{rid}",
            headers=headers,
            timeout=30,
        )
        run_resp.raise_for_status()
        workflow_id = run_resp.json().get("workflow_id")
        if workflow_id is None:
            return None

        list_resp = requests.get(
            f"{base}/repos/{repo}/actions/workflows/{workflow_id}/runs",
            headers=headers,
            params={"status": "completed", "per_page": 20},
            timeout=30,
        )
        list_resp.raise_for_status()
        runs = list_resp.json().get("workflow_runs") or []
    except Exception:
        log.warning("Could not list prior Actions runs for PASS collapse", exc_info=True)
        return None

    for run in runs:
        other_id = str(run.get("id") or "")
        if other_id == rid:
            continue
        conclusion = run.get("conclusion")
        if conclusion:
            return str(conclusion)
    return None


def should_collapse_consecutive_pass(
    *,
    ok: bool,
    collapse_consecutive_pass: bool,
    previous_conclusion: Optional[str],
) -> bool:
    """True when a PASS should be skipped because the prior run was also green.

    ``previous_conclusion`` is the prior completed run's conclusion, or ``None``
    when unknown / no prior run (do not collapse — prefer a heartbeat PASS).
    """
    if not ok or not collapse_consecutive_pass:
        return False
    return previous_conclusion == "success"


def format_discord_payload(
    *,
    ok: bool,
    title: str = "krabby-bench",
    commit: str = "",
    subject: str = "",
    failed_stage: Optional[str] = None,
    run_url: str = "",
    detail: str = "",
    branch: str = "",
    event: str = "",
    job_results: Optional[Mapping[str, str]] = None,
    failed_steps: Optional[Sequence[str]] = None,
    help_url: str = "",
) -> dict:
    """Build a Discord webhook JSON body (content + embed)."""
    status = "PASS" if ok else "FAIL"
    color = 0x2ECC71 if ok else 0xE74C3C
    short_sha = commit[:7] if commit else ""
    lines = [f"**{status}** — {title}"]
    if short_sha or subject:
        if short_sha and subject:
            lines.append(f"Commit: `{short_sha}` {subject}")
        elif short_sha:
            lines.append(f"Commit: `{short_sha}`")
        else:
            lines.append(f"Commit: {subject}")
    if branch or event:
        ref_bits = []
        if branch:
            ref_bits.append(f"`{branch}`")
        if event:
            ref_bits.append(f"({event})")
        lines.append("Branch: " + " ".join(ref_bits))
    if not ok and failed_stage:
        lines.append(f"Failed stage: **{failed_stage}**")
    steps_line = format_failed_steps(failed_steps)
    if not ok and steps_line:
        lines.append(steps_line)
    jobs_block = format_job_results(job_results)
    if jobs_block:
        lines.append(jobs_block)
    if run_url:
        lines.append(f"[View run]({run_url})")
    if help_url:
        lines.append(f"[CI failures playbook]({help_url})")
    if detail:
        # Discord embed description soft limit ~4096; keep actionable, not a full log.
        clipped = detail if len(detail) <= _DETAIL_SOFT_LIMIT else detail[:_DETAIL_SOFT_LIMIT] + "…"
        lines.append(clipped)

    description = "\n".join(lines)
    content_bits = [f"{status}: {title}"]
    if failed_stage:
        content_bits.append(f"({failed_stage})")
    elif failed_steps:
        content_bits.append(f"({failed_steps[0]})")
    return {
        "content": " ".join(content_bits),
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
    branch: str = "",
    event: str = "",
    job_results: Optional[Mapping[str, str]] = None,
    failed_steps: Optional[Sequence[str]] = None,
    help_url: str = "",
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
        branch=branch or default_branch(),
        event=event or default_event(),
        job_results=job_results,
        failed_steps=failed_steps,
        help_url=help_url,
    )
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        log.info("Discord notification sent (%s)", "PASS" if ok else "FAIL")
        return True
    except Exception:
        log.error("Discord notification failed", exc_info=True)
        return False


def notify_ci_result(
    *,
    ok: bool,
    title: str,
    failed_stage: Optional[str] = None,
    detail: str = "",
    branch: str = "",
    event: str = "",
    commit: str = "",
    subject: str = "",
    job_results: Optional[Mapping[str, str]] = None,
    failed_steps: Optional[Sequence[str]] = None,
    include_failed_steps_from_api: bool = True,
    collapse_consecutive_pass: bool = False,
    help_url: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Notify Discord about a CI workflow outcome with branch/event/step detail.

    When ``include_failed_steps_from_api`` is True and ``failed_steps`` is empty
    on a failure, attempts to resolve failed step names via the Actions API.
    Commit subject defaults to ``git log -1 --format=%s`` when unset.
    Links ``docs/CI-FAILURES.md`` unless ``help_url`` is set to "" to suppress.

    When ``collapse_consecutive_pass`` is True, a PASS is skipped if the previous
    completed run of this workflow also concluded ``success`` (FAIL always posts).
    """
    if collapse_consecutive_pass and ok:
        prev = previous_workflow_run_conclusion()
        if should_collapse_consecutive_pass(
            ok=True,
            collapse_consecutive_pass=True,
            previous_conclusion=prev,
        ):
            log.info(
                "Discord skipped: consecutive PASS for %s (prior run was success)",
                title,
            )
            return False

    steps: Optional[list[str]]
    if failed_steps is not None:
        steps = list(failed_steps)
    elif not ok and include_failed_steps_from_api:
        steps = list_failed_steps_from_run()
    else:
        steps = None

    stage = failed_stage
    if not ok and not stage and steps:
        # First "job › step" → use job as stage when caller did not set one.
        stage = steps[0].split(" › ", 1)[0]

    if help_url is None:
        playbook = default_ci_failures_doc_url()
    else:
        playbook = help_url

    return post_discord(
        webhook_url,
        ok=ok,
        title=title,
        commit=commit,
        subject=subject or default_commit_subject(),
        failed_stage=stage,
        detail=detail,
        branch=branch,
        event=event,
        job_results=job_results,
        failed_steps=steps,
        help_url=playbook,
    )


def notify_harness_result(
    *,
    ok: bool,
    failed_stage: Optional[str] = None,
    detail: str = "",
    run_url: str = "",
    commit: str = "",
    subject: str = "",
    webhook_url: Optional[str] = None,
    collapse_consecutive_pass: bool = True,
) -> bool:
    """Notify Discord about a four-stage harness outcome.

    FAIL always; when ``collapse_consecutive_pass`` is True, PASS only after a
    non-green prior workflow run (same rule as ``notify_ci_result``).
    """
    if collapse_consecutive_pass and ok:
        prev = previous_workflow_run_conclusion()
        if should_collapse_consecutive_pass(
            ok=True,
            collapse_consecutive_pass=True,
            previous_conclusion=prev,
        ):
            log.info(
                "Discord skipped: consecutive PASS for krabby-bench harness "
                "(prior run was success)"
            )
            return False
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

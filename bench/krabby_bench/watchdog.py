"""Main poll loop: check ECR digest → smoke test → alert."""
from __future__ import annotations

import logging
import os
import signal
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

from krabby_bench._alert import send_alert, should_alert
from krabby_bench._config import Config, PID_PATH
from krabby_bench._ecr import get_digest
from krabby_bench._smoke import run_smoke
from krabby_bench._state import load_state, save_state

_force_recheck = threading.Event()


def _handle_sigusr1(_signum, _frame):
    log.info("Force recheck requested (SIGUSR1) — will clear digest on next poll")
    _force_recheck.set()

log = logging.getLogger(__name__)


def _krabby_bin() -> str:
    return shutil.which("krabby") or os.path.expanduser("~/.local/bin/krabby")


def _get_image_ref() -> str:
    from krabby._state import resolve_image_ref, installed_image  # type: ignore[import]
    return resolve_image_ref(installed_image())


def poll_once(config: Config, state: dict) -> dict:
    """Run one poll cycle. Returns updated state."""
    try:
        digest = get_digest(config.ecr.repo, config.ecr.tag)
    except Exception:
        log.exception("ECR digest fetch failed")
        return state

    if digest == state.get("last_tested_digest"):
        log.debug("Digest unchanged (%s), skipping", digest[:16])
        return state

    log.info("New digest %s — running update + smoke", digest[:16])

    try:
        subprocess.run(
            [_krabby_bin(), "update", "--image", f"{config.ecr.repo}:{config.ecr.tag}"],
            check=True,
        )
        log.info("krabby update succeeded")
    except Exception:
        log.exception("krabby update failed")
        return state

    try:
        image_ref = _get_image_ref()
        log.debug("Resolved image ref: %s", image_ref)
    except Exception:
        log.exception("Failed to resolve installed image ref")
        return state

    result = run_smoke(config.smoke.firmware_channel, image_ref)

    new_state = {**state, "last_tested_digest": digest}

    if not result.ok:
        log.warning("Smoke failed: %s / %s — %s", digest[:16], result.step, result.detail)
        alert_key = f"{digest}:{result.step}"
        if should_alert(state, alert_key, config.alert.dedup_window):
            try:
                send_alert(config.alert, config.smtp, config.github, digest, result)
                log.warning("Alert sent for %s / %s", digest[:16], result.step)
            except Exception:
                log.exception("Alert delivery failed")
            new_state["last_alert_at"] = datetime.now(timezone.utc).isoformat()
            new_state["last_alert_key"] = alert_key
        else:
            log.info("Alert suppressed (dedup window active)")
    else:
        log.info("Smoke passed: %s all boards %s", digest[:16], result.ver_observed)

    save_state(config.state_path, new_state)
    return new_state


def _load_credentials(config: Config) -> tuple | None:
    """Try SSM for SmtpConfig + GithubConfig. Returns None if SSM is not configured or fails."""
    if not config.ssm.prefix:
        return None
    from krabby_bench._secrets import read_device_key
    from krabby_bench._ssm import fetch_secrets
    log.info("Reading device key from disk")
    if not (key := read_device_key()):
        log.warning("Device key unavailable (secrets.enc missing or unreadable)")
        return None
    log.info("Device key loaded — fetching credentials from SSM (%s)", config.ssm.prefix)
    result = fetch_secrets(config.ssm.prefix, *key)
    if result:
        log.info("SSM credentials loaded successfully")
    return result


def run(config: Config, log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s %(message)s")
    log.info("krabby-bench watchdog starting (interval=%ds)", config.ecr.poll_interval)

    PID_PATH.write_text(str(os.getpid()))
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    log.debug("PID %d written to %s; SIGUSR1 handler installed", os.getpid(), PID_PATH)

    state = load_state(config.state_path)
    cred_warning_logged = False
    last_cred_refresh = 0.0  # forces a credential fetch on the first iteration

    while True:
        now = time.monotonic()
        if now - last_cred_refresh >= config.ssm.credentials_refresh_interval:
            if (creds := _load_credentials(config)):
                config.smtp, config.github = creds
                cred_warning_logged = False
            elif config.ssm.prefix and not cred_warning_logged:
                log.warning("SSM credentials unavailable — alert delivery disabled")
                cred_warning_logged = True
            last_cred_refresh = now

        if _force_recheck.is_set():
            log.info("Force recheck: clearing last tested digest — next poll will re-run update + smoke")
            state.pop("last_tested_digest", None)
            save_state(config.state_path, state)
            _force_recheck.clear()

        state = poll_once(config, state)
        time.sleep(config.ecr.poll_interval)

"""Main poll loop: check ECR digest → smoke test → alert."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from krabby_bench._alert import send_alert, should_alert
from krabby_bench._config import Config
from krabby_bench._ecr import get_digest
from krabby_bench._smoke import run_smoke
from krabby_bench._state import load_state, save_state

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
    except Exception:
        log.exception("krabby update failed")
        return state

    image_ref = _get_image_ref()
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
    if not (key := read_device_key()):
        return None
    return fetch_secrets(config.ssm.prefix, *key)


def run(config: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("krabby-bench watchdog starting (interval=%ds)", config.ecr.poll_interval)
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

        state = poll_once(config, state)
        time.sleep(config.ecr.poll_interval)

"""AWS SSM Parameter Store credential fetch."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def fetch_secrets(prefix: str, key_id: str, key_secret: str) -> tuple | None:
    """Fetch SmtpConfig + GithubConfig from SSM. Returns None on any error."""
    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed — SSM credential fetch unavailable")
        return None

    from krabby_bench._config import GithubConfig, SmtpConfig

    try:
        client = boto3.client(
            "ssm",
            aws_access_key_id=key_id,
            aws_secret_access_key=key_secret,
            region_name="us-east-1",
        )
        params: dict[str, str] = {}
        paginator = client.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(Path=prefix, WithDecryption=True):
            for p in page["Parameters"]:
                name = p["Name"].removeprefix(prefix).lstrip("/")
                params[name] = p["Value"]
        if not params:
            log.warning("SSM fetch returned no parameters under %r — credentials unavailable", prefix)
            return None
        return (
            SmtpConfig(
                host=params.get("smtp-host", ""),
                port=int(params.get("smtp-port", "587")),
                user=params.get("smtp-user", ""),
                password=params.get("smtp-password", ""),
                from_addr=params.get("smtp-from", ""),
                to_addr=params.get("smtp-to", ""),
            ),
            GithubConfig(
                repo=params.get("github-repo", ""),
                token=params.get("github-token", ""),
            ),
        )
    except Exception:
        log.warning("SSM fetch failed — credentials unavailable")
        return None

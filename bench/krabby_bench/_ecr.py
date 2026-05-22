"""ECR Public digest polling via the OCI registry HTTP API (no credentials required)."""
from __future__ import annotations

import json
import urllib.request


def get_digest(repo_uri: str, tag: str) -> str:
    """Return the image digest for repo_uri:tag without AWS credentials.

    Uses the OCI Distribution Spec anonymous token flow against ECR Public.
    repo_uri must be a public.ecr.aws URI, e.g. public.ecr.aws/t7t7b3i3/krabby-locomotion.
    """
    path = "/".join(repo_uri.split("/")[1:])  # e.g. t7t7b3i3/krabby-locomotion

    token_url = (
        f"https://public.ecr.aws/token"
        f"?service=public.ecr.aws&scope=repository:{path}:pull"
    )
    with urllib.request.urlopen(token_url) as resp:
        token = json.loads(resp.read())["token"]

    manifest_url = f"https://public.ecr.aws/v2/{path}/manifests/{tag}"
    req = urllib.request.Request(manifest_url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
    })
    with urllib.request.urlopen(req) as resp:
        return resp.headers["Docker-Content-Digest"]

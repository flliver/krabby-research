"""ECR Public digest polling via the OCI registry HTTP API (no credentials required)."""
from __future__ import annotations

import hashlib
import json
import urllib.request


def get_digest(repo_uri: str, tag: str) -> str:
    """Return the image digest for repo_uri:tag without AWS credentials.

    Uses the OCI Distribution Spec anonymous token flow against ECR Public.
    Computes sha256 of the manifest body since ECR Public omits the
    Docker-Content-Digest header for OCI index responses.
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
        "Accept": "application/vnd.oci.image.index.v1+json",
    })
    with urllib.request.urlopen(req) as resp:
        body = resp.read()

    return "sha256:" + hashlib.sha256(body).hexdigest()

"""ECR Public digest polling via boto3 describe_images (no pull required)."""
from __future__ import annotations

import boto3


def get_digest(repo_uri: str, tag: str) -> str:
    """Return the image digest for repo_uri:tag without pulling.

    ECR Public API is only available in us-east-1 regardless of image location.
    """
    repo_name = repo_uri.split("/")[-1]
    ecr = boto3.client("ecr-public", region_name="us-east-1")
    resp = ecr.describe_images(
        repositoryName=repo_name,
        imageIds=[{"imageTag": tag}],
    )
    return resp["imageDetails"][0]["imageDigest"]

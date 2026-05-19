"""ECR digest polling via boto3 describe_images (no pull required)."""
from __future__ import annotations

import boto3


def get_digest(repo_uri: str, tag: str, region: str) -> str:
    """Return the image digest for repo_uri:tag without pulling."""
    registry_id = repo_uri.split(".")[0]
    repo_name = repo_uri.split("/")[-1]
    ecr = boto3.client("ecr", region_name=region)
    resp = ecr.describe_images(
        registryId=registry_id,
        repositoryName=repo_name,
        imageIds=[{"imageTag": tag}],
    )
    return resp["imageDetails"][0]["imageDigest"]

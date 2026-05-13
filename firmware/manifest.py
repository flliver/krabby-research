"""
S3 firmware store manifest parsing.

Bucket layout:
  index.json                             - all branches + their latest build_key
  <branch>/latest.json                   - pointer to the latest build for one branch
  <branch>/<build_key>/manifest.json     - full build metadata
  <branch>/<build_key>/firmware.hex      - the HEX file

build_key format: YYYYMMDD-HHMMSS-<sha7>   (lexicographic sort == chronological)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class BuildManifest:
    schema_version: int
    branch: str
    commit: str
    commit_date: str        # YYYY-MM-DD
    build_timestamp: str    # ISO 8601
    board_fqbn: str
    ver_string: str         # content after "VER " — e.g. "1.0 mainline abc1234"
    hex_filename: str


@dataclass
class BranchEntry:
    branch: str
    build_key: str          # YYYYMMDD-HHMMSS-<sha7>; sorts chronologically
    hex_url: str
    manifest_url: str


@dataclass
class FirmwareIndex:
    schema_version: int
    updated: str            # ISO 8601
    branches: dict[str, BranchEntry]


# --- parsers ---

def parse_manifest(data: dict) -> BuildManifest:
    """Parse a manifest.json dict. Raises ValueError on missing required fields."""
    _require(data, "manifest", "schema_version", "branch", "commit", "commit_date",
             "build_timestamp", "board_fqbn", "ver_string", "hex_filename")
    return BuildManifest(
        schema_version=int(data["schema_version"]),
        branch=data["branch"],
        commit=data["commit"],
        commit_date=data["commit_date"],
        build_timestamp=data["build_timestamp"],
        board_fqbn=data["board_fqbn"],
        ver_string=data["ver_string"],
        hex_filename=data["hex_filename"],
    )


def parse_latest(data: dict) -> BranchEntry:
    """Parse a <branch>/latest.json dict."""
    _require(data, "latest", "branch", "build_key", "hex_url", "manifest_url")
    return BranchEntry(
        branch=data["branch"],
        build_key=data["build_key"],
        hex_url=data["hex_url"],
        manifest_url=data["manifest_url"],
    )


def parse_index(data: dict) -> FirmwareIndex:
    """Parse the root index.json dict."""
    _require(data, "index", "schema_version", "updated", "branches")
    branches = {}
    for name, entry in data["branches"].items():
        _require(entry, f"index.branches[{name!r}]", "build_key", "hex_url", "manifest_url")
        branches[name] = BranchEntry(
            branch=name,
            build_key=entry["build_key"],
            hex_url=entry["hex_url"],
            manifest_url=entry["manifest_url"],
        )
    return FirmwareIndex(
        schema_version=int(data["schema_version"]),
        updated=data["updated"],
        branches=branches,
    )


def latest_release_branch(index: FirmwareIndex) -> Optional[BranchEntry]:
    """Return the BranchEntry for the most recently built release/* branch, or None."""
    releases = [e for name, e in index.branches.items() if name.startswith("release/")]
    if not releases:
        return None
    return max(releases, key=lambda e: e.build_key)


# --- internal ---

def _require(data: dict, ctx: str, *fields: str) -> None:
    missing = [f for f in fields if f not in data]
    if missing:
        raise ValueError(f"{ctx}: missing required fields: {missing}")

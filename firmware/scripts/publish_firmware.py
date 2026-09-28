#!/usr/bin/env python3
"""Upload compiled firmware HEX and update S3 manifests.

Expects:
  - firmware/build/arduino.ino.hex  (from arduino-cli --output-dir firmware/build)
  - firmware/arduino/version.h       (from gen_version_h.py)
  - AWS credentials in the environment (via OIDC in CI)

Usage: python firmware/scripts/publish_firmware.py
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from firmware.scripts.gen_version_h import _git

BUCKET = "krabby-firmware-public"
HEX_PATH = Path("firmware/build/arduino.ino.hex")
HEX_FILENAME = "firmware.hex"
VERSION_H_PATH = Path("firmware/arduino/version.h")
BOARD_FQBN = "arduino:avr:mega"


HISTORY_CAP = 200


def _merge_build(builds_doc: dict, build_entry: dict, cap: int = HISTORY_CAP) -> dict:
    """Add build_entry to a <branch>/builds.json doc, keeping the newest per commit.

    Drops any existing entry for the same commit before appending. A daily scheduled
    rebuild of an unchanged release commit produces a fresh (timestamped) build_key
    but the same commit, so keying dedup on commit — not build_key — stops duplicate
    rows accumulating and evicting real history via the cap. Mutates and returns the
    doc. Display order doesn't matter here: `parse_builds` re-sorts by build_key.
    """
    builds_doc.setdefault("schema_version", 1)
    commit = build_entry.get("commit")
    builds = [b for b in builds_doc.get("builds", []) if b.get("commit") != commit]
    builds.append(build_entry)
    builds_doc["builds"] = builds[-cap:]
    return builds_doc


def _read_version_h(path: Path) -> tuple[str, str, str]:
    """Return (version, branch, commit) from a generated version.h."""
    text = path.read_text()
    def _extract(macro: str) -> str:
        m = re.search(rf'#define {macro} "([^"]+)"', text)
        if not m:
            raise ValueError(f"{macro} not found in {path}")
        return m.group(1)
    return (
        _extract("KRABBY_FW_VERSION"),
        _extract("KRABBY_FW_BRANCH"),
        _extract("KRABBY_FW_COMMIT"),
    )


def main() -> None:
    import boto3
    from botocore.exceptions import ClientError

    if not HEX_PATH.exists():
        sys.exit(f"HEX not found: {HEX_PATH}. Run arduino-cli compile first.")

    version, branch, commit = _read_version_h(VERSION_H_PATH)
    now = datetime.now(timezone.utc)
    build_key = now.strftime("%Y%m%d-%H%M%S") + f"-{commit}"
    s3_prefix = f"{branch}/{build_key}"
    base_url = f"https://{BUCKET}.s3.amazonaws.com/{s3_prefix}"
    hex_url = f"{base_url}/firmware.hex"
    manifest_url = f"{base_url}/manifest.json"

    commit_date = _git(["git", "log", "-1", "--format=%cs"])  # YYYY-MM-DD

    manifest = {
        "schema_version": 1,
        "branch": branch,
        "commit": commit,
        "commit_date": commit_date,
        "build_timestamp": now.isoformat(),
        "board_fqbn": BOARD_FQBN,
        "ver_string": f"{version} {branch} {commit}",
        "hex_filename": HEX_FILENAME,
    }

    s3 = boto3.client("s3", region_name="us-east-1")

    s3.upload_file(
        HEX_PATH, BUCKET, f"{s3_prefix}/{HEX_FILENAME}",
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    print(f"Uploaded  {s3_prefix}/{HEX_FILENAME}")

    s3.put_object(
        Bucket=BUCKET, Key=f"{s3_prefix}/manifest.json",
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Uploaded  {s3_prefix}/manifest.json")

    latest = {
        "branch": branch,
        "build_key": build_key,
        "hex_url": hex_url,
        "manifest_url": manifest_url,
    }
    s3.put_object(
        Bucket=BUCKET, Key=f"{branch}/latest.json",
        Body=json.dumps(latest, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Updated   {branch}/latest.json")

    # Append to the per-branch build history that powers `krabby-firmware show <branch>`.
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{branch}/builds.json")
        builds_doc = json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            builds_doc = {"schema_version": 1, "branch": branch, "builds": []}
        else:
            raise

    builds_doc["branch"] = branch
    build_entry = {
        "build_key": build_key,
        "commit": commit,
        "commit_date": commit_date,
        "ver_string": manifest["ver_string"],
        "hex_url": hex_url,
        "manifest_url": manifest_url,
    }
    _merge_build(builds_doc, build_entry)
    s3.put_object(
        Bucket=BUCKET, Key=f"{branch}/builds.json",
        Body=json.dumps(builds_doc, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Updated   {branch}/builds.json ({len(builds_doc['builds'])} builds)")

    # Patch index.json in place. Not atomic — concurrent pushes from different branches can
    # race on this key. Acceptable for current push volume; use conditional writes if needed.
    try:
        obj = s3.get_object(Bucket=BUCKET, Key="index.json")
        index = json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            index = {"schema_version": 1, "updated": "", "branches": {}}
        else:
            raise

    index["updated"] = now.isoformat()
    index["branches"][branch] = {
        "build_key": build_key,
        "hex_url": hex_url,
        "manifest_url": manifest_url,
    }
    s3.put_object(
        Bucket=BUCKET, Key="index.json",
        Body=json.dumps(index, indent=2).encode(),
        ContentType="application/json",
    )
    print("Updated   index.json")


if __name__ == "__main__":
    main()

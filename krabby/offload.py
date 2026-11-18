"""Host-side rosbag segment offloader (runs under ``krabby agent``).

Uploads closed collector segments from the bags directory to S3 using temporary
credentials from the IoT Credentials Provider (device certificate). Never reads
the active segment; pauses while a teleop edge WebSocket is connected; rate-limits
upload throughput; resumes after crash/network loss (aborts incomplete multipart
and retries the file cleanly).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

from krabby._iot import CERT_PATH, KEY_PATH, ROOT_CA_PATH, load_config as load_iot_config
from krabby._locomotion_config import DEFAULT_BAGS_HOST_DIR, load_config as load_loco_config

logger = logging.getLogger(__name__)

# Fixed alias matching ControlPlaneStack.
KRAB_BAGS_ROLE_ALIAS = "KrabBagsRoleAlias"

_STATE_NAME = ".offload_state.json"
_SEGMENT_RE = re.compile(r"^krabby_(\d+)_(\d{8})_(\d{6})$")
_DEFAULT_RATE_BPS = 1_000_000
# S3 multipart minimum (non-final) part size.
_PART_SIZE = 5 * 1024 * 1024


@dataclass
class OffloadConfig:
    bags_dir: Path
    bucket: str
    role_alias: str = KRAB_BAGS_ROLE_ALIAS
    credential_endpoint: str = ""
    thing_name: str = ""
    rate_bytes_per_s: int = _DEFAULT_RATE_BPS
    enabled: bool = False
    poll_interval_s: float = 5.0


def load_offload_config() -> OffloadConfig:
    """Assemble offload settings from IoT + locomotion config on the device."""
    iot = load_iot_config()
    loco = load_loco_config()
    bags_dir = Path(str(loco.get("bags_host_dir") or DEFAULT_BAGS_HOST_DIR)).expanduser()
    bucket = str(loco.get("bags_bucket") or iot.get("bags_bucket") or "")
    rate = int(loco.get("bags_upload_rate_bytes_per_s") or _DEFAULT_RATE_BPS)
    return OffloadConfig(
        bags_dir=bags_dir,
        bucket=bucket,
        role_alias=str(iot.get("bags_role_alias") or KRAB_BAGS_ROLE_ALIAS),
        credential_endpoint=str(iot.get("credential_endpoint") or ""),
        thing_name=str(iot.get("thing_name") or ""),
        rate_bytes_per_s=max(1, rate),
        enabled=bool(loco.get("bags_offload")),
    )


def list_closed_segments(bags_dir: Path) -> list[Path]:
    """Return completed rosbag2 segment dirs (have metadata.yaml), oldest first.

    The active writer segment has no ``metadata.yaml`` until closed, so it is excluded.
    """
    if not bags_dir.is_dir():
        return []
    out: list[Path] = []
    for child in bags_dir.iterdir():
        if child.is_dir() and (child / "metadata.yaml").is_file():
            out.append(child)
    return sorted(out, key=lambda p: (p / "metadata.yaml").stat().st_mtime)


def _segment_s3_prefix(thing_name: str, segment_dir: Path) -> str:
    """Build ``<thing>/<YYYY-MM-DD>/<HHMMSS>_<seq>/`` from segment directory name."""
    m = _SEGMENT_RE.match(segment_dir.name)
    if m:
        seq, ymd, hms = m.group(1), m.group(2), m.group(3)
        date = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        return f"{thing_name}/{date}/{hms}_{seq}/"
    ts = time.gmtime((segment_dir / "metadata.yaml").stat().st_mtime)
    date = time.strftime("%Y-%m-%d", ts)
    hms = time.strftime("%H%M%S", ts)
    return f"{thing_name}/{date}/{hms}_{segment_dir.name}/"


def _load_state(bags_dir: Path) -> dict[str, Any]:
    path = bags_dir / _STATE_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(bags_dir: Path, state: dict[str, Any]) -> None:
    path = bags_dir / _STATE_NAME
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def fetch_iot_credentials(
    *,
    credential_endpoint: str,
    role_alias: str,
    thing_name: str,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """HTTPS mTLS exchange of the device cert for temporary AWS credentials."""
    import ssl

    url = f"https://{credential_endpoint}/role-aliases/{role_alias}/credentials"
    ctx = ssl.create_default_context(cafile=str(ROOT_CA_PATH))
    ctx.load_cert_chain(certfile=str(CERT_PATH), keyfile=str(KEY_PATH))
    req = Request(url, method="GET", headers={"x-amzn-iot-thingname": thing_name})
    with urlopen(req, context=ctx, timeout=timeout_s) as resp:  # noqa: S310 — device TLS to AWS
        body = json.loads(resp.read().decode("utf-8"))
    creds = body.get("credentials") or body
    return {
        "AccessKeyId": creds["accessKeyId"],
        "SecretAccessKey": creds["secretAccessKey"],
        "SessionToken": creds["sessionToken"],
    }


def _rate_sleep(n_bytes: int, rate_bytes_per_s: int, t0: float, sent_before: int) -> None:
    """Sleep so cumulative bytes stay under the configured bytes/sec ceiling."""
    rate = max(1, int(rate_bytes_per_s))
    sent = sent_before + n_bytes
    elapsed = max(time.monotonic() - t0, 1e-6)
    allowed = rate * elapsed
    if sent > allowed:
        time.sleep((sent - allowed) / float(rate))


def _abort_multipart_quiet(s3_client: Any, bucket: str, key: str, upload_id: str) -> None:
    try:
        s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
    except Exception as exc:
        logger.warning("abort_multipart_upload %s: %s", key, exc)


def _upload_one_file(
    fpath: Path,
    *,
    key: str,
    cfg: OffloadConfig,
    s3_client: Any,
    file_state: dict[str, Any],
    persist: Callable[[], None],
    teleop_busy: Optional[Callable[[], bool]] = None,
) -> None:
    """Upload one file with multipart parts, rate limiting, and clean resume."""
    local_size = fpath.stat().st_size
    if file_state.get("done"):
        head = s3_client.head_object(Bucket=cfg.bucket, Key=key)
        if int(head["ContentLength"]) == local_size:
            return
        # Remote missing/wrong — clear and re-upload.
        file_state.clear()

    # Incomplete prior attempt: abort and restart so we never append duplicate parts.
    old_id = file_state.get("upload_id")
    if old_id:
        _abort_multipart_quiet(s3_client, cfg.bucket, key, str(old_id))
        file_state.clear()
        persist()

    resp = s3_client.create_multipart_upload(Bucket=cfg.bucket, Key=key)
    upload_id = resp["UploadId"]
    file_state.update({"upload_id": upload_id, "parts": [], "key": key})
    persist()

    parts: list[dict[str, Any]] = []
    t0 = time.monotonic()
    sent = 0
    part_number = 1
    with fpath.open("rb") as fh:
        while True:
            if teleop_busy is not None and teleop_busy():
                raise RuntimeError("teleop became active mid-upload; will retry later")
            chunk = fh.read(_PART_SIZE)
            if not chunk:
                break
            part = s3_client.upload_part(
                Bucket=cfg.bucket,
                Key=key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=chunk,
            )
            parts.append({"ETag": part["ETag"], "PartNumber": part_number})
            file_state["parts"] = parts
            persist()
            _rate_sleep(len(chunk), cfg.rate_bytes_per_s, t0, sent)
            sent += len(chunk)
            part_number += 1

    if not parts:
        # Empty file: S3 still needs a completed multipart or put_object.
        part = s3_client.upload_part(
            Bucket=cfg.bucket,
            Key=key,
            PartNumber=1,
            UploadId=upload_id,
            Body=b"",
        )
        parts = [{"ETag": part["ETag"], "PartNumber": 1}]

    s3_client.complete_multipart_upload(
        Bucket=cfg.bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    head = s3_client.head_object(Bucket=cfg.bucket, Key=key)
    remote_size = int(head["ContentLength"])
    if remote_size != local_size:
        raise RuntimeError(
            f"size mismatch after upload {key}: local={local_size} remote={remote_size}"
        )
    file_state.clear()
    file_state["done"] = True
    file_state["key"] = key
    persist()


def upload_segment(
    segment_dir: Path,
    *,
    cfg: OffloadConfig,
    s3_client: Any,
    teleop_busy: Optional[Callable[[], bool]] = None,
) -> None:
    """Multipart-upload every file under ``segment_dir``, verify, then delete local."""
    prefix = _segment_s3_prefix(cfg.thing_name, segment_dir)
    state = _load_state(cfg.bags_dir)
    uploads = state.setdefault("uploads", {})
    seg_key = str(segment_dir)
    seg_state = uploads.setdefault(seg_key, {"prefix": prefix, "files": {}})

    def persist() -> None:
        _save_state(cfg.bags_dir, state)

    files = sorted(f for f in segment_dir.rglob("*") if f.is_file())
    for fpath in files:
        rel = fpath.relative_to(segment_dir).as_posix()
        key = prefix + rel
        file_state = seg_state["files"].setdefault(rel, {})
        _upload_one_file(
            fpath,
            key=key,
            cfg=cfg,
            s3_client=s3_client,
            file_state=file_state,
            persist=persist,
            teleop_busy=teleop_busy,
        )

    shutil.rmtree(segment_dir)
    uploads.pop(seg_key, None)
    persist()
    logger.info("Uploaded and deleted segment %s -> s3://%s/%s", segment_dir, cfg.bucket, prefix)


def _boto3_s3_client(creds: dict[str, str], region: str) -> Any:
    import boto3

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


class BagsOffloadWorker:
    """Background worker: scan → upload closed segments while teleop is idle."""

    def __init__(
        self,
        cfg: OffloadConfig,
        *,
        teleop_busy: Callable[[], bool],
        region: str = "us-east-2",
        fetch_creds: Optional[Callable[..., dict[str, str]]] = None,
        s3_factory: Optional[Callable[[dict[str, str], str], Any]] = None,
    ) -> None:
        self._cfg = cfg
        self._teleop_busy = teleop_busy
        self._region = region
        self._fetch_creds = fetch_creds or (
            lambda: fetch_iot_credentials(
                credential_endpoint=cfg.credential_endpoint,
                role_alias=cfg.role_alias,
                thing_name=cfg.thing_name,
            )
        )
        self._s3_factory = s3_factory or _boto3_s3_client
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backoff_s = 2.0

    def start(self) -> None:
        if not self._cfg.enabled:
            logger.info("Bags offload disabled (locomotion.json bags_offload=false)")
            return
        if not self._cfg.bucket or not self._cfg.credential_endpoint or not self._cfg.thing_name:
            logger.warning(
                "Bags offload enabled but bucket/credential_endpoint/thing_name incomplete; idle"
            )
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="bags-offload", daemon=True)
        self._thread.start()
        logger.info(
            "Bags offload worker started (dir=%s bucket=%s rate=%s B/s)",
            self._cfg.bags_dir,
            self._cfg.bucket,
            self._cfg.rate_bytes_per_s,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._teleop_busy():
                    self._stop.wait(self._cfg.poll_interval_s)
                    continue
                segments = list_closed_segments(self._cfg.bags_dir)
                if not segments:
                    self._stop.wait(self._cfg.poll_interval_s)
                    continue
                creds = self._fetch_creds()
                s3 = self._s3_factory(creds, self._region)
                upload_segment(
                    segments[0],
                    cfg=self._cfg,
                    s3_client=s3,
                    teleop_busy=self._teleop_busy,
                )
                self._backoff_s = 2.0
            except Exception as exc:
                logger.error("Bags offload error: %s", exc, exc_info=True)
                self._stop.wait(self._backoff_s)
                self._backoff_s = min(self._backoff_s * 2.0, 120.0)

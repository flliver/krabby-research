"""Unit tests for krabby.offload (mocked S3 / credentials)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from krabby.offload import (
    BagsOffloadWorker,
    OffloadConfig,
    list_closed_segments,
    upload_segment,
    _segment_s3_prefix,
)


def _make_segment(bags: Path, name: str, *, payload: bytes = b"mcap-bytes-here") -> Path:
    seg = bags / name
    seg.mkdir(parents=True)
    (seg / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    (seg / "data.mcap").write_bytes(payload)
    return seg


def test_list_closed_segments_sorted(tmp_path: Path) -> None:
    bags = tmp_path / "bags"
    a = _make_segment(bags, "krabby_000001_20261117_120000")
    b = _make_segment(bags, "krabby_000002_20261117_120500")
    import os
    import time

    older = time.time() - 100
    newer = time.time() - 10
    os.utime(a / "metadata.yaml", (older, older))
    os.utime(b / "metadata.yaml", (newer, newer))
    assert list_closed_segments(bags) == [a, b]


def test_list_closed_segments_skips_active_without_metadata(tmp_path: Path) -> None:
    bags = tmp_path / "bags"
    closed = _make_segment(bags, "krabby_000001_20261117_120000")
    active = bags / "krabby_000002_20261117_120500"
    active.mkdir()
    (active / "data.mcap").write_bytes(b"partial")
    assert list_closed_segments(bags) == [closed]


def test_segment_s3_prefix_from_name() -> None:
    p = Path("krabby_000042_20261117_153045")
    assert _segment_s3_prefix("orin1", p) == "orin1/2026-11-17/153045_000042/"


def test_upload_segment_verify_and_delete(tmp_path: Path) -> None:
    bags = tmp_path / "bags"
    seg = _make_segment(bags, "krabby_000003_20261117_130000")
    cfg = OffloadConfig(
        bags_dir=bags,
        bucket="krab-bags-test",
        thing_name="orin1",
        rate_bytes_per_s=10_000_000,
        enabled=True,
        credential_endpoint="creds.example",
    )
    sizes = {f.name: f.stat().st_size for f in seg.rglob("*") if f.is_file()}
    s3 = MagicMock()
    s3.create_multipart_upload.return_value = {"UploadId": "uid-1"}
    s3.upload_part.return_value = {"ETag": '"etag1"'}
    s3.head_object.side_effect = lambda **kw: {
        "ContentLength": sizes[kw["Key"].rsplit("/", 1)[-1]]
    }

    upload_segment(seg, cfg=cfg, s3_client=s3)
    assert not seg.exists()
    assert s3.complete_multipart_upload.called


def test_upload_segment_aborts_incomplete_then_retries(tmp_path: Path) -> None:
    bags = tmp_path / "bags"
    seg = _make_segment(bags, "krabby_000004_20261117_140000", payload=b"abc")
    cfg = OffloadConfig(
        bags_dir=bags,
        bucket="krab-bags-test",
        thing_name="orin1",
        rate_bytes_per_s=10_000_000,
        enabled=True,
        credential_endpoint="creds.example",
    )
    # Seed incomplete multipart state for data.mcap.
    state_path = bags / ".offload_state.json"
    state_path.write_text(
        '{"uploads": {"'
        + str(seg)
        + '": {"prefix": "orin1/2026-11-17/140000_000004/", "files": {'
        '"data.mcap": {"upload_id": "stale-uid", "parts": [{"ETag": "x", "PartNumber": 1}], '
        '"key": "orin1/2026-11-17/140000_000004/data.mcap"}}}}}'
    )
    sizes = {f.name: f.stat().st_size for f in seg.rglob("*") if f.is_file()}
    s3 = MagicMock()
    s3.create_multipart_upload.return_value = {"UploadId": "uid-new"}
    s3.upload_part.return_value = {"ETag": '"etag1"'}
    s3.head_object.side_effect = lambda **kw: {
        "ContentLength": sizes[kw["Key"].rsplit("/", 1)[-1]]
    }

    upload_segment(seg, cfg=cfg, s3_client=s3)
    assert s3.abort_multipart_upload.called
    assert not seg.exists()


def test_upload_skips_already_done_file(tmp_path: Path) -> None:
    import json

    bags = tmp_path / "bags"
    seg = _make_segment(bags, "krabby_000005_20261117_150000", payload=b"xyz")
    cfg = OffloadConfig(
        bags_dir=bags,
        bucket="krab-bags-test",
        thing_name="orin1",
        rate_bytes_per_s=10_000_000,
        enabled=True,
        credential_endpoint="creds.example",
    )
    prefix = "orin1/2026-11-17/150000_000005/"
    state = {
        "uploads": {
            str(seg): {
                "prefix": prefix,
                "files": {
                    "data.mcap": {"done": True, "key": f"{prefix}data.mcap"},
                    "metadata.yaml": {"done": True, "key": f"{prefix}metadata.yaml"},
                },
            }
        }
    }
    (bags / ".offload_state.json").write_text(json.dumps(state))
    sizes = {f.name: f.stat().st_size for f in seg.rglob("*") if f.is_file()}
    s3 = MagicMock()
    s3.head_object.side_effect = lambda **kw: {
        "ContentLength": sizes[kw["Key"].rsplit("/", 1)[-1]]
    }

    upload_segment(seg, cfg=cfg, s3_client=s3)
    assert not s3.create_multipart_upload.called
    assert not seg.exists()


def test_worker_pauses_when_teleop_busy(tmp_path: Path) -> None:
    bags = tmp_path / "bags"
    _make_segment(bags, "krabby_000001_20261117_120000")
    cfg = OffloadConfig(
        bags_dir=bags,
        bucket="b",
        thing_name="t",
        credential_endpoint="e",
        enabled=True,
        poll_interval_s=0.05,
    )
    fetch = MagicMock(side_effect=AssertionError("should not fetch while busy"))
    worker = BagsOffloadWorker(
        cfg,
        teleop_busy=lambda: True,
        fetch_creds=fetch,
        s3_factory=lambda *_: MagicMock(),
    )
    worker.start()
    import time

    time.sleep(0.2)
    worker.stop()
    fetch.assert_not_called()


def test_worker_disabled_noop(tmp_path: Path) -> None:
    cfg = OffloadConfig(
        bags_dir=tmp_path,
        bucket="b",
        thing_name="t",
        credential_endpoint="e",
        enabled=False,
    )
    worker = BagsOffloadWorker(cfg, teleop_busy=lambda: False)
    worker.start()
    assert worker._thread is None

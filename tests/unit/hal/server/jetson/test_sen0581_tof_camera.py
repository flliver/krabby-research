"""SEN0581 protocol parser and depth conversion (recorded fixtures, no hardware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hal.server.jetson.sen0581_tof_camera import (
    SEN0581_FRAME_BYTES,
    SEN0581_HEIGHT,
    SEN0581_WIDTH,
    Sen0581FrameParser,
    parse_sen0581_frame,
    sen0581_depth_u8_to_meters,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sen0581"


@pytest.fixture(scope="module")
def frame0() -> bytes:
    return (_FIXTURES / "frame_0.bin").read_bytes()


def test_fixture_frame_shape_dtype_meters(frame0: bytes) -> None:
    assert len(frame0) == SEN0581_FRAME_BYTES
    depth = parse_sen0581_frame(frame0)
    assert depth.shape == (SEN0581_HEIGHT, SEN0581_WIDTH)
    assert depth.dtype == np.float32
    assert float(np.nanmin(depth)) >= 0.0
    # UNIT=1: non-invalid samples are v/1000 meters and stay under the 1.5 m band.
    valid = depth[depth > 0.0]
    assert valid.size > 0
    assert float(valid.max()) <= 0.255  # uint8 mm / 1000, excluding 0xFF→0


def test_invalid_u8_maps_to_zero() -> None:
    raw = np.array([[10, 255], [0, 100]], dtype=np.uint8)
    meters = sen0581_depth_u8_to_meters(raw)
    assert meters.dtype == np.float32
    assert meters[0, 0] == pytest.approx(0.01)
    assert meters[0, 1] == 0.0
    assert meters[1, 0] == 0.0
    assert meters[1, 1] == pytest.approx(0.1)


def test_parser_three_frames_blob() -> None:
    blob = (_FIXTURES / "three_frames.bin").read_bytes()
    parser = Sen0581FrameParser()
    frames = parser.feed(blob)
    assert len(frames) == 3
    for fr in frames:
        assert fr.shape == (SEN0581_HEIGHT, SEN0581_WIDTH)
        assert fr.dtype == np.float32


def test_parser_recovers_from_truncated_then_complete(frame0: bytes) -> None:
    trunc = (_FIXTURES / "truncated_frame.bin").read_bytes()
    assert len(trunc) < SEN0581_FRAME_BYTES
    parser = Sen0581FrameParser()
    assert parser.feed(trunc) == []
    # Complete the truncated head, then a full following frame.
    need = SEN0581_FRAME_BYTES - len(trunc)
    frames = parser.feed(frame0[-need:] + frame0)
    assert len(frames) >= 1
    assert frames[0].shape == (SEN0581_HEIGHT, SEN0581_WIDTH)


def test_parser_resyncs_mid_stream() -> None:
    mid = (_FIXTURES / "mid_stream.bin").read_bytes()
    parser = Sen0581FrameParser()
    frames = parser.feed(mid)
    assert len(frames) >= 2
    for fr in frames:
        assert fr.shape == (SEN0581_HEIGHT, SEN0581_WIDTH)


def test_parse_rejects_bad_header() -> None:
    with pytest.raises(ValueError, match="header"):
        parse_sen0581_frame(b"\x00\x00" + bytes(SEN0581_FRAME_BYTES - 2))

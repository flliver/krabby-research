"""SDP helpers for teleop WebRTC (no aiortc required)."""

from __future__ import annotations

from teleop.edge.sdp_util import (
    count_video_m_lines,
    offer_has_h264_video,
    video_m_line_budget_error_json,
)


def test_count_video_m_lines_zero_falls_back_to_one() -> None:
    assert count_video_m_lines("v=0\nm=audio 9\n") == 1


def test_count_video_m_lines_single() -> None:
    sdp = "v=0\nm=audio 9\nm=video 9\n"
    assert count_video_m_lines(sdp) == 1


def test_count_video_m_lines_multiple() -> None:
    sdp = "v=0\nm=video 1\nm=video 2\nm=video 3\n"
    assert count_video_m_lines(sdp) == 3


def test_video_m_line_budget_none_unlimited() -> None:
    sdp = "v=0\n" + "\n".join(["m=video 9"] * 10) + "\n"
    assert video_m_line_budget_error_json(sdp, None) is None


def test_video_m_line_budget_rejects() -> None:
    sdp = "v=0\nm=video 9\nm=video 9\nm=video 9\n"
    err = video_m_line_budget_error_json(sdp, 2)
    assert err is not None
    assert "too many" in err
    assert "3" in err
    assert "robot_settings" in err


def test_offer_has_h264_video_true() -> None:
    sdp = (
        "v=0\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\n"
        "a=rtpmap:96 H264/90000\n"
    )
    assert offer_has_h264_video(sdp) is True


def test_offer_has_h264_video_false() -> None:
    sdp = (
        "v=0\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\n"
        "a=rtpmap:96 VP8/90000\n"
    )
    assert offer_has_h264_video(sdp) is False

"""Decode H.265 camera topics from a collector bag back to RGB or metric depth.

Usage examples::

    python -m data_collection.tools.decode_h265 /path/to/bag \\
        --catalog-id front_rgbd --channel rgb --out-dir /tmp/frames

    python -m data_collection.tools.decode_h265 /path/to/bag \\
        --catalog-id belly_left_tof --channel depth --check-roundtrip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def _decode_hevc_annex_b_frames(payloads: list[bytes]) -> list[np.ndarray]:
    """Decode a sequence of HEVC annex-B access units to RGB or gray uint8 frames.

    Prefers OpenCV's video decoder via a temporary elementary stream when available;
    falls back to GStreamer ``appsrc ! h265parse ! avdec_h265 ! … ! appsink``.
    """
    if not payloads:
        return []

    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    blob = b"".join(payloads)
    if cv2 is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".h265", delete=True) as tmp:
            tmp.write(blob)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            frames: list[np.ndarray] = []
            if cap.isOpened():
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # OpenCV yields BGR; convert to RGB for RGB streams. Gray stays 2D.
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else:
                        frames.append(frame)
            cap.release()
            if frames:
                return frames

    return _decode_hevc_via_gstreamer(blob)


def _decode_hevc_via_gstreamer(blob: bytes) -> list[np.ndarray]:
    from hal.server.gstreamer_runtime import ensure_gst_initialized

    ensure_gst_initialized()
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp

    desc = (
        "appsrc name=src is-live=false format=time ! "
        "h265parse ! avdec_h265 ! videoconvert ! "
        "video/x-raw,format=RGB ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=0 drop=false"
    )
    pipeline = Gst.parse_launch(desc)
    appsrc = pipeline.get_by_name("src")
    appsink = pipeline.get_by_name("sink")
    assert isinstance(appsrc, GstApp.AppSrc)
    assert isinstance(appsink, GstApp.AppSink)
    pipeline.set_state(Gst.State.PLAYING)
    buf = Gst.Buffer.new_wrapped(blob)
    appsrc.push_buffer(buf)
    if hasattr(appsrc, "end_stream"):
        appsrc.end_stream()
    else:
        appsrc.emit("end-of-stream")

    frames: list[np.ndarray] = []
    while True:
        sample = appsink.try_pull_sample(5 * Gst.SECOND)
        if sample is None:
            break
        out_buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w = int(struct.get_value("width"))
        h = int(struct.get_value("height"))
        ok, mapinfo = out_buf.map(Gst.MapFlags.READ)
        if not ok:
            continue
        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            out_buf.unmap(mapinfo)
        frames.append(arr)
    pipeline.set_state(Gst.State.NULL)
    return frames


def iter_compressed_camera_messages(
    bag_dir: Path,
    *,
    catalog_id: str,
    channel: str,
) -> Iterator[tuple[int, str, bytes]]:
    """Yield ``(stamp_ns, format, payload)`` for ``/camera/{id}/{channel}_h265``."""
    from rosbags.highlevel import AnyReader
    from data_collection.serialization import (
        COMPRESSED_IMAGE_MSGTYPE,
        catalog_camera_topic,
        recording_stream_name,
    )

    topic = catalog_camera_topic(catalog_id, recording_stream_name(channel, "h265"))
    with AnyReader([bag_dir]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise FileNotFoundError(f"topic {topic} not found in {bag_dir}")
        for conn, stamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, COMPRESSED_IMAGE_MSGTYPE)
            yield int(stamp_ns), str(msg.format), bytes(msg.data)


def decode_bag_camera(
    bag_dir: Path,
    *,
    catalog_id: str,
    channel: str,
) -> tuple[list[np.ndarray], Optional[tuple[float, float]]]:
    """Decode one camera stream. Depth returns float32 meters; RGB returns uint8 RGB."""
    from data_collection.depth_quant import gray8_to_float32_depth
    from data_collection.serialization import parse_hevc_format

    payloads: list[bytes] = []
    fmt: str | None = None
    for _stamp, format_str, data in iter_compressed_camera_messages(
        bag_dir, catalog_id=catalog_id, channel=channel
    ):
        payloads.append(data)
        fmt = format_str
    if not payloads:
        return [], None
    assert fmt is not None
    codec, rng = parse_hevc_format(fmt)
    if codec != "hevc":
        raise ValueError(f"unsupported CompressedImage format {fmt!r}")

    frames_u8 = _decode_hevc_annex_b_frames(payloads)
    if channel == "depth":
        if rng is None:
            raise ValueError(f"depth stream missing range= in format {fmt!r}")
        lo, hi = rng
        out: list[np.ndarray] = []
        for fr in frames_u8:
            if fr.ndim == 3:
                gray = fr[..., 0]
            else:
                gray = fr
            out.append(gray8_to_float32_depth(gray, lo, hi))
        return out, rng
    return frames_u8, None


def check_depth_roundtrip(
    depth_m: np.ndarray,
    *,
    d_min: float,
    d_max: float,
) -> float:
    """Encode→decode without HEVC; return max abs error among finite pixels."""
    from data_collection.depth_quant import (
        depth_quant_step_m,
        float32_depth_to_gray8,
        gray8_to_float32_depth,
    )

    gray = float32_depth_to_gray8(depth_m, d_min, d_max)
    back = gray8_to_float32_depth(gray, d_min, d_max)
    valid = np.isfinite(depth_m) & (depth_m >= d_min) & (depth_m <= d_max)
    if not np.any(valid):
        return 0.0
    err = np.max(np.abs(back[valid] - depth_m[valid].astype(np.float32)))
    step = depth_quant_step_m(d_min, d_max)
    if err > step + 1e-6:
        raise AssertionError(
            f"depth round-trip max abs error {err} exceeds quantization step {step}"
        )
    return float(err)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag_dir", type=Path, help="Rosbag2 segment directory")
    p.add_argument("--catalog-id", required=True)
    p.add_argument("--channel", choices=("rgb", "depth"), required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--check-roundtrip",
        action="store_true",
        help="For depth: assert 8-bit quant round-trip stays within one step (no HEVC)",
    )
    args = p.parse_args(argv)

    if args.check_roundtrip and args.channel == "depth":
        from data_collection.depth_quant import depth_quant_range_for_catalog_id

        lo, hi = depth_quant_range_for_catalog_id(args.catalog_id)
        # Synthetic ramp across the band.
        h, w = 16, 16
        ramp = np.linspace(lo, hi, h * w, dtype=np.float32).reshape(h, w)
        err = check_depth_roundtrip(ramp, d_min=lo, d_max=hi)
        print(f"ok depth quant round-trip max_abs_err={err:.6g} m range=({lo},{hi})")
        return 0

    frames, rng = decode_bag_camera(
        args.bag_dir, catalog_id=args.catalog_id, channel=args.channel
    )
    print(f"decoded {len(frames)} frame(s) range={rng}")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            path = args.out_dir / f"{args.catalog_id}_{args.channel}_{i:05d}.npy"
            np.save(path, fr)
        print(f"wrote {len(frames)} .npy files under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

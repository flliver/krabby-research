#!/usr/bin/env python3
"""Read a rosbag2 (mcap) directory produced by data_collection and optionally display RGB images.

Uses **rosbags** (same stack as the writer). Standard ROS 2 ``ros2 bag play`` also works on these bags.

Example:
  python scripts/playback_krabby_bag.py /data/bags/krabby_000000_20250408_120000 --topic /camera/front_rgbd/rgb --max 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    try:
        import numpy as np
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import get_typestore
        from rosbags.typesys.stores import Stores
    except ImportError:
        print(
            "rosbags is required (see images/locomotion/requirements.txt or images/isaacsim/requirements.txt)",
            file=sys.stderr,
        )
        return 1

    p = argparse.ArgumentParser(description="Playback / inspect Krabby mcap bags")
    p.add_argument("bag_dir", type=Path, help="Path to rosbag2 directory (contains metadata.yaml)")
    p.add_argument(
        "--topic",
        type=str,
        default="/camera/front_rgbd/rgb",
        help="Image topic to decode (default: /camera/front_rgbd/rgb)",
    )
    p.add_argument("--max", type=int, default=10, help="Max messages to show from that topic")
    p.add_argument(
        "--display",
        action="store_true",
        help="Show images with OpenCV (requires opencv-python headless or GUI)",
    )
    args = p.parse_args()
    bag_dir = args.bag_dir
    if not (bag_dir / "metadata.yaml").is_file():
        print(f"Not a rosbag2 directory (missing metadata.yaml): {bag_dir}", file=sys.stderr)
        return 1

    ts = get_typestore(Stores.LATEST)
    shown = 0
    with AnyReader([bag_dir]) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            print(f"No connection for topic {args.topic!r}. Available:", file=sys.stderr)
            for c in reader.connections:
                print(f"  {c.topic} ({c.msgtype})", file=sys.stderr)
            return 1
        if args.display:
            try:
                import cv2
            except ImportError:
                print("--display requires opencv-python", file=sys.stderr)
                return 1
        for conn, timestamp, raw in reader.messages(connections=conns):
            if conn.msgtype != "sensor_msgs/msg/Image":
                print(f"Topic {args.topic} is not sensor_msgs/Image ({conn.msgtype})", file=sys.stderr)
                return 1
            msg = ts.deserialize_cdr(raw, "sensor_msgs/msg/Image")
            h, w = int(msg.height), int(msg.width)
            enc = msg.encoding
            arr = np.asarray(msg.data, dtype=np.uint8)
            print(f"msg t={timestamp} ns size={w}x{h} enc={enc!r} bytes={arr.size}")
            if args.display and enc in ("rgb8", "bgr8"):
                img = arr.reshape((h, w, -1))
                if enc == "rgb8":
                    img = img[:, :, ::-1]
                cv2.imshow("krabby_bag", img)
                cv2.waitKey(0)
            shown += 1
            if shown >= args.max:
                break
    if args.display:
        try:
            import cv2

            cv2.destroyAllWindows()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

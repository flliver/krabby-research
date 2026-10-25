"""Isaac HAL depth clip + teleop preview bands (meters).

All values match ``JETSON_SENSOR_CATALOG`` ``gst_depth_quant_range_m`` on the corresponding rgbd
row (front ZED 0.2–25 m; flank MaixSense 0.15–1.5 m).

Pinhole ``clipping_range`` on each ``*_rgb`` / ``*_camera`` pair uses the same ``(near, far)`` as that
row's depth band so RGB and depth see the same distance limits.
"""

from __future__ import annotations

# (near, far) for pinhole depth spawn clipping_range on front_camera / side_*_camera.
ISAAC_FRONT_DEPTH_CLIP_M: tuple[float, float] = (0.2, 25.0)
ISAAC_SIDE_DEPTH_CLIP_M: tuple[float, float] = (0.15, 1.5)

# Teleop GRAY16 / ``depth_meters_to_rgb24_u8`` preview (SensorInfo.extra depth_range_m).
ISAAC_FRONT_DEPTH_PREVIEW_M: tuple[float, float] = (0.2, 25.0)
ISAAC_SIDE_DEPTH_PREVIEW_M: tuple[float, float] = (0.15, 1.5)

ISAAC_FRONT_RGB_CLIP_M: tuple[float, float] = ISAAC_FRONT_DEPTH_CLIP_M
ISAAC_SIDE_RGB_CLIP_M: tuple[float, float] = ISAAC_SIDE_DEPTH_CLIP_M

# HAL observation sanitization: finite depth above clip becomes this value (meters).
ISAAC_FRONT_DEPTH_POSINF_CLIP_M: float = ISAAC_FRONT_DEPTH_CLIP_M[1]
ISAAC_SIDE_DEPTH_POSINF_CLIP_M: float = ISAAC_SIDE_DEPTH_CLIP_M[1]

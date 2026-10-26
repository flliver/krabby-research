"""Isaac HAL depth clip + teleop preview bands (meters).

Depth preview/clip matches ``JETSON_SENSOR_CATALOG`` ``gst_depth_quant_range_m`` (front ZED
0.2–25 m; flank MaixSense 0.15–1.5 m). Front RGB uses that same front band.

Side RGB far clip is 4 m so flank color is not cut off at the MaixSense depth band. Side depth
stays 0.15–1.5 m.
"""

from __future__ import annotations

# (near, far) for pinhole depth spawn clipping_range on front_camera / side_*_camera.
ISAAC_FRONT_DEPTH_CLIP_M: tuple[float, float] = (0.2, 25.0)
ISAAC_SIDE_DEPTH_CLIP_M: tuple[float, float] = (0.15, 1.5)

# Teleop GRAY16 / ``depth_meters_to_rgb24_u8`` preview (SensorInfo.extra depth_range_m).
ISAAC_FRONT_DEPTH_PREVIEW_M: tuple[float, float] = (0.2, 25.0)
ISAAC_SIDE_DEPTH_PREVIEW_M: tuple[float, float] = (0.15, 1.5)

ISAAC_FRONT_RGB_CLIP_M: tuple[float, float] = ISAAC_FRONT_DEPTH_CLIP_M
ISAAC_SIDE_RGB_CLIP_M: tuple[float, float] = (0.15, 4.0)

# HAL observation sanitization: finite depth above clip becomes this value (meters).
ISAAC_FRONT_DEPTH_POSINF_CLIP_M: float = ISAAC_FRONT_DEPTH_CLIP_M[1]
ISAAC_SIDE_DEPTH_POSINF_CLIP_M: float = ISAAC_SIDE_DEPTH_CLIP_M[1]

"""HAL server implementation for Jetson."""
# JetsonHalServer is resolved lazily (PEP 562): importing it pulls in the full
# server stack (zmq, compute.parkour -> torch/rsl_rl), which isn't installed on
# the bare Jetson host where light submodules like zed_camera are used
# standalone (e.g. scripts/zed_imu_probe.py).

__all__ = ["JetsonHalServer"]


def __getattr__(name):
    if name == "JetsonHalServer":
        from hal.server.jetson.hal_server import JetsonHalServer

        return JetsonHalServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

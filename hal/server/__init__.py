"""HAL server package."""
# Re-export main classes for cleaner imports. HalServerBase is resolved
# lazily (PEP 562) so light submodules (e.g. hal.server.jetson.zed_camera on
# the bare Jetson host) can be imported without pulling in zmq.
from hal.server.config import HalServerConfig

__all__ = ["HalServerBase", "HalServerConfig"]


def __getattr__(name):
    if name == "HalServerBase":
        from hal.server.server import HalServerBase

        return HalServerBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


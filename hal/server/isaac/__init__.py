"""IsaacSim HAL backend."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hal_server import IsaacSimHalServer
    from .isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
    from .isaacsim_mcusdk import IsaacSimMCUSDK

__all__ = ["IsaacSimHalServer", "IsaacSimMCUSDK", "IsaacSimFirmwareMCUSDK"]


def __getattr__(name: str):
    if name == "IsaacSimHalServer":
        from .hal_server import IsaacSimHalServer

        return IsaacSimHalServer
    if name == "IsaacSimMCUSDK":
        from .isaacsim_mcusdk import IsaacSimMCUSDK

        return IsaacSimMCUSDK
    if name == "IsaacSimFirmwareMCUSDK":
        from .isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK

        return IsaacSimFirmwareMCUSDK
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


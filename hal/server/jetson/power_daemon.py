"""Handle leader-MCU shutdown messages on the Orin."""
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from firmware.interfaces.power_messages import PowerMessage, PowerMessageType

logger = logging.getLogger("KrabbyPowerDaemon")

HOST_POWEROFF_READY_PATH = Path("/run/krabby/poweroff.ready")
HOST_POWEROFF_REQUEST_PATH = Path("/run/krabby/poweroff.request")


def build_host_poweroff_requester(
    ready_path: Path = HOST_POWEROFF_READY_PATH,
    request_path: Path = HOST_POWEROFF_REQUEST_PATH,
) -> Optional[Callable[[], None]]:
    if not ready_path.is_file():
        return None

    def request_poweroff() -> None:
        try:
            fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as request:
            request.write("poweroff\n")

    return request_poweroff


class OrinPowerDaemon:
    """Queues host poweroff before acknowledging the MCU."""

    def __init__(
        self,
        sdk,
        request_poweroff: Optional[Callable[[], None]] = None,
    ):
        self._sdk = sdk
        self._request_poweroff = request_poweroff
        self._handled = False

    def start(self) -> None:
        self._sdk.on_power_message = self.handle
        if self._request_poweroff is None:
            logger.warning("Orin power daemon has no host-poweroff action; ACK disabled")
        else:
            logger.info("Orin power daemon armed; listening for POWERING_DOWN")

    def stop(self) -> None:
        cb = getattr(self._sdk, "on_power_message", None)
        if getattr(cb, "__self__", None) is self:
            self._sdk.on_power_message = None

    def handle(self, msg: PowerMessage) -> None:
        """Acknowledge and begin host poweroff once per shutdown cycle."""
        if msg.type is PowerMessageType.RESUMING:
            self._handled = False
            return
        if msg.type is PowerMessageType.EMERGENCY_SHUTDOWN:
            logger.critical(
                "Emergency shutdown received (%s); no ACK is defined",
                msg.reason.value,
            )
            return
        if msg.type is not PowerMessageType.POWERING_DOWN:
            return
        if self._handled:
            return
        reason = msg.reason.value if msg.reason is not None else "-"
        if self._request_poweroff is None:
            logger.critical(
                "Power-down received (%s), but host poweroff is not configured; "
                "not acknowledging",
                reason,
            )
            return
        try:
            self._request_poweroff()
        except OSError:
            logger.exception(
                "Power-down received (%s), but the host request failed; "
                "not acknowledging",
                reason,
            )
            return
        self._handled = True
        logger.warning(
            "Power-down received (%s / %s); host poweroff requested; acking",
            msg.type.value,
            reason,
        )
        try:
            self._sdk.send_power_message(PowerMessage(PowerMessageType.SHUTDOWN_ACK))
        except OSError:
            logger.exception("Host poweroff requested, but SHUTDOWN_ACK failed")

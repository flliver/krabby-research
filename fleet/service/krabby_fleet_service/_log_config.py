"""Journal-friendly logging for krabby-fleet-service (survives uvicorn startup).

Production defaults keep journal volume low (lifecycle + errors at INFO).
Per-frame teleop/MQTT tracing is DEBUG — enable temporarily with
``KRABBY_FLEET_SIGNALING_TRACE=1`` (or ``KRABBY_FLEET_LOG_LEVEL=DEBUG``).
"""
from __future__ import annotations

import os
from typing import Any

_SERVICE_LEVEL = os.environ.get("KRABBY_FLEET_LOG_LEVEL", "INFO").upper()
_TRACE = os.environ.get("KRABBY_FLEET_SIGNALING_TRACE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MQTT_LEVEL = "DEBUG" if _TRACE else _SERVICE_LEVEL
_SIGNALING_LEVEL = "DEBUG" if _TRACE else _SERVICE_LEVEL

# Passed to uvicorn.run(log_config=...) so app loggers are not drowned by defaults.
LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"level": "INFO", "handlers": ["default"]},
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.error": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.access": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "krabby_fleet_service": {"level": _SERVICE_LEVEL, "handlers": ["default"], "propagate": False},
        "krabby_fleet_service._mqtt": {
            "level": _MQTT_LEVEL,
            "handlers": ["default"],
            "propagate": False,
        },
        "krabby_fleet_service._signaling": {
            "level": _SIGNALING_LEVEL,
            "handlers": ["default"],
            "propagate": False,
        },
    },
}

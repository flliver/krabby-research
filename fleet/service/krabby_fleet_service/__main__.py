"""krabby-fleet-service entry point: runs the app under uvicorn on 127.0.0.1:8080."""
from __future__ import annotations


def main() -> None:
    import uvicorn

    from krabby_fleet_service._log_config import LOGGING_CONFIG

    uvicorn.run(
        "krabby_fleet_service.app:app",
        host="127.0.0.1",
        port=8080,
        log_config=LOGGING_CONFIG,
    )


if __name__ == "__main__":
    main()

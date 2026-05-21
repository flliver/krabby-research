"""Browser-driven HAL RGB-D catalog ids (signaling ``catalog_ids`` on hello / offer)."""

from __future__ import annotations

from typing import Any


def parse_viewer_catalog_ids_from_payload(payload: dict[str, Any], *, max_lines: int) -> list[str] | None:
    """Parse ``catalog_ids`` from a signaling JSON object.

    Returns:
        ``None`` — field absent or invalid; keep the robot's current list (bootstrap until first set).
        ``[]`` — explicit empty array in JSON: revert to bootstrap ids (typically primary camera only).
        Non-empty list — replace active ids (trimmed strings, **deduped in order**, capped to ``max_lines``).
    """
    if "catalog_ids" not in payload:
        return None
    raw = payload.get("catalog_ids")
    if raw is None:
        return None
    if raw == []:
        return []
    if not isinstance(raw, list):
        return None
    cap = max(1, min(32, int(max_lines)))
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out if out else []

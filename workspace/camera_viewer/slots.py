"""Named filter+selection slots — load/save/list/delete to a JSON file.

The slot file lives next to the original cameras.json with `.slots` inserted
before the `.json` extension. So `cameras.json` → `cameras.slots.json`.

A slot captures:
- Every filter's state (via FilterStack.to_state())
- The full selection state (picked indices + lock toggle)
- A name + saved-at timestamp

Slot names are case-sensitive. Saving with a name that already exists
overwrites the existing slot (no confirm). Use distinct names if you want
distinct slots.

The slot file's schema is versioned in case we evolve. v1 is the only
version right now.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from filters import FilterStack, SelectionState


SCHEMA_VERSION = 1


def slots_path_for(cameras_path: Path) -> Path:
    """Compute the slot-file path that sits parallel to cameras.json.

    `data/.../cameras.json` → `data/.../cameras.slots.json`
    """
    # pathlib's with_stem keeps the parent + suffix; we need to insert ".slots"
    # before the existing suffix.
    return cameras_path.with_name(cameras_path.stem + ".slots" + cameras_path.suffix)


class SlotsManager:
    """Owns the slot file and provides save / load / list / delete."""

    def __init__(self, cameras_path: Path):
        self.path = slots_path_for(cameras_path)
        self.cameras_path = cameras_path
        self.slots: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.slots = []
            return
        try:
            doc = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Slot file at {self.path} is not valid JSON: {e}") from e
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported slot-file schema version: "
                f"got {doc.get('schema_version')}, expected {SCHEMA_VERSION}"
            )
        self.slots = doc.get("slots", [])

    def _flush(self) -> None:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "cameras_path": str(self.cameras_path),
            "slots": self.slots,
        }
        self.path.write_text(json.dumps(doc, indent=2) + "\n")

    def names(self) -> list[str]:
        """Slot names in saved-at order (oldest first)."""
        return [s["name"] for s in self.slots]

    def save(
        self,
        name: str,
        filters: FilterStack,
        selection: SelectionState,
    ) -> None:
        """Save the current filter+selection state under `name`. Overwrites
        any existing slot with the same name.
        """
        slot = {
            "name": name,
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "filter_state": filters.to_state(),
            "selection_state": selection.to_state(),
        }
        # Replace existing if same name; else append
        for i, existing in enumerate(self.slots):
            if existing["name"] == name:
                self.slots[i] = slot
                self._flush()
                return
        self.slots.append(slot)
        self._flush()

    def load(
        self,
        name: str,
        filters: FilterStack,
        selection: SelectionState,
    ) -> dict:
        """Apply the named slot to filters + selection in-place.

        Returns the slot dict (so callers can read saved_at, etc.).
        """
        for slot in self.slots:
            if slot["name"] == name:
                filters.from_state(slot.get("filter_state", {}))
                selection.from_state(slot.get("selection_state", {}))
                return slot
        raise KeyError(f"no slot named {name!r}")

    def delete(self, name: str) -> bool:
        """Delete the named slot. Returns True if a slot was deleted."""
        for i, s in enumerate(self.slots):
            if s["name"] == name:
                del self.slots[i]
                self._flush()
                return True
        return False

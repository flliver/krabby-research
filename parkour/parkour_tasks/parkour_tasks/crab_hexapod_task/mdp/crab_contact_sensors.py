"""Per-foot contact helpers for crab_hex (feet are ``*_Footpad`` rigid bodies)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensor

# Legacy scene keys (unused when using aggregate ``contact_forces`` sensor).
CRAB_HEX_FOOTPAD_CONTACT_KEYS: tuple[str, ...] = (
    "contact_feet_fl",
    "contact_feet_fr",
    "contact_feet_ml",
    "contact_feet_mr",
    "contact_feet_rl",
    "contact_feet_rr",
)

# Foot rigid-body names on ``ParkourHexContactSensor`` / articulation (distal contact pads).
CRAB_HEX_FOOTPAD_BODY_NAMES: tuple[str, ...] = (
    "FL_Footpad",
    "FR_Footpad",
    "ML_Footpad",
    "MR_Footpad",
    "RL_Footpad",
    "RR_Footpad",
)

# Backward-compatible aliases (deprecated).
CRAB_HEX_TIBIA_CONTACT_KEYS = CRAB_HEX_FOOTPAD_CONTACT_KEYS
CRAB_HEX_TIBIA_BODY_NAMES = CRAB_HEX_FOOTPAD_BODY_NAMES


def crab_hex_footpad_contacts_ready(scene: InteractiveScene) -> bool:
    s = scene.sensors
    return all(k in s for k in CRAB_HEX_FOOTPAD_CONTACT_KEYS)


def iter_crab_hex_footpad_contact_sensors(scene: InteractiveScene) -> list[ContactSensor]:
    return [scene.sensors[k] for k in CRAB_HEX_FOOTPAD_CONTACT_KEYS]


def crab_hex_tibia_contacts_ready(scene: InteractiveScene) -> bool:
    return crab_hex_footpad_contacts_ready(scene)


def iter_crab_hex_tibia_contact_sensors(scene: InteractiveScene) -> list[ContactSensor]:
    return iter_crab_hex_footpad_contact_sensors(scene)

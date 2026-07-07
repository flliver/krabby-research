"""Static per-joint facts for all 18 joints — the single home for what kind of
joint a name denotes.

Joint names follow the grammar ``<leg><kind>`` (e.g. ``FLHY`` = front-left
hip-yaw): legs FL/FR/ML/MR/RL/RR, kinds HY (hip-yaw), HL (hip-lift), KL (knee).
Everything here is derived from that grammar plus one per-kind table, so the
registry cannot drift per-joint.

Deliberately data-only: live state (cal_state, position) comes from telemetry,
and behavior lives in the firmware sketch. The facts here mirror declarations
in ``arduino/arduino.ino`` (actuator construction and ACT_LIST_* board order);
``tests/unit/firmware/test_joint_registry.py`` parses that source and fails if
the two disagree.

Deliberately dependency-free (stdlib only), like ``observation_mapping.py``:
the Jetson HAL server, the bench CLI, and the GUI all import it.
"""
from dataclasses import dataclass

LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")

KINDS = ("HY", "HL", "KL")

# kind -> (sensor, cal_directions, jog_pwm_max, end_stop_calibratable)
# HY: the encoder counts the fast motor shaft (before the 1:100 gearbox), and
#     above ~150 PWM the edge rate saturates the MCU's pin-change interrupts
#     (firmware coasts the joint with ERR hall_storm). The current mounting has
#     no end-stops, so an end-stop cal sweep would drive until the leg jams —
#     HY joints must never be swept (see calibrate-all's yaw opt-in).
_KIND_FACTS = {
    "HY": ("HALL", ("left", "right"), 150, False),
    "HL": ("HALL", ("extend", "retract"), 200, True),
    "KL": ("POT", ("extend", "retract"), 200, True),
}

# board role -> the legs it drives, in slot order (mirrors ACT_LIST_* in arduino.ino)
BOARD_LEGS = {"FRONT": ("FL", "FR"), "LEFT": ("RL", "ML"), "RIGHT": ("RR", "MR")}


@dataclass(frozen=True, slots=True)
class JointSpec:
    name: str                        # "FLHY"
    leg: str                         # "FL"
    kind: str                        # "HY" | "HL" | "KL"
    sensor: str                      # "HALL" | "POT" — matches the CAL wire vocab
    cal_directions: tuple            # valid K-command direction tokens
    jog_pwm_max: int                 # storm-safe jog ceiling
    end_stop_calibratable: bool      # False = never include in an end-stop cal sweep
    board: str                       # "FRONT" | "LEFT" | "RIGHT"
    slot: int                        # 0-5 actuator index on that board

    @property
    def is_yaw(self) -> bool:
        return self.kind == "HY"

    @property
    def position_absolute_at_boot(self) -> bool:
        """Pots are absolute by physics; Hall counts are boot-relative and need
        an end-stop anchor (M17 Task 2 §6.5 self-heal) before pos is absolute."""
        return self.sensor == "POT"


def _build() -> dict:
    out = {}
    for board, legs in BOARD_LEGS.items():
        for li, leg in enumerate(legs):
            for ki, kind in enumerate(KINDS):
                sensor, dirs, pwm, calable = _KIND_FACTS[kind]
                name = leg + kind
                out[name] = JointSpec(name, leg, kind, sensor, dirs, pwm,
                                      calable, board, li * 3 + ki)
    return out


JOINTS: dict = _build()


def spec(name: str) -> JointSpec:
    """The JointSpec for a joint name; raises ValueError for unknown names."""
    if (js := JOINTS.get(name)) is None:
        raise ValueError(f"unknown joint {name!r}; expected <leg><kind> with "
                         f"leg in {'/'.join(LEGS)} and kind in {'/'.join(KINDS)}")
    return js


def board_joints(board: str) -> tuple:
    """A board's 6 JointSpecs in slot order (matches its ACT_LIST_* and telemetry)."""
    return tuple(sorted((js for js in JOINTS.values() if js.board == board),
                        key=lambda js: js.slot))

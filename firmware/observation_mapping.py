"""Pure-Python HAL observation mapping shared by the HAL server and the
``krabby-firmware observe`` bench command.

This module is deliberately dependency-free (no numpy, no torch) so the
lightweight firmware CLI can import it without pulling the HAL/inference stack.
The HAL server wraps the list outputs into ``np.float32`` arrays.

Covers the M17 Task 6 transforms on top of raw MCU telemetry:
  - per-leg current → ``contact_forces`` (Option A, see below),
  - Python-side joint-velocity EMA (the firmware emits position, not velocity).
"""

from __future__ import annotations

# --- Current-sense → contact_forces (M17 Task 6 §3, Option A) ---------------
# The parkour model's contact_forces vector is 5-wide (trained against a
# quadruped's foot set) but the hex has 6 legs. Option A: map five legs into the
# five slots and DROP one middle leg — the two middle legs (ML/MR) are
# geometrically redundant for a forward gait, so we keep ML and drop MR.
# Per-leg load proxy is the sum of that leg's joint currents (raw ADC counts
# from JointTelemetry.current); a leg in stance draws more current.
#
# FIRST-PASS SCALING — placeholder, expected to be refined in M15. The model
# expects contact_forces in [-0.5, 0.5] (see HardwareObservations docstring).
# Summed raw current maps through (sum / CONTACT_FULLSCALE - 0.5), clipped:
# 0 current → -0.5 (no contact), CONTACT_FULLSCALE → +0.5 (firm contact).
# CONTACT_FULLSCALE MUST be retuned against Task 4's loaded-vs-unloaded avgIS
# ranges once the current-sense IS-line fault is resolved on the bench; the
# value below is a structural placeholder, not a calibrated constant.
CONTACT_LEGS: tuple[str, ...] = ("FL", "FR", "ML", "RL", "RR")
CONTACT_DROPPED_LEG = "MR"
CONTACT_FULLSCALE = 300.0  # summed raw current mapping to slot = +0.5


def leg_prefix(joint_name: str) -> str:
    """Leg prefix for a joint name in either convention.

    Firmware names are ``<leg><joint>`` (e.g. ``FLKL`` → ``FL``); HAL names are
    ``<leg>_<joint>`` (e.g. ``FL_knee`` → ``FL``).
    """
    return joint_name.split("_", 1)[0] if "_" in joint_name else joint_name[:2]


def contact_forces_from_leg_currents(leg_currents: dict[str, float]) -> list[float]:
    """Map per-leg summed current to the 5-slot contact_forces vector.

    ``leg_currents`` is keyed by leg prefix (``FL``…``RR``). Legs absent from the
    dict (no telemetry yet) get 0.0 (unknown) rather than -0.5 so a missing
    reading isn't asserted as "definitely no contact". See module header for the
    first-pass scaling rationale.
    """
    forces = [0.0] * len(CONTACT_LEGS)
    for i, leg in enumerate(CONTACT_LEGS):
        c = leg_currents.get(leg)
        if c is None:
            continue
        scaled = c / CONTACT_FULLSCALE - 0.5
        forces[i] = -0.5 if scaled < -0.5 else 0.5 if scaled > 0.5 else scaled
    return forces


# --- Joint velocity (M17 Task 6, 6c) ----------------------------------------
JOINT_VEL_EMA_ALPHA = 0.2  # single-pole EMA; ~0.2 suppresses serial-jitter spikes


class JointVelocityEstimator:
    """Per-joint velocity from successive positions, single-pole EMA-smoothed.

    The MCU firmware reports position but no velocity, so velocity is the
    differentiated position. ``update`` returns the smoothed velocity in
    position-units per second; the first sample for a joint returns 0.0.
    """

    def __init__(self, alpha: float = JOINT_VEL_EMA_ALPHA):
        self.alpha = alpha
        self._last: dict[str, tuple[float, float]] = {}  # name -> (pos, t_seconds)
        self._ema: dict[str, float] = {}

    def update(self, name: str, pos: float, t: float) -> float:
        prev = self._last.get(name)
        self._last[name] = (pos, t)
        if prev is None:
            return self._ema.get(name, 0.0)
        dt = t - prev[1]
        if dt <= 1e-6:
            return self._ema.get(name, 0.0)
        raw = (pos - prev[0]) / dt
        ema = self.alpha * raw + (1.0 - self.alpha) * self._ema.get(name, 0.0)
        self._ema[name] = ema
        return ema

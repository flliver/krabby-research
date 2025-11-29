"""Telemetry model types for HAL input data.

These models represent sensor data and robot state flowing from hardware/simulation
to the policy wrapper. Designed for zero-copy operations matching training format exactly.

Training observation format: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Training observation dimensions (from parkour_rl_cfg.py)
NUM_PROP = 53  # Proprioceptive features
NUM_SCAN = 132  # Scan/depth features
NUM_PRIV_EXPLICIT = 9  # Privileged explicit features
NUM_PRIV_LATENT = 29  # Privileged latent features
NUM_HIST = 10  # History length
HISTORY_DIM = NUM_HIST * NUM_PROP  # 530

# Total observation dimension
OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + HISTORY_DIM  # 753


@dataclass
class NavigationCommand:
    """Navigation command for robot movement.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        schema_version: Schema version string (e.g., "1.0")
        vx: Forward velocity (m/s)
        vy: Lateral velocity (m/s)
        yaw_rate: Angular velocity (rad/s)
    """

    timestamp_ns: int
    schema_version: str = "1.0"
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate navigation command."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")

    @classmethod
    def create_now(cls, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> "NavigationCommand":
        """Create navigation command with current timestamp."""
        return cls(
            timestamp_ns=time.time_ns(),
            schema_version="1.0",
            vx=vx,
            vy=vy,
            yaw_rate=yaw_rate,
        )


@dataclass
class ParkourObservation:
    """Parkour observation in exact training format with zero-copy support.

    This matches the training format exactly:
    [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        schema_version: Schema version string (e.g., "1.0")
        observation: Flat float32 array of shape (OBS_DIM,) matching training format
                    This array should be a view when possible to avoid copies.
    """

    timestamp_ns: int
    schema_version: str = "1.0"
    observation: Optional[np.ndarray] = None  # Shape: (OBS_DIM,), dtype: float32

    def __post_init__(self) -> None:
        """Validate observation."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")
        if self.observation is not None:
            if not isinstance(self.observation, np.ndarray):
                raise ValueError("observation must be a numpy array")
            if self.observation.dtype != np.float32:
                # Only convert if necessary - prefer views
                if self.observation.dtype == np.float64:
                    self.observation = self.observation.astype(np.float32, copy=False)
                else:
                    self.observation = self.observation.astype(np.float32)
            if self.observation.shape != (OBS_DIM,):
                raise ValueError(
                    f"Observation shape {self.observation.shape} != expected ({OBS_DIM},)"
                )

    def get_proprioceptive(self) -> np.ndarray:
        """Get proprioceptive features (num_prop=53) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation[:NUM_PROP]

    def get_scan(self) -> np.ndarray:
        """Get scan/depth features (num_scan=132) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation[NUM_PROP : NUM_PROP + NUM_SCAN]

    def get_priv_explicit(self) -> np.ndarray:
        """Get privileged explicit features (num_priv_explicit=9) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        start = NUM_PROP + NUM_SCAN
        return self.observation[start : start + NUM_PRIV_EXPLICIT]

    def get_priv_latent(self) -> np.ndarray:
        """Get privileged latent features (num_priv_latent=29) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        start = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT
        return self.observation[start : start + NUM_PRIV_LATENT]

    def get_history(self) -> np.ndarray:
        """Get history features (history_dim=530) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation[-HISTORY_DIM:]

    @classmethod
    def from_parts(
        cls,
        proprioceptive: np.ndarray,
        scan: np.ndarray,
        priv_explicit: np.ndarray,
        priv_latent: np.ndarray,
        history: np.ndarray,
        timestamp_ns: Optional[int] = None,
    ) -> "ParkourObservation":
        """Create observation from parts (may copy if arrays are not contiguous).

        Args:
            proprioceptive: Shape (NUM_PROP,), dtype float32
            scan: Shape (NUM_SCAN,), dtype float32
            priv_explicit: Shape (NUM_PRIV_EXPLICIT,), dtype float32
            priv_latent: Shape (NUM_PRIV_LATENT,), dtype float32
            history: Shape (HISTORY_DIM,), dtype float32
            timestamp_ns: Optional timestamp (defaults to current time)

        Returns:
            ParkourObservation with concatenated observation array
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        # Validate shapes
        if proprioceptive.shape != (NUM_PROP,):
            raise ValueError(f"proprioceptive shape {proprioceptive.shape} != ({NUM_PROP},)")
        if scan.shape != (NUM_SCAN,):
            raise ValueError(f"scan shape {scan.shape} != ({NUM_SCAN},)")
        if priv_explicit.shape != (NUM_PRIV_EXPLICIT,):
            raise ValueError(f"priv_explicit shape {priv_explicit.shape} != ({NUM_PRIV_EXPLICIT},)")
        if priv_latent.shape != (NUM_PRIV_LATENT,):
            raise ValueError(f"priv_latent shape {priv_latent.shape} != ({NUM_PRIV_LATENT},)")
        if history.shape != (HISTORY_DIM,):
            raise ValueError(f"history shape {history.shape} != ({HISTORY_DIM},)")

        # Concatenate (this creates a new array, but it's necessary for construction)
        # In the hot path, we should use pre-allocated buffers
        observation = np.concatenate(
            [
                np.asarray(proprioceptive, dtype=np.float32),
                np.asarray(scan, dtype=np.float32),
                np.asarray(priv_explicit, dtype=np.float32),
                np.asarray(priv_latent, dtype=np.float32),
                np.asarray(history, dtype=np.float32),
            ],
            dtype=np.float32,
        )

        return cls(timestamp_ns=timestamp_ns, observation=observation)


@dataclass
class ParkourModelIO:
    """Combined input model aggregating all telemetry for policy inference.

    Uses zero-copy observation format matching training exactly.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        schema_version: Schema version string (e.g., "1.0")
        nav_cmd: Navigation command (for reference, not in observation)
        observation: Complete observation in training format
    """

    timestamp_ns: int
    schema_version: str = "1.0"
    nav_cmd: Optional[NavigationCommand] = None
    observation: Optional[ParkourObservation] = None

    def __post_init__(self) -> None:
        """Validate ParkourModelIO."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")

    def is_complete(self) -> bool:
        """Check if all required components are present."""
        return self.nav_cmd is not None and self.observation is not None

    def is_synchronized(self, max_age_ns: int = 10_000_000) -> bool:
        """Check if all components have timestamps within max_age_ns of each other.

        Args:
            max_age_ns: Maximum age difference in nanoseconds (default 10ms)

        Returns:
            True if all components are synchronized, False otherwise
        """
        if not self.is_complete():
            return False

        timestamps = [
            self.nav_cmd.timestamp_ns,
            self.observation.timestamp_ns,
        ]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        return (max_ts - min_ts) <= max_age_ns

    def get_observation_array(self) -> np.ndarray:
        """Get observation array as a view (zero-copy).

        Returns:
            Observation array of shape (OBS_DIM,) matching training format
        """
        if self.observation is None or self.observation.observation is None:
            raise ValueError("Observation not set")
        return self.observation.observation

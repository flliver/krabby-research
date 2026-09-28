from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

import math
from dataclasses import MISSING
from .uniform_parkour_command import UniformParkourCommand

@configclass
class ParkourCommandCfg(CommandTermCfg):
    class_type: type = UniformParkourCommand
    asset_name: str = MISSING
    heading_control_stiffness: float = 1.0
    small_commands_to_zero: bool = True 

    stand_frac: float | None = None
    """PLAN H B1 (2026-09-03): when set, a resampled slot stands with probability
    ``stand_frac`` (Bernoulli) and walking slots draw vx from
    ``[max(lin_vel_clip, lo) + 0.01, hi]`` -- replacing the implicit
    ``small_commands_to_zero`` rule (|vx| <= clip -> 0), which zeroes ~57% of slots on the
    0.0:0.35 band. None (default) keeps today's rule bit-identically.
    See ``command_sampling.sample_slot``."""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = MISSING

    @configclass 
    class Clips:
        lin_vel_clip: float = MISSING 
        ang_vel_clip: float = MISSING 

    ranges: Ranges = MISSING
    clips: Clips = MISSING

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

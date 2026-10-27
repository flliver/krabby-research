
from __future__ import annotations

import copy
import functools
import numpy as np
import trimesh
from collections.abc import Callable
from scipy.ndimage import binary_dilation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..terrains import ParkourSubTerrainBaseCfg

def _parkour_field_to_mesh_impl(func: Callable, edge_axes: str) -> Callable:
    """Shared body behind `parkour_field_to_mesh` (`edge_axes="x"`, today's behaviour,
    bit-identical) and `parkour_field_to_mesh_xy` (`edge_axes="xy"`, both-axes riser mask
    for terrains with non-axis-aligned obstacles, e.g. the omni ring). Only the edge-mask
    computation branches on `edge_axes`; everything else is the original wrapper body
    verbatim.

    Omni phase-2 plan, step 2 (2026-10-06): a raw body MAY return a 4th value -- the
    pre-roughness stamped block heights (int16, the body's own pixel frame, zero off-block; only
    the omni field does). Then, and only then, the wrapper (a) returns a 7th value, the exact
    bool block occupancy (``stamped != 0``) in the same bordered frame as ``heights``, and (b)
    when ``cfg.vertical_edges`` is set, hands the stamped field to
    ``convert_height_field_to_mesh(riser_field=...)`` so every block edge becomes a vertical
    riser. A 3-value body takes the original path untouched (6 return values, same call).
    """
    if edge_axes not in ("x", "xy"):
        raise ValueError(f"edge_axes must be 'x' or 'xy', got {edge_axes!r}")

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: ParkourSubTerrainBaseCfg, num_goals:int):
        if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
            raise ValueError(
                f"The border width ({cfg.border_width}) must be greater than or equal to the"
                f" horizontal scale ({cfg.horizontal_scale})."
            )
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1

        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        # override size of the terrain to account for the border
        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * cfg.horizontal_scale for dim in sub_terrain_size]
        # update the config
        terrain_size = copy.deepcopy(cfg.size)

        cfg.size = tuple(sub_terrain_size)
        # generate the height field
        out = func(difficulty, cfg, num_goals)
        if len(out) == 4:
            z_gen, goals, goal_heights, stamped = out
        else:
            z_gen, goals, goal_heights = out
            stamped = None
        goals -= np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen
        block_mask = None
        riser_kw = {}
        if stamped is not None:
            stamped_full = np.zeros_like(heights)
            stamped_full[border_pixels:-border_pixels, border_pixels:-border_pixels] = stamped
            block_mask = stamped_full != 0
            if getattr(cfg, "vertical_edges", False):
                riser_kw = {"riser_field": stamped_full}
        # set terrain size back to config
        # convert to trimesh
        half_edge_width = int(cfg.edge_width_thresh / cfg.horizontal_scale)
        if edge_axes == "x":
            vertices, triangles, x_edge_mask = convert_height_field_to_mesh(
                heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold, **riser_kw
            )
            structure = np.ones((half_edge_width*2+1, 1))
            edge_mask = binary_dilation(x_edge_mask, structure=structure)
        else:  # "xy"
            vertices, triangles, x_edge_mask, y_edge_mask = convert_height_field_to_mesh(
                heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold, return_y_edges=True,
                **riser_kw,
            )
            structure = np.ones((half_edge_width*2+1, half_edge_width*2+1))
            edge_mask = binary_dilation(x_edge_mask | y_edge_mask, structure=structure)
        cfg.size = terrain_size
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        if cfg.use_simplified:
            mesh = mesh.simplify_quadric_decimation(face_count = int(0.65*triangles.shape[0]) , aggression=3)
        # compute origin
        x1 = int((cfg.size[0] * 0.5 - 1) / cfg.horizontal_scale)
        x2 = int((cfg.size[0] * 0.5 + 1) / cfg.horizontal_scale)
        y1 = int((cfg.size[1] * 0.5 - 1) / cfg.horizontal_scale)
        y2 = int((cfg.size[1] * 0.5 + 1) / cfg.horizontal_scale)
        origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
        origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])
        # PLAN H (2026-09-03): the int16 height field (border included) is returned so the
        # generator can keep it for spawn-spread z lookup / clearance checks.
        if block_mask is not None:
            # Omni phase-2 step 2: the exact block occupancy, same frame as ``heights``.
            return [mesh], origin, goals, goal_heights, edge_mask, heights, block_mask
        return [mesh], origin, goals, goal_heights, edge_mask, heights

    return wrapper


parkour_field_to_mesh = functools.partial(_parkour_field_to_mesh_impl, edge_axes="x")
"""Bit-identical to the pre-refactor decorator: dilates the x-normal riser mask only
(a `(2h+1) x 1` structure), matching every existing forward call site."""

parkour_field_to_mesh_xy = functools.partial(_parkour_field_to_mesh_impl, edge_axes="xy")
"""Both-axes twin for terrains with non-axis-aligned obstacles (e.g. the omni ring):
unions the x- and y-normal riser masks, then dilates over the same `(2h+1) x (2h+1)`
neighbourhood (`h` = the same `half_edge_width` the x-only decorator uses)."""


def convert_height_field_to_mesh(
    height_field: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None = None,
    *,
    return_y_edges: bool = False,
    riser_field: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Height field -> (vertices, triangles, x-riser mask[, y-riser mask]).

    ``riser_field`` (omni phase-2 plan step 2, 2026-10-06; default None = the original code path,
    byte for byte): an int array the shape of ``height_field`` holding ONLY the stamped block
    heights (zero elsewhere, no roughness). Wherever it steps up or down between two neighbouring
    pixels, the vertex on the low side is moved under the high side's -- a vertical riser --
    whatever the height difference, in addition to the usual ``slope_threshold`` trigger on
    ``height_field`` itself. Roughness is not in ``riser_field``, so it can never raise a wall;
    blocks under the 0.12 m threshold, which otherwise render as one-pixel ramps, get walls. The
    vertex/triangle COUNT is unchanged (vertices move, none are added).
    """
    num_rows, num_cols = height_field.shape
    # create a mesh grid of the height field
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    # copy height field to avoid modifying the original array
    hf = height_field.copy()
    # correct vertical surfaces above the slope threshold
    if riser_field is not None:
        # omni field walls (see the docstring); the threshold branch below is the original code.
        move_x, move_y, move_corners = _riser_moves(hf, riser_field, horizontal_scale, vertical_scale, slope_threshold)
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
    elif slope_threshold is not None:
        # scale slope threshold based on the horizontal and vertical scale
        slope_threshold *= horizontal_scale / vertical_scale
        # allocate arrays to store the movement of the vertices
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        # move vertices along the x-axis
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        # move vertices along the y-axis
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        # move vertices along the corners
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create vertices for the mesh
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    # create triangles for the mesh
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3
    if return_y_edges:
        return vertices, triangles, move_x != 0, move_y != 0
    return vertices, triangles, move_x!=0


def _riser_moves(
    hf: np.ndarray,
    riser_field: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(move_x, move_y, move_corners)`` for :func:`convert_height_field_to_mesh`'s
    ``riser_field`` path: the original threshold rule on ``hf`` OR any step in ``riser_field``.

    Same arithmetic as the original branch (``move[:-1] += up``, ``move[1:] -= down`` per axis and
    along the diagonal), with each trigger widened from ``hf`` jump ``> threshold`` to
    ``(hf jump > threshold) | (riser jump > 0)``. A ``slope_threshold`` of None contributes no
    ``hf`` trigger (risers only).
    """
    if riser_field.shape != hf.shape:
        raise ValueError(f"riser_field shape {riser_field.shape} != height field shape {hf.shape}")
    num_rows, num_cols = hf.shape
    rf = np.asarray(riser_field).astype(np.int32)
    h = hf.astype(np.int32)
    thr = None if slope_threshold is None else slope_threshold * horizontal_scale / vertical_scale

    def _step(hi_h, lo_h, hi_r, lo_r):
        # "hi_* side minus lo_* side": True where the first operand is the higher neighbour.
        trig = (hi_r - lo_r) > 0
        if thr is not None:
            trig = trig | ((hi_h - lo_h) > thr)
        return trig

    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_corners = np.zeros((num_rows, num_cols))
    move_x[: num_rows - 1, :] += _step(h[1:, :], h[:-1, :], rf[1:, :], rf[:-1, :])
    move_x[1:num_rows, :] -= _step(h[:-1, :], h[1:, :], rf[:-1, :], rf[1:, :])
    move_y[:, : num_cols - 1] += _step(h[:, 1:], h[:, :-1], rf[:, 1:], rf[:, :-1])
    move_y[:, 1:num_cols] -= _step(h[:, :-1], h[:, 1:], rf[:, :-1], rf[:, 1:])
    move_corners[: num_rows - 1, : num_cols - 1] += _step(h[1:, 1:], h[:-1, :-1], rf[1:, 1:], rf[:-1, :-1])
    move_corners[1:num_rows, 1:num_cols] -= _step(h[:-1, :-1], h[1:, 1:], rf[:-1, :-1], rf[1:, 1:])
    return move_x, move_y, move_corners

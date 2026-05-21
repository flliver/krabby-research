# Copyright (c) 2025, Krabby / crab_hexapod_task contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Hexapod contact sensor: aggregate chassis + all leg link cubes for the crab USD.

Isaac Lab's :class:`~isaaclab.sensors.contact_sensor.contact_sensor.ContactSensor` resolves
``prim_path`` by taking ``_parent_prims[0]`` and listing **direct** children matching the leaf
pattern. For ``{ENV_REGEX_NS}/Robot/krabby/.*/.*`` the first ``.*`` under ``krabby`` can be
``chassis``, so only ``chassis`` (not ``chassis/body``) is considered — breaking
``body_names=".*_Footpad"`` on crab foot MDP terms. This subclass gathers rigid bodies with
``PhysxContactReportAPI`` under the spawned ``Robot`` prim and builds PhysX path globs from the
**full relative path** from ``Robot`` (e.g. ``chassis/body``, ``Leg_FL/FL_Footpad``), not only the
final prim name, so nested layouts match ``create_rigid_body_view`` (same idea as stock
:class:`~isaaclab.sensors.contact_sensor.contact_sensor.ContactSensor`, but multi-level).
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaacsim.core.simulation_manager import SimulationManager
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import convert_quat
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils import configclass
from pxr import PhysxSchema

_CRAB_LEGS = ("Leg_FL", "Leg_FR", "Leg_ML", "Leg_MR", "Leg_RL", "Leg_RR")


def _collect_hex_contact_suffixes(robot_prim_path: str) -> list[str]:
    """Return unique path suffixes from ``robot_prim_path`` to each contact-reporting rigid body.

    Tries both flat layout (``chassis/body``, ``Leg_*/*``) and authored layout under ``krabby/``.
    Suffixes are used to build PhysX globs; leaf names (e.g. ``FL_Footpad``) still come from the
    rigid-body view's prim paths for MDP regex resolution.
    """
    suffixes: list[str] = []
    seen: set[str] = set()

    def _add_prim(prim) -> None:
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            return
        full = prim.GetPath().pathString
        prefix = robot_prim_path + "/"
        if not full.startswith(prefix):
            return
        suf = full[len(prefix) :]
        if suf and suf not in seen:
            seen.add(suf)
            suffixes.append(suf)

    chassis_patterns = (
        f"{robot_prim_path}/krabby/chassis/body",
        f"{robot_prim_path}/chassis/body",
    )
    for pattern in chassis_patterns:
        for prim in sim_utils.find_matching_prims(pattern):
            _add_prim(prim)

    for leg in _CRAB_LEGS:
        for pattern in (
            f"{robot_prim_path}/krabby/{leg}/.*",
            f"{robot_prim_path}/{leg}/.*",
        ):
            for prim in sim_utils.find_matching_prims(pattern):
                _add_prim(prim)

    def _suffix_sort_key(s: str) -> tuple[int, str]:
        # Prefer chassis base link(s), then stable leg order
        if s.endswith("/body") or s.rsplit("/", 1)[-1] == "body":
            return (0, s)
        return (1, s)

    return sorted(suffixes, key=_suffix_sort_key)


class ParkourHexContactSensor(ContactSensor):
    """Contact sensor that unions chassis + all ``Leg_*`` rigid links for the Krabby USD."""

    def _initialize_impl(self):
        # ``InteractiveScene`` already resolved ``prim_path`` (e.g. ``/World/envs/env_.*/Robot/.*``).
        SensorBase._initialize_impl(self)

        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        robot_prim_path = self._parent_prims[0].GetPath().pathString
        path_suffixes = _collect_hex_contact_suffixes(robot_prim_path)

        if not path_suffixes:
            raise RuntimeError(
                "ParkourHexContactSensor could not find any rigid bodies with PhysxContactReportAPI"
                f" under {robot_prim_path} (chassis + legs). Enable activate_contact_sensors on the robot spawn."
            )

        # One glob per rigid body: PhysX does not treat ``(a/b|c/d)`` as alternation over path segments.
        robot_path_expr = self.cfg.prim_path.rsplit("/", 1)[0]
        body_patterns = [f"{robot_path_expr}/{s}".replace(".*", "*") for s in path_suffixes]
        filter_prim_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]

        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_patterns)
        # Multi-pattern contact view: repeat the same filter list per sensor pattern (PhysX API contract).
        if filter_prim_paths_glob:
            filter_arg = [list(filter_prim_paths_glob) for _ in body_patterns]
        else:
            filter_arg = []

        self._contact_physx_view = self._physics_sim_view.create_rigid_contact_view(
            body_patterns,
            filter_patterns=filter_arg,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(path_suffixes) * self._num_envs,
        )

        self._num_bodies = self.body_physx_view.count // self._num_envs
        if self._num_bodies != len(path_suffixes):
            raise RuntimeError(
                "Failed to initialize contact reporter for ParkourHexContactSensor."
                f"\n\tExpected bodies (suffixes from Robot): {path_suffixes}"
                f"\n\tResolved count per env: {self._num_bodies}"
            )

        self._data.net_forces_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history = torch.zeros(
                self._num_envs, self.cfg.history_length, self._num_bodies, 3, device=self._device
            )
        else:
            self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)

        if self.cfg.track_pose:
            self._data.pos_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
            self._data.quat_w = torch.zeros(self._num_envs, self._num_bodies, 4, device=self._device)

        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if len(self.cfg.filter_prim_paths_expr) == 0:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid filter pattern to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is {self.cfg.max_contact_data_count_per_prim}. "
                    "Please set it to a value greater than 0."
                )

        if self.cfg.track_contact_points:
            self._data.contact_pos_w = torch.full(
                (self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3),
                torch.nan,
                device=self._device,
            )
        if self.cfg.track_friction_forces:
            self._data.friction_forces_w = torch.full(
                (self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3),
                0.0,
                device=self._device,
            )
        if self.cfg.track_air_time:
            self._data.last_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.last_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)

        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_filters = self.contact_physx_view.filter_count
            self._data.force_matrix_w = torch.zeros(
                self._num_envs, self._num_bodies, num_filters, 3, device=self._device
            )
            if self.cfg.history_length > 0:
                self._data.force_matrix_w_history = torch.zeros(
                    self._num_envs, self.cfg.history_length, self._num_bodies, num_filters, 3, device=self._device
                )
            else:
                self._data.force_matrix_w_history = self._data.force_matrix_w.unsqueeze(1)

    @property
    def body_names(self) -> list[str]:
        """Like :meth:`ContactSensor.body_names`, but correct for multi-pattern PhysX views.

        ``create_rigid_body_view([pattern...])`` lays out ``prim_paths`` pattern-major
        (all envs for body 0, then all for body 1, …). Names must describe one env for MDP regex
        resolution.
        """
        prim_paths = self.body_physx_view.prim_paths
        return [prim_paths[i * self._num_envs].split("/")[-1] for i in range(self._num_bodies)]

    def _pattern_major_flat_to_env_major(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder flattened PhysX outputs from pattern-major to env-major (see ``body_names``)."""
        n_env = self._num_envs
        n_body = self._num_bodies
        if x.shape[0] != n_env * n_body:
            return x
        rest = list(x.shape[1:])
        return x.view(n_body, n_env, *rest).transpose(0, 1).reshape(n_env * n_body, *rest)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        net_forces_w = self.contact_physx_view.get_net_contact_forces(dt=self._sim_physics_dt)
        net_forces_w = self._pattern_major_flat_to_env_major(net_forces_w)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]

        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(1, dims=1)
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_filters = self.contact_physx_view.filter_count
            force_matrix_w = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            force_matrix_w = self._pattern_major_flat_to_env_major(force_matrix_w)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]
            if self.cfg.history_length > 0:
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(1, dims=1)
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]

        if self.cfg.track_pose:
            pose = self.body_physx_view.get_transforms()
            pose = self._pattern_major_flat_to_env_major(pose)
            pose = pose.view(-1, self._num_bodies, 7)[env_ids]
            pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
            self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)

        if self.cfg.track_contact_points:
            _, buffer_contact_points, _, _, buffer_count, buffer_start_indices = (
                self.contact_physx_view.get_contact_data(dt=self._sim_physics_dt)
            )
            self._data.contact_pos_w[env_ids] = self._unpack_contact_buffer_data(
                buffer_contact_points, buffer_count, buffer_start_indices
            )[env_ids]

        if self.cfg.track_friction_forces:
            friction_forces, _, buffer_count, buffer_start_indices = self.contact_physx_view.get_friction_data(
                dt=self._sim_physics_dt
            )
            self._data.friction_forces_w[env_ids] = self._unpack_contact_buffer_data(
                friction_forces, buffer_count, buffer_start_indices, avg=False, default=0.0
            )[env_ids]

        if self.cfg.track_air_time:
            elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.cfg.force_threshold
            is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
            is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
            self._data.last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_air_time[env_ids],
            )
            self._data.current_air_time[env_ids] = torch.where(
                ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )
            self._data.last_contact_time[env_ids] = torch.where(
                is_first_detached,
                self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_contact_time[env_ids],
            )
            self._data.current_contact_time[env_ids] = torch.where(
                is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

    def _debug_vis_callback(self, event):
        if self.body_physx_view is None:
            return
        net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
        marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        if self.cfg.track_pose:
            frame_origins: torch.Tensor = self._data.pos_w
        else:
            pose = self.body_physx_view.get_transforms()
            pose = self._pattern_major_flat_to_env_major(pose)
            frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
        self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))


@configclass
class ParkourHexContactSensorCfg(ContactSensorCfg):
    """Same fields as :class:`ContactSensorCfg`; uses :class:`ParkourHexContactSensor`."""

    class_type: type = ParkourHexContactSensor

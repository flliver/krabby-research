# SPDX-License-Identifier: BSD-3-Clause
"""Omni-task MDP helpers.

Most of the MDP is the forward task's (``crab_hex_forward_task/mdp``, reused unchanged);
this package holds the pieces that are genuinely omnidirectional -- starting with the
command term (a WORLD travel direction + speed and a WORLD facing target, in four
command-regime slots) and its pure-torch math.

``parkour_tasks.__init__`` blacklists the ``".mdp"`` substring, so this package is NOT
imported by ``import_packages``; its modules are reached by explicit dotted imports from
the omni env cfgs (``parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command``) and, for
the symmetry augmentation, through the rsl-rl dotted string in
``config/crab_hex/agents/rsl_rl_ppo_cfg.py``.

Nothing is re-exported here on purpose: the modules below import ``isaaclab`` (the command
term subclasses ``UniformParkourCommand``), and importing them from this ``__init__`` would
make every path-load of a *pure* module in this package -- the command math, the mirror --
drag in Isaac.

Modules (documentary only -- none of these names is bound here):

``crab_hex_omni_command`` / ``crab_hex_omni_command_math`` (D1)
    The four-slot command term (world travel direction + speed, world facing) and its pure
    math (ellipse cap, world->body conversion, the samplers).
``omni_parkour_event`` (D9/B5, needs Isaac)
    ``OmniParkourEvent`` / ``OmniParkourEventsCfg`` -- the ``base_parkour`` term: tile-centre
    spawn reference, goal index that never advances, radial/sector exposure, and terrain
    promotion from the episode's TRACKING RATIO instead of the corridor's downrange distance.
``omni_exposure`` (pure)
    The sector/disc/block-mask/promotion arithmetic and the ledger wrapper that keeps the
    forward exposure key names (the campaign driver greps them) with ring meanings.
``omni_events`` (B6, needs Isaac only for ``SceneEntityCfg``)
    ``reset_root_state_ring`` -- tile-centre spawn with a uniform world yaw -- plus the
    quaternion helpers both omni reset events share.
``crab_hex_omni_rsi`` (A7/D6, pure)
    ``reset_from_reference_states_omni`` -- re-yawed reference frames from a provenance-
    tagged bank, with the first command staged so the frame starts on-command.
``omni_terrain_knobs`` (B7, pure)
    ``apply_omni_ring_knobs`` and the ring-generator / geometry-preset guards.
``crab_hex_omni_observations`` (A5/D2, needs Isaac)
    ``CrabHexOmniParkourObservations`` -- the 1149-wide forward observation with five slots
    re-filled (yaw-rate command, facing error on every tile, the 2-D command and its norm),
    plus ``_check_omni`` (the ``KRABBY_HEADING`` guard the env cfgs also call).
``crab_hex_omni_rewards`` (A6/D4, needs Isaac)
    The four reward twins under the forward term NAMES, ``terminate_episode_no_goals``, and
    the ``CrabHexOmniFlatWalk{Rewards,Terminations}Cfg`` classes.
``crab_hex_omni_mirror`` (A8/D5, pure)
    The four-image symmetry augmentation (L/R x fore-aft = R_z(pi)) and its entry point
    ``crab_hex_omni_symmetry_augmentation`` -- the rsl-rl dotted string above.
``crab_hex_omni_motion_rewards`` (motion-quality plan step 2, pure torch)
    The four DEFAULT-OFF penalty terms ``config/crab_hex/motion_knobs.py`` adds to the rewards
    cfg only when their weight knob is armed (second-difference smoothness, rod joint speed,
    loaded-foot slip, stand-gated pose deviation).
``crab_hex_omni_yaw_window`` (body-frame command mode 2026-10-15, pure torch)
    The trailing yaw window and the two incomes (windowed yaw tracking, yaw-progress tent) the
    DEFAULT-OFF body reward set (``KRABBY_OMNI_COMMAND_FRAME=body``) pays through
    ``crab_hex_omni_rewards.RewardYawWindowExp`` / ``RewardYawProgressTent``. The body command
    itself is ``crab_hex_omni_command.CrabHexOmniBodyCommand`` (its math in
    ``crab_hex_omni_command_math``).
"""

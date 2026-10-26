# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming-video variant of play.py (constant memory; see the video block).

Script to play a checkpoint if an RL agent from RSL-RL.

Usage (from ``parkour/``; one Isaac process at a time, under a memory cap)::

    systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \\
        <isaac python> scripts/rsl_rl/play_stream_video.py --task <...-Play-v0> --checkpoint <abs .pt> \\
        --headless --video --video_length N --enable_cameras [--num_envs 1]

Output: ONE mp4 of ``--video_length`` steps, streamed frame by frame (imageio / libx264, flat memory), to
``<checkpoint dir>/videos/play/stream-<video_length>steps.mp4``; the script exits after the last frame. It is the
memory-safe replacement for ``play.py --video`` (gym's recorder wrapper buffers every frame: a 5-min clip is ~41 GB).

Side effect, as play.py: it WRITES an export beside the checkpoint before playing -- ``exported_teacher/`` (JIT + ONNX)
for a teacher, ``exported_deploy/`` for a student (the omni depth student: ``export_omni_student_deploy``).

Omni heads: export the policy's carry env first (``crab_hex_omni_task/policy/README.md``: e.g.
``KRABBY_OMNI_CLOCK_HIDE=1 KRABBY_STD_PARAM=soft KRABBY_INIT_STD=0.8 KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25``);
``KRABBY_PLAY_EPISODE_S`` sets the omni PLAY episode length (default 60 s)."""

"""Launch Isaac Sim Simulator first."""

# NOTE(windows): import h5py before AppLauncher boots Kit so h5py's bundled
# hdf5.dll is mapped into the process first. Isaac Sim's sensor extension
# ships a different hdf5.dll under isaacsim/.../generic_model_output/ and
# loads it during Kit startup; if h5py is imported afterwards (transitively
# via isaaclab.utils.datasets), its _errors.pyd fails to link with
# "DLL load failed while importing _errors: The specified procedure could
# not be found." Importing it first sidesteps the collision. No-op on Linux.
import h5py  # noqa: F401

import argparse
import os
import sys

# See train.py: without the source dir first, ``parkour_tasks`` is a namespace package and
# ``import_packages`` (every task package's gym registrations) never runs.
_PARKOUR_TASKS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "parkour_tasks"))
if _PARKOUR_TASKS_SRC not in sys.path:
    sys.path.insert(0, _PARKOUR_TASKS_SRC)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play an RSL-RL checkpoint and stream ONE constant-memory mp4 (imageio / libx264) of --video_length "
    "steps to <checkpoint dir>/videos/play/stream-<video_length>steps.mp4 -- the memory-safe replacement for "
    "play.py --video."
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record one streamed video of --video_length steps (enables cameras).",
)
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    # IsaacLab versions where RL helpers are split into isaaclab_rl.
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    # Backward compatibility for older IsaacLab layouts.
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
# Task 4 M1b: the omni depth student runs the ONE shared step (trainer / harness / play / play_stream_video).
from scripts.rsl_rl.exporter import export_omni_student_deploy
from scripts.rsl_rl.modules.omni_student_step import OmniStudentStep, is_omni_student
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Streaming video recording (replaces gym.wrappers.RecordVideo, which buffers EVERY frame in
    # RAM until the clip closes -- a 15000-frame clip is tens of GB and took the machine down on
    # 2026-09-17). Frames go straight to an ffmpeg pipe; memory stays flat for any length.
    render_env = env if args_cli.video else None
    video_writer = None
    video_path = None
    if args_cli.video:
        import imageio
        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"stream-{args_cli.video_length}steps.mp4")
        fps = int(round(1.0 / float(env.unwrapped.step_dt)))
        video_writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8,
                                          macro_block_size=None, ffmpeg_log_level="error")
        print(f"[INFO] Streaming video to {video_path} at {fps} fps for {args_cli.video_length} steps.", flush=True)
        _ucfg = env.unwrapped.cfg
        print(f"[INFO] episode_length_s={getattr(_ucfg, 'episode_length_s', None)} "
              f"resampling_time_range={getattr(getattr(_ucfg.commands, 'base_velocity', None), 'resampling_time_range', None)}", flush=True)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    # obtain the trained policy for inference

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device) 
    omni_student = None
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor" and is_omni_student(agent_cfg):
        # Task 4 M1b: the omni depth student (4 cameras) -- the shared step and its own deploy export.
        # (The forward branch below overwrites obs cols 6:8 with a yaw estimate; the omni step never does.)
        ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        omni_student = OmniStudentStep.from_runner(ppo_runner, device=env.unwrapped.device)
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_omni_student_deploy(ppo_runner, path=export_model_dir)
    elif agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        export_deploy_policy_as_onnx(
                            policy_nn, 
                            estimator,
                            depth_encoder,
                            agent_cfg,
                            normalizer=ppo_runner.obs_normalizer, 
                            path=export_model_dir, 
                            filename="policy.onnx"
                        )

    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        if agent_cfg.algorithm.class_name != "DistillationWithExtractor":
            with torch.inference_mode():
                # agent stepping
                obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding = True)
            # env stepping
        elif omni_student is not None:
            with torch.inference_mode():
                actions = omni_student.act(obs, extras["observations"], env.unwrapped.common_step_counter)
        else:
            depth_camera = extras["observations"]['depth_camera'].to(env.device)
            with torch.inference_mode():
                if env.unwrapped.common_step_counter %5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        obs, _, dones, extras = env.step(actions)
        if omni_student is not None:
            with torch.inference_mode():
                omni_student.reset(dones)
        if args_cli.video:
            # Provenance for "no resets except a fall": every env reset is logged with its step.
            try:
                n_done = int(dones.sum().item()) if hasattr(dones, "sum") else int(sum(dones))
            except Exception:  # noqa: BLE001
                n_done = 0
            if n_done:
                print(f"[VIDEO] reset at step {timestep + 1} ({n_done} env(s))", flush=True)
            frame = render_env.render()
            if frame is not None:
                video_writer.append_data(frame)
            timestep += 1
            if timestep % 500 == 0:
                print(f"[VIDEO] {timestep}/{args_cli.video_length} frames", flush=True)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if video_writer is not None:
        video_writer.close()
        print(f"[VIDEO] wrote {video_path} ({timestep} frames)", flush=True)
    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

"""Deploy export of the omni depth student (Task 4 pipeline, step 4 / M1b, DESIGN B5). DEFAULT OFF.

Two graphs, each written as TorchScript (``.pt``, ``torch.jit.trace``) AND ONNX (``.onnx``), into the
``exported_deploy/`` directory ``play.py`` / ``play_stream_video.py`` already use for the forward student:

``depth_latest``  ``(depth_image (1, K=4, 58, 87), proprioception (1, 89), h_in (1, 1, 512))``
                  ``-> (depth_latent (1, 32), h_out (1, 1, 512))``
                  The GRU state is an explicit input / output (the forward ``depth_latest.onnx`` is
                  stateless and drops it -- DESIGN B5's "an M15 item"), so the graph is usable as is: the
                  runtime keeps ``h``, feeds ``h_out`` back at the next encode and zeroes it at a reset. The
                  constant field pair is written into proprio cols 11-12 inside the graph.
``policy``        ``(obs (1, 1149), depth_latent (1, 32)) -> actions (1, 18)``
                  The SAME :class:`~.omni_student_step.OmniStudentObsPrep` the sim step runs (field pair on
                  the current step + 10 history frames, estimator into cols 207-215), then the depth actor.
                  The input ``obs`` is not modified (the forward exporter writes its input in place).

Runtime contract (also written to ``omni_student_io.json``): encode every ``DEPTH_PERIOD`` (5) policy steps
and hold the latent in between; camera order front, left, right, rear; images normalised
``clip(d, 2 m) / 2 - 0.5``; raw actions are clipped to +-1 (``clip_actions``) before the action term, as in
training.
"""

from __future__ import annotations

import copy
import inspect
import json
import os

import torch
import torch.nn as nn

from .omni_student_step import (
    DEPTH_GROUPS,
    DEPTH_PERIOD,
    FIELD_TILE_FLAGS,
    OBS_WIDTH,
    TILE_FLAG_COLS,
    OmniStudentObsPrep,
    is_omni_student,
)

ONNX_OPSET = 17


class OmniDepthEncoderExport(nn.Module):
    """``(depth, proprio, h_in) -> (latent, h_out)``: ``RecurrentMultiDepthBackbone.step`` with the field pair."""

    def __init__(self, depth_encoder: nn.Module):
        super().__init__()
        # the live GRU state is not part of the export (and a non-leaf / inference tensor cannot be deep-copied)
        held = depth_encoder.hidden_states
        depth_encoder.hidden_states = torch.zeros(1, 0, int(depth_encoder.recurrent_size))
        try:
            self.encoder = copy.deepcopy(depth_encoder).cpu().eval()
        finally:
            depth_encoder.hidden_states = held
        self.encoder.grad_checkpoint = False
        num_prop = int(self.encoder.num_prop)
        mask = torch.zeros(1, num_prop, dtype=torch.bool)
        fill = torch.zeros(1, num_prop)
        for col, val in zip(TILE_FLAG_COLS, FIELD_TILE_FLAGS):
            mask[0, col] = True
            fill[0, col] = float(val)
        self.register_buffer("flag_mask", mask, persistent=False)
        self.register_buffer("flag_fill", fill, persistent=False)

    def forward(self, depth_image, proprioception, h_in):
        prop = torch.where(self.flag_mask, self.flag_fill, proprioception)
        return self.encoder.step(depth_image, prop, h_in)


class OmniDeployPolicyExport(nn.Module):
    """``(obs, depth_latent) -> actions``: the sim step's observation prep + the depth actor."""

    def __init__(self, depth_actor: nn.Module, estimator: nn.Module, *, num_prop: int, num_scan: int,
                 num_priv_explicit: int):
        super().__init__()
        self.actor = copy.deepcopy(depth_actor).cpu().eval()
        self.prep = OmniStudentObsPrep(copy.deepcopy(estimator).cpu().eval(), num_prop=num_prop,
                                       num_scan=num_scan, num_priv_explicit=num_priv_explicit)

    def forward(self, obs, depth_latent):
        return self.actor(self.prep(obs), True, depth_latent)


def dummy_inputs(encoder_export: OmniDepthEncoderExport, *, image_shape=(58, 87), num_cameras=None,
                 obs_width=OBS_WIDTH, latent_dim=None, seed: int | None = None):
    """``(encoder_inputs, policy_inputs)`` for batch 1 (zeros, or seeded normals when ``seed`` is given)."""
    enc = encoder_export.encoder
    k = int(enc.num_cameras if num_cameras is None else num_cameras)
    latent = int(enc.latent_dim if latent_dim is None else latent_dim)
    gen = None if seed is None else torch.Generator().manual_seed(int(seed))

    def t(*shape):
        return torch.zeros(*shape) if gen is None else torch.randn(*shape, generator=gen)

    encoder_inputs = (t(1, k, *image_shape), t(1, int(enc.num_prop)), t(1, 1, int(enc.recurrent_size)))
    policy_inputs = (t(1, obs_width), t(1, latent))
    return encoder_inputs, policy_inputs


def _onnx_export(module, inputs, path, input_names, output_names) -> None:
    kwargs = dict(export_params=True, opset_version=ONNX_OPSET, input_names=input_names,
                  output_names=output_names, dynamic_axes={}, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # the TorchScript exporter (the forward exports use it too)
    torch.onnx.export(module, inputs, path, **kwargs)


def export_omni_student(depth_actor, estimator, depth_encoder, *, num_prop: int, num_scan: int,
                        num_priv_explicit: int, path: str, image_shape=(58, 87), onnx: bool = True,
                        zero_depth: bool = False) -> dict:
    """Write ``policy.{pt,onnx}``, ``depth_latest.{pt,onnx}`` and ``omni_student_io.json`` under ``path``."""
    os.makedirs(path, exist_ok=True)
    enc = OmniDepthEncoderExport(depth_encoder)
    pol = OmniDeployPolicyExport(depth_actor, estimator, num_prop=num_prop, num_scan=num_scan,
                                 num_priv_explicit=num_priv_explicit)
    enc_in, pol_in = dummy_inputs(enc, image_shape=image_shape)
    files = {}
    with torch.no_grad():
        files["depth_jit"] = os.path.join(path, "depth_latest.pt")
        torch.jit.trace(enc, enc_in, check_trace=False).save(files["depth_jit"])
        files["policy_jit"] = os.path.join(path, "policy.pt")
        torch.jit.trace(pol, pol_in, check_trace=False).save(files["policy_jit"])
        if onnx:
            files["depth_onnx"] = os.path.join(path, "depth_latest.onnx")
            _onnx_export(enc, enc_in, files["depth_onnx"], ["depth_image", "proprioception", "h_in"],
                         ["depth_latent", "h_out"])
            files["policy_onnx"] = os.path.join(path, "policy.onnx")
            _onnx_export(pol, pol_in, files["policy_onnx"], ["obs", "depth_latent"], ["actions"])
    e = enc.encoder
    io = {
        "kind": "omni_depth_student",
        "camera_groups": list(DEPTH_GROUPS),
        "depth_image": [1, int(e.num_cameras), *[int(v) for v in image_shape]],
        "depth_normalisation": "clip(distance, 2.0 m) / 2.0 - 0.5",
        "proprioception": [1, int(e.num_prop)],
        "gru_hidden": [1, 1, int(e.recurrent_size)],
        "depth_latent": [1, int(e.latent_dim)],
        "obs": [1, OBS_WIDTH],
        "encode_every_policy_steps": DEPTH_PERIOD,
        "tile_flag_cols": list(TILE_FLAG_COLS),
        "field_tile_flags": list(FIELD_TILE_FLAGS),
        "estimator_cols": [int(num_prop) + int(num_scan), int(num_prop) + int(num_scan) + int(num_priv_explicit)],
        "actions_clip": 1.0,
        "onnx_opset": ONNX_OPSET if onnx else None,
        # a zeroed-depth diagnostic student (depth_encoder.zero_depth) was trained on all-zero images
        "trained_on_zero_depth": bool(zero_depth),
    }
    files["io"] = os.path.join(path, "omni_student_io.json")
    with open(files["io"], "w") as fh:
        json.dump(io, fh, indent=1)
    return files


def export_omni_student_from_runner(runner, path: str, onnx: bool = True) -> dict:
    """:func:`export_omni_student` on a built omni-student runner (play.py / play_stream_video.py)."""
    if not is_omni_student(runner.cfg):
        raise ValueError("export_omni_student_from_runner: the runner cfg does not mark the omni student")
    est, de = runner.cfg["estimator"], runner.cfg["depth_encoder"]
    shape = tuple(int(v) for v in de.get("depth_shape", (58, 87)))
    return export_omni_student(runner.alg.depth_actor, runner.alg.estimator, runner.alg.depth_encoder,
                               num_prop=int(est["num_prop"]), num_scan=int(est["num_scan"]),
                               num_priv_explicit=int(est["num_priv_explicit"]), path=path,
                               image_shape=shape, onnx=onnx, zero_depth=bool(de.get("zero_depth", False)))

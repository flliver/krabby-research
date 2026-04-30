# Experiment 004 — AnyRecon on sky-house-dining (DEFERRED — does not address the milestone)

**Status:** ⏸ deferred — investigated, deemed not worth running for M11
**Date investigated:** 2026-04-30
**Pipeline:** AnyRecon (OpenImagingLab/AnyRecon) — novel-view synthesis via Wan2.1-I2V-14B-720P + LoRA
**Reference:** OLAI corpus `3d-reconstruction/any-recon`

---

## Disposition

This experiment was scaffolded with intent to run AnyRecon on the same
24 frames used for the MAtCha experiment. **After investigating the
released code and reasoning through the value proposition, we decided
not to run it.** Two compounding findings:

### Finding 1: AnyRecon does not produce 3D output

Despite the paper's framing as "3D Reconstruction with Video Diffusion
Model," the released code is the **diffusion novel-view-synthesis
component only**. Verified by:

- `pipeline/anyrecon_pipeline.py::__call__` returns `frames` (a list of
  PIL images), not 3D state
- `run_AnyRecon.py` ends in `save_video(video_output, ...)` — output is
  an MP4
- Zero matches across the entire repo for any 3D-mesh / point-cloud /
  PLY / OBJ / TSDF / GSM keywords
- `requirements.txt` lists no 3D libraries (open3d, trimesh, pytorch3d,
  plyfile)
- All examples contain only `.mp4`, `.json`, `.txt`, `.csv` files —
  zero non-video artifacts

The paper describes a "Global Scene Memory" (point clouds + camera
poses) that conditions the diffusion. That GSM, if it exists in the
released code, is internal to the pipeline and not exposed as output.
This is consistent with how many "3D" papers ship — the headline
component (the diffusion model) is released; the engineering glue that
produces 3D artifacts is not.

**Implication:** AnyRecon alone produces no M11-relevant deliverable.
To get a mesh from it, we'd need a 2-stage pipeline:
`real frames → AnyRecon (Stage 1, generates novel views) → MAtCha
(Stage 2, builds geometry from the synthesized + real views)`.

### Finding 2: "Densifying" with AnyRecon is dominated by sampling the source video

The 2-stage proposal would replace some of MAtCha's input frames with
**generated** novel views. But:

- Our source video has **6,804 real frames at 30 fps**. Any subset of
  those is geometrically consistent by construction.
- AnyRecon-generated frames are plausible but not guaranteed to be
  geometrically consistent. They're a 2D diffusion model's interpretation
  of "what would a view from this position look like?"
- For real-scene reconstruction targeting **sim-collision quality**,
  hallucinated geometry is a structural disadvantage — the M11
  deliverable's job is to faithfully model the physical scene, not to
  produce a believable-looking rendering.

The only scenario where AnyRecon could legitimately add value is
**synthesizing viewpoints the camera never captured** (e.g., crouched
under-views of a chair the operator only photographed from standing
height). But:

- Typical M11 walk-through captures provide reasonable coverage
- For coverage gaps, the **right fix is recapture**, not synthesis
- Accepting hallucinated geometry to plug a real-world data gap is
  exactly the failure mode the OLAI corpus entry on AnyRecon warns
  about for the Lyra2+MAtCha pipeline

### Conclusion

For M11's actual goal (real-scene reconstruction → IsaacSim collision
mesh), AnyRecon offers no path that isn't strictly worse than
"re-sample more frames from the original video." We are deferring this
experiment.

---

## Why preserve this folder

Two reasons future readers (or future Jeremy) might value the work
captured here:

1. **Avoid re-investigating.** If someone asks "have you tried
   AnyRecon?" the answer is "yes, here's exactly what we found." The
   conclusion isn't obvious from the corpus entry or the paper title.
2. **AnyRecon may matter for a different milestone.** If a future
   project needs *novel-view synthesis* (e.g., to render a
   reconstructed scene from new camera angles for marketing/training-
   video purposes, or as input to a NeRF), AnyRecon could be the right
   tool. The investigation here is the head start.

---

## Investigation summary (preserved)

### What AnyRecon expects as input

A structured per-batch directory:

```
<root_dir>/
├── condition/
│   ├── <name>_chunk_N_frames_<start>_<end>.mp4       # condition video
│   └── <name>_chunk_N_frames_<start>_<end>_info.txt  # "Condition Frame Count: N"
├── mask/
│   └── <name>_chunk_N_frames_<start>_<end>.mp4       # binary mask video
├── cameras/
│   └── <name>.json                                    # is_condition + source_frames per frame
└── metadata.csv                                       # file_name + text + mask_index
```

The first 2-N frames are real input photos; the rest are placeholders
where the model generates novel views.

### Stack (would have required porting)

- Python 3.10
- PyTorch 2.4.1 + cu118 (official) — would have needed cu128 port for
  RTX 5080 (sm_120), same playbook as MAtCha
- Wan2.1-I2V-14B-720P base model (~28 GB) + AnyRecon LoRA + T5 XXL
  encoder (~10 GB) + CLIP (~2 GB) + VAE (~1 GB) = ~40 GB checkpoints
- Built on `diffsynth` framework (Alibaba)
- Inference resolution hardcoded to **512×896** (~16:9 wide)
- Distilled to 4 inference steps (the LoRA is the speedup mechanism)
- VRAM-managed via `enable_vram_management(num_persistent_param_in_dit=None)`
  — should fit 16 GB through aggressive offloading; not validated

### Inference call (from `run_AnyRecon.py`)

```python
video_output = pipe(
    prompt=" ",
    negative_prompt="...",  # Chinese anti-artifact prompt
    input_image=image,      # first real frame
    input_video=video,      # full condition+placeholder sequence
    num_inference_steps=4,
    seed=1,
    num_frames=num_frames,
    tiled=True,
    height=512, width=896,
    mask_indices=[0],
    mask_frames=mask_video,
    is_block=args.is_block
)
save_video(video_output, output_save_path, fps=10, ...)
```

### Reasonable revisit conditions

This experiment may be worth running if **any** of the following changes:

- The repo gains a mesh / GSM extraction script
- M11's deliverable shifts to include rendered camera-paths through
  reconstructed scenes (a use case where novel-view synthesis is
  genuinely useful)
- A future scene capture has known coverage gaps that recapture can't
  fix
- Compute access expands such that the 2-stage AnyRecon→MAtCha pipeline
  is cheap enough to run as an A/B alongside MAtCha-direct

---

## What this folder does NOT contain

The `Dockerfile.anyrecon`, `prepare_inputs.py`, and `runner.sh` are
**not written**, since we're not running the experiment. The
investigation was sufficient to make the deferral decision; further
scaffolding would be wasted work.

If a future session reverses this decision, the investigation above
provides everything needed to start: stack, input format, inference
call, hardware footprint.

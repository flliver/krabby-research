# M11 Scene Reconstruction — Reference Materials

Sources identified from Firefox browsing session with Fletcher Liverance on 2026-03-23.

## Papers

### SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos
- **arXiv:** https://arxiv.org/abs/2412.09401
- **GitHub:** https://github.com/PKU-VCL-3DV/SLAM3R
- **Status:** CVPR 2025 Highlight
- **Relevance:** M11 Appendix A — alternative reconstruction pipeline. Two-hierarchy neural network (I2P local + L2W global) that directly regresses dense 3D pointmaps from monocular video at 20+ FPS. Video-first capture vs COLMAP's photo-based approach. Fallback if COLMAP proves too slow or photo capture is impractical.
- **M11 Tasks:** T0 (alternative to COLMAP SfM), T1 (alternative to COLMAP MVS — produces dense pointmap directly)

### Holosoma: Learning Sim-to-Real Humanoid Locomotion in 15 Minutes
- **arXiv:** https://arxiv.org/html/2512.01996v1
- **GitHub:** https://github.com/amazon-far/holosoma
- **Relevance:** M11 T3/T4 — one of the two evaluation locomotion models. Proprioception-only (no vision), provides baseline comparison against Extreme Parkour's depth-based approach. Clean extension pattern for adding new robot embodiments (proven in Milestone 5). Replaces SoloParkour (deferred — repo unpublished).
- **M11 Tasks:** T3 (Dockerized integration), T4 (quad→hex adaptation, reward shaping)

### MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views
- **arXiv:** https://arxiv.org/abs/2412.06767
- **Project page:** https://anttwo.github.io/matcha/
- **GitHub:** https://github.com/Anttwo/MAtCha (we run this, pinned at b119fd96 — see images/matcha/)
- **Status:** CVPR 2025 Spotlight
- **Relevance:** Our PRIMARY T1 reconstruction pipeline. 2D Gaussian surfels on an atlas of charts; watertight meshes via tetra extraction + multires TSDF. The Mip-NeRF 360 bicycle is its showcase scene — our dtu-bicycle reference renders and the STO-SCN-041 repro harness validate against the paper's published quality.
- **M11 Tasks:** T1 (primary mesh recovery), repro baseline (STO-SCN-041)

## NVIDIA Research

### NVIDIA NuRec / 3DGUT
- **Two Minute Papers (3DGUT):** https://www.youtube.com/results?search_query=two+minute+papers+3DGUT
- **Two Minute Papers (NuRec):** https://www.youtube.com/results?search_query=two+minute+papers+NURec
- **Video:** https://www.youtube.com/watch?v=WNsSzX0L4Es — "NVIDIA's Insane AI Found The Math Of Reality" (Two Minute Papers)
- **Relevance:** M11 core reference workflow. NuRec = COLMAP + 3DGUT. COLMAP provides SfM/MVS (our primary pipeline). 3DGUT adds Gaussian splatting for photorealistic rendering — optional for M11 since the robot uses depth, not RGB. Documented in M11 Appendix B as optional visual layer.
- **M11 Tasks:** T0 (COLMAP SfM is from NuRec workflow), T1 (COLMAP MVS is from NuRec workflow), Appendix B (3DGUT visual layer)

### PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction
- **Project page:** https://research.nvidia.com/labs/sil/projects/ppisp/
- **Found via:** Clicked from NuRec/3DGUT video description
- **Relevance:** NVIDIA radiance field research — adjacent to 3DGUT. Relevant if we add the Gaussian splatting visual layer (Appendix B). Not part of the collision mesh pipeline but demonstrates NVIDIA's broader reconstruction ecosystem.
- **M11 Tasks:** Appendix B (optional — if RGB observations become needed)

## Other Resources Browsed That Day

### Blender MCP
- **Reddit thread:** https://www.reddit.com/r/LocalLLaMA/comments/1k2ilye/blender_mcp_can_anyone_actually_get_good_results/
- **Relevance:** AI-assisted 3D environment creation. Tangential to M11 — could be useful for mesh inspection/editing in Blender during T2 (mesh conditioning), or for USD asset verification.

### Krabby Contracts — M10 Fleet Management
- **GitHub:** https://github.com/flliver/krabby-contracts/blob/main/milestones/M10/OVERVIEW.md
- **Relevance:** Reviewed morning of Fletcher's visit. M10 is the follow-on milestone (fleet management, teleop) — blocked on hardware. M11 was chosen as starting point because it has zero hardware dependencies.

### VGGT: Visual Geometry Grounded Transformer
- **arXiv:** https://arxiv.org/abs/2503.11651
- **GitHub:** https://github.com/facebookresearch/vggt
- **Project page:** https://vgg-t.github.io/
- **Status:** CVPR 2025 Best Paper Award
- **Relevance:** Potential replacement for entire COLMAP pipeline. Single feed-forward transformer (1.2B params) that predicts camera params, depth maps, point maps, and 3D tracks from images in under 1 second. Outputs in COLMAP format — compatible with our existing mesh conditioning pipeline (Open3D Poisson → mesh conditioning → USD). 100 images in ~3 sec on H100. Requires Ampere+ GPU (RTX 4080/5080 qualify).
- **M11 Tasks:** T0+T1 replacement — produces dense point clouds directly, no SfM/MVS needed
- **Key advantage over SLAM3R:** Also predicts camera intrinsics/extrinsics (useful for downstream), exports COLMAP-compatible format
- **Key advantage over COLMAP:** Sub-second vs hours, no iterative optimization, no GPU SIFT non-determinism

## Not Browsed But Referenced in M11 Spec

These are cited in the grant OVERVIEW but were NOT opened in Firefox during the session. They may have been discussed verbally or shared by Fletcher on his device:

- **Extreme Parkour** — depth-based locomotion model (the other eval model alongside Holosoma). No GitHub/arXiv visit found.
- **COLMAP** — https://colmap.github.io/ — primary SfM/MVS tool. Not visited in Firefox (may have been discussed verbally).
- **GaussGym** — https://gauss-gym.com/ — demonstrates dual Gaussian+collision approach at 100K+ steps/sec. Referenced in Appendix B. Not visited.
- **Open3D** — https://www.open3d.org/ — Poisson reconstruction, normals, decimation. Not visited.
- **MASt3R-SLAM** — https://github.com/edexheim/mast3r_slam — fallback to SLAM3R. Not visited.
- **Spann3R** — secondary fallback for feed-forward reconstruction. Not visited.
- **SoloParkour** — deferred (code repo not published).
- **LocoMamba** — deferred.
- **Isaac Lab MeshConverter** — USD conversion tool. Not visited.

## Session Timeline (2026-03-23)

| Time | What |
|------|------|
| 10:29 AM | Reviewed M10 OVERVIEW on GitHub (pre-meeting prep) |
| 1:27 PM | Checked muffin GitHub Actions |
| 4:24 PM | Blender MCP discussion on Reddit |
| 4:46 PM | YouTube → Two Minute Papers search |
| 5:00 PM | Searched "two minute papers 3DGUT" |
| 5:05 PM | Searched "two minute papers NuRec" → watched video → clicked to NVIDIA PPISP |
| 5:11 PM | Searched SLAM3R → GitHub repo |
| 5:12 PM | SLAM3R arXiv paper |
| 5:16 PM | Searched Holosoma → GitHub repo → arXiv paper |
| 7:47 PM | Gastown (multi-agent workspace manager — unrelated to M11) |

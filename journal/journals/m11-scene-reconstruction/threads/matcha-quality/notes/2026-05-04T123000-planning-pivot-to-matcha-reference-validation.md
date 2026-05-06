---
kind: note
captured: 2026-05-04T12:30:00-07:00
consolidated: false
tags: [planning, validation, matcha, reference-quality, tetrahedralization, tsdf]
---
# Planning Pivot: Validating Against MAtCha Paper's Reference Quality

## Triggering Realization
A review of our progress surfaced a critical gap: while our TSDF meshes for the bicycle scene appear high-quality, we have not formally benchmarked them against the reference images from the official MAtCha project page. More importantly, we have not been able to reproduce the high-quality **adaptive tetrahedralization** result shown in the paper, which is the default mesh extraction method.

The "tetra-era" meshes we produced previously do not match the fidelity of the paper's reference image. This is a significant issue, as we cannot be confident we are getting the best possible result from the pipeline's default mode.

## New Strategic Goal
Before proceeding to final mesh conditioning and IsaacSim integration (the former Phase C), we must first **validate that our pipeline can reproduce the reference quality** for both mesh extraction methods. This establishes a known-good baseline and ensures we are not leaving quality on the table.

## Action Plan
This decision led to a restructuring of the M11 plan, introducing a new "Phase C" dedicated to this validation effort. The concrete steps are:

1.  **Archive Reference Images:** Download and save the two reference images from the MAtCha project website (one for TSDF, one for adaptive tetrahedralization) into the workspace at `milestones/011-scene-reconstruction/reference_images/` for easy access.

2.  **Reproduce Camera Perspectives:** In the `scene_tsdf.blend` file for our bicycle scene, create and save two new camera angles that precisely match the perspectives of the reference images. This is essential for a direct, apples-to-apples comparison.

3.  **Validate TSDF Quality:** Formally compare a render from our best TSDF mesh against the TSDF reference image. The goal is to verify that we have, as we believe, already met this quality bar.

4.  **Match Adaptive Tetrahedralization Quality:** Begin a focused effort to reproduce the quality of the adaptive tetrahedralization reference image. This will involve experimenting with MAtCha's parameters (revisiting alignment configs, dense regularization, etc.) until our tetrahedral mesh visually matches the reference.

This new validation phase is now the immediate priority and current focus of the project.

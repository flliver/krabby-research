---
kind: note
captured: 2026-05-04T12:00:00-07:00
consolidated: false
tags: [scaling, submap-fusion, global-sfm, planning, m12+]
---
# Detailed Submap-Fusion Strategy for Large-Scale Reconstruction

This note captures a detailed, step-by-step workflow for scaling our M11 reconstruction pipeline to handle large-scale environments, such as entire properties with multiple rooms. It builds on the "submap-fusion" concept by proposing a concrete implementation that leverages the camera's path as the primary source of truth for aligning multiple local reconstructions.

## The Camera Spine-Based Workflow

This approach treats the known, continuous path of the camera as a "spine" that provides the ground truth for positioning and orienting multiple, smaller scene reconstructions.

1.  **Select 10-15 photos** and run it through the 3D-reconstruction pipeline to create the first sub-scene.
2.  **Select 5-10 *different* photos and 5 of the same photos** from a nearby location on the camera path and run them through the same 3D reconstruction pipeline to create a second sub-scene.
3.  Position the cameras for each reconstruction into their respective 3D scenes based on the pipeline's output.
4.  Use the camera metadata (position and orientation) from the known camera path to determine "true up" and re-orient the meshes/points for each sub-scene into a common coordinate system.
5.  Repeat steps (1)-(4) N times to cover the entire area with multiple overlapping sub-scenes.
6.  Create a "merge" scene, initially a clone of the first sub-scene, which will represent the fully merged environment.
7.  For each secondary scene (#2 to #N), add its points/meshes into the merged scene, using the 5 overlapping cameras for alignment against the existing geometry.
8.  After all meshes have been added, use an ICP (Iterative Closest Point) refinement or a similar solution to snap together overlapping sections if necessary.
9.  Merge and gap-fill surfaces where there are conflicts or where similar meshes are nearby or overlapping.
10. Use a surface reconstruction technique (like TSDF fusion) to make the ground and other obvious surfaces "watertight."
11. Apply a final smoothing pass using either Laplacian or Taubin smoothing to clean up the geometry, with a preference for Taubin to minimize shrinkage.

## Alternative Flow: Single Global SfM Pass

An alternative approach, discussed previously, involves a "Global SfM Pass." In this workflow, a single, sparse SfM is run on a handful of keyframes sampled from the *entire* video. This is intended to create one unified coordinate system from the outset. All subsequent, dense local reconstructions are then performed within this pre-established global frame.

### Critique of the Global SfM Approach

A significant concern was raised regarding the accuracy of a sparse global SfM pass. The core argument is that with a sparse selection of frames (e.g., 10-15 from a long walk-through of a house), there may be **insufficient visual overlap to produce a reliable, globally consistent camera pose graph.**

The resulting reconstruction could be "complete garbage," with misaligned or wildly incorrect geometry. The connections between sub-scenes would rely on visual feature matching across distant, non-overlapping frames, which is fragile.

The camera spine-based workflow described above is proposed as a more robust solution. It addresses this weakness by treating the continuous camera path itself as the **locked source-of-truth**. Each sub-scene reconstruction is grounded by its known position along this spine, ensuring a stable and accurate initial alignment for the final merge.

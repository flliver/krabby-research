---
xid: HUG-SCN-005
kind: hug
effort: scn
status: active
date: 2026-06-11
author: operator
bd-id: krabby-v5h
---

# DAG of DAGs — operator direction for pipeline decomposition

## Quote

Now that I'm examining and thinking things through more carefully - I think we might need a "DAG of DAGs" solution. I want to talk this through so we don't leave anything out:
  
Generally, this is how I'm thinking of the problem space (for our M11 milestone):
  
**GIVEN**: Either a VIDEO or N_IMAGES

### Settings

I general - settings seem to fall into 3 categories:

- tunable - a truly modifiable setting typically with known enum values, range, etc
- unexplored - a possibly tunable setting that from a whitepaper's perspective is treated as a constant
- constants - the task essentially hardcodes a value. considered "execution facts"

### Identity

An 
 
- type: (str) - kebab style type
- hash: (str) - [0-9A-Z]+ hash of data

### Tasks

TASK: hash-n-images
- INPUT: N images
- SETTINGS: (none)
- OUTPUT:
  * N hashes
  * HOH: Hash of Hashes (N images)
- IDENTITY: n-images/<hash>

## Non Comprehensive List of Pipelines

### Pipeline: Video-To-Images

A simple pipeline that converts a video to a set of images.

- NAME: `video-to-images`
- INPUTs: 
  * scene: str
  * video: str
- DEDUCED_INPUT:
   * `scenes/<scene>/videos/<video>/video.<ext>`
- SETTINGS:
  * image_type: PNG|JPG
- OUTPUTs:
  * N images stored in: `scenes/<scene>/images/<hash>/` with:
    - `image.jpeg`
    - `metadata.json`
  * extraction info: `scenes/<scene>/videos/<video>/video-to-images/`
    - `metadata.json`
    - `conversion.log`

### Pipeline: Images Subset

Produces a subset of the images available within a scene. 
Calculate "hash" by:
- sorting all image hashes in a list
- hash those hashes to produce a single hash

- NAME: `images-subset`
- INPUTs:
  * scene: str
  * hashes: set(str) - A set of the hashes of the images making up the subset
  * mechanism: HUMAN
  * label: str - kebab label
  * primary: bool
- DEDUCED_INPUT:
  * hash: str - sort the ingress hashes and hash those
- SETTINGS:
  * (any/all settings from the viser UI)
- IDENTITY: `scenes/<scene>/images/subsets/<hash>`
- OUTPUTs:
  * `scenes/<scene>/images/subsets/<hash>/subset.json` (pointers to files)
  * `scenes/<scene>/images/subsets/primary` -> `<hash>` (symlink if primary=true)

### Pipeline: Orient Cameras

Produces a file that orients the camera (image) positions within 3D space.

pipeline-3: orient-cameras
- INPUT:
  * scene: str
- IDENTITY: `scenes/<scene>/images/subsets/primary/cameras`
- SETTINGS: (?? ... how have we been orienting cameras)
  > ANSWERED (locked #2, see Direction): today we orient the MESH
  > (RANSAC floor-fit, STO-SCN-004) per-run and back-apply to
  > cameras. This spec inverts that: orient primary's cameras ONCE;
  > everything downstream is born oriented. Split into two stages:
  > `solve-cameras` then `orient-cameras` (defined in Direction).
- OUTPUTs:
  * `scenes/<scene>/images/subsets/primary/cameras/oriented.json` - contains the oriented cameras

### Pipeline: Represent via Matcha


- **NAME**: `represent-via-matcha`
- INPUTs: 
  * scene: str
  * images_hash: hash - of the subset
- SETTINGS: 
  * (tunables) - (mesh_res, ...?)
  * (frozen)  - (...?)
  * (pins) - (pinned facts)
- DEDUCED_INPUTs:
  * subset metadata: `scenes/<scene>/images/subsets/<images_hash>/subset.json`
  * images - (via subset metadata)
  * cameras metadata: `scenes/<scene>/images/subsets/primary/cameras/oriented.json`
  * IDENTITY_HASH: Hash of SETTINGS+scene+images_hash
- IDENTITY
  * `scenes/<scene>/represent/matcha/<IDENTITY_HASH>`
- OUTPUTs:
  * `scenes/<scene>/represent/matcha/<IDENTITY_HASH>/<filename>` dense representation - (gaussians)
  * training records - (logs, measured VRAM peak, duration, metrics)
  * produced variables - (gaussian count, final loss, etc)
- IDENTITY: Hash of SETTINGS+HOH (produce a hash from the input HOH hashed with the SETTINGS)

  pipeline-5: meshify-via-matcha
  - INPUTs: N-
    * dense_representation (via IDENTITY HASH of dense representation provider)
    * cameras (via oriented-cameras IDENTITY)
    * images (via )



## Context

Authored 2026-06-11 during the EPI-SCN-PIPELINE-STUDIO review.
After the Studio MVP shipped its first reproducible-by-record run,
the operator re-thought the pipeline model: a **DAG of DAGs** with
**content-addressed identities** (`IDENTITY_HASH = hash(resolved
inputs + settings)`), replacing run-name paths. The Quote above is
the operator's working breakdown; Direction below records the
decisions as they lock, one issue at a time.

## Direction

### Locked #1 — `primary` semantics + the ref-resolution rule (2026-06-11)

`primary` is the scene's **canonical posed pool**, not a casual
pointer:

```
raw capture (e.g. 5000 frames)
   ↓ sample (representative)
PRIMARY subset (could be hundreds)   ← solved ONCE: 100% of cameras
   ↓ curate (per experiment)            placed in 3D, gauge fixed
reconstruction subsets ⊆ primary     (small scenes: all == primary == subset)
```

- All non-primary raw images are effectively **discarded**.
- **Solve once, inherit everywhere:** a reconstruction subset never
  solves its own cameras — its cameras are a row-selection of
  primary's solved set. Every experiment on a scene therefore lives
  in ONE shared gauge; meshes/views/renders are directly comparable
  without re-alignment.
- A subset's cameras-input identity is fully deduced:
  `hash(primary_solve_identity + subset_hash)`.
- **Ref-resolution rule (the git model):** refs (`primary`, labels)
  are for humans at invocation time. A stage MUST resolve every ref
  to a concrete hash before running, record the resolved hash in
  DEDUCED_INPUTS, and only resolved hashes enter IDENTITY_HASH.
  Re-establishing primary (rare: better sampling, new footage) then
  never changes what past runs meant — only what future runs
  resolve to.

### Locked #2 — cameras take two stages: `solve-cameras` → `orient-cameras` (2026-06-11)

Nothing in the original draft *produced* cameras; and orientation is
**inverted** vs today: current practice orients the mesh (RANSAC
floor-fit on dense geometry, STO-SCN-004) per-run and back-applies;
this spec orients **primary's cameras once** — all downstream
artifacts are born oriented.

**STAGE `solve-cameras`** — place 100% of a subset's cameras in 3D
- INPUTS: scene, subset hash (normally primary's)
- DEDUCED: images via subset.json
- SETTINGS: tunable: none yet · frozen: solver `mast3r-sfm`,
  `sfm_config: unposed` · (>300 frames: chunk_size/overlap — the
  photo-spine folds in as this stage's big-pool strategy)
- IDENTITY: `subsets/<hash>/cameras/<identity_hash>`
- OUTPUTS: `cameras.json` (poses+intrinsics, arbitrary gauge) +
  `points.ply` (sparse cloud)

**STAGE `orient-cameras`** — fix the gauge (gravity/floor, z-up)
- INPUTS: solve identity
- SETTINGS: tunable: `method` · frozen: RANSAC params
- OUTPUTS: the transform (rotation, z_shift) + `oriented.json`

Two stages, not one: orientation is re-runnable without re-solving
(10-min solve vs 2-s gauge fix), and `method` is its own experiment
axis — separate identities keep a re-orient from re-keying the solve.

**Open (T-002):** the validated floor-fit runs on a dense mesh,
which doesn't exist at this point in the new order. Candidate
methods — (a) RANSAC on sparse `points.ply` (plausible,
unvalidated), (b) one-time bootstrap from the first reconstruction's
mesh-fit, (c) operator-assisted floor pick. The contract is locked;
the method is the first implementation story's verification job.

## Applied in

_(Reverse pointer — populated by the reconciler from forward `hugs:`
citations on STORY/EPIC frontmatter. Don't hand-author; the
reconciler maintains this field.)_

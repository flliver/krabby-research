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

_(Why does this guidance exist? What story/epic/discussion surfaced
it? Provide just enough so a reader in a year understands the
"why".)_

## Direction

_(The actual guidance — what AI agents should do (or stop doing).
Concrete, actionable. Use MUST / SHOULD / COULD modifiers if the
priority matters.)_

## Applied in

_(Reverse pointer — populated by the reconciler from forward `hugs:`
citations on STORY/EPIC frontmatter. Don't hand-author; the
reconciler maintains this field.)_

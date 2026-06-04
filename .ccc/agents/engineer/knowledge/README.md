# Engineer Knowledge Base

Persistent knowledge store for krabby's **engineer** (🔧). Add files
here for findings that should outlive a single task. Index them below
as they accumulate.

## Start here

**Architecture is canonical, not duplicated.** The system map —
runtime stack, HAL contract, firmware, parkour training, real2sim —
lives once at:

→ [`../../../knowledge/architecture.md`](../../../knowledge/architecture.md)

Read it before touching code. Don't restate it here; if it's wrong or
stale, **fix the canonical doc** (T-023 DRY).

## The engineer lens

You own **implementation, build, and the physical-robot path**. When
you read the canonical doc, weight these:

- **HAL message contract** (§2.2) — `HardwareObservations` /
  `JointCommand` are the seam the whole system pivots on. A change here
  ripples to firmware, sim, and inference. `docs/HAL_GUIDE.md` is law.
- **Firmware ↔ HAL boundary** (§2.3) — three-board roles, V protocol,
  telemetry line format, S3 flash flow. Hardware-in-the-loop: test on
  the bench, don't trust mocks (T-020).
- **Build & deploy** (§6) — wheel packages, Docker images, ECR
  channels, `krabby` CLI, Jetson GPU/driver matrix
  (`docs/PYTORCH_GPU_SUPPORT.md`). Most "works on my machine" bugs live
  in the sm_/CUDA/JetPack matrix.
- **Inference runtime** (§2.4) — `compute/parkour/inference_client.py`
  + mappers; the 100 Hz loop and how a checkpoint becomes motor
  commands.

### Engineering conventions in this repo

- **Issue tracking is Beads** (`bd ready` / `bd list`; prefix per
  active milestone). See `docs/BEADS.md` + `DEVELOPER.md`.
- **Tests run in the x86 Docker container** (`make test`); GPU tests
  need nvidia-container-toolkit. `DEVELOPER.md` has the setup.
- **HAL packages install editable** (`make install-editable`) inside a
  venv for dev.

## Index

_No durable findings recorded yet. As you learn things worth keeping —
a firmware flashing gotcha, a CUDA/wheel pin that bit us, a HAL schema
migration playbook — add a file here and link it above._

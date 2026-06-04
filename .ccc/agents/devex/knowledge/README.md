# DevEx Knowledge Base

Persistent knowledge store for krabby's **devex** (🔨). Add files here
for findings that should outlive a single task — a CUDA/wheel pin that
bit us, a firmware-flash gotcha, a `make`-target fix, a VSCode
diagnostic pattern. Index them below as they accumulate.

## Start here

**Architecture is canonical, not duplicated.** The system map — build &
deploy (§6), runtime, parkour, real2sim — lives once at:

→ [`../../../knowledge/architecture.md`](../../../knowledge/architecture.md)

Read §5 (repo map) + §6 (build & deploy) before tooling work. Don't
restate it here; if it's wrong or stale, fix the canonical doc (T-023).

## The devex lens

You own the **developer experience** of the monorepo. Weight these when
you read the canonical doc:

- **Build & deploy (§6)** — wheel packages, Docker images, ECR
  channels, `krabby` CLI, the GPU/CUDA/JetPack matrix. Most
  "works-on-my-machine" bugs live in `docs/PYTORCH_GPU_SUPPORT.md`.
- **Repo map (§5)** — the wheel-package seams (`hal/ compute/
  controller/ krabby/ parkour/ real2sim/`); `make install-editable`
  wires them for dev.
- **Env setup** — `DEVELOPER.md` is the canonical clone→running path
  (CUDA, Python 3.11, IsaacLab, Docker GPU). Keep it true.

### Conventions

- **Issue tracking is Beads** (`bd ready` / `ccc-bd ready --assignee=devex`).
- **Tests run in the x86 Docker container** (`make test`); GPU tests
  need nvidia-container-toolkit.
- **VSCode authority** — capture diagnostics as knowledge here; use
  `/contribute-knowledge` for cross-project insights.

## Index

_No durable findings recorded yet. Add files here as you learn things
worth keeping, and link them above._

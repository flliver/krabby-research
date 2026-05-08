---
kind: note
captured: 2026-05-01T16:39:58-07:00
consolidated: false
tags: []
---
# bbeeprz access path + MAtCha source locations

Discoveries from the inventory pass for "how do I drive bbeeprz from JDP-Mac" and "where does MAtCha source live."

## bbeeprz access — the envoy pattern

Found documented in `AI/outposts/manifest.md` and `AI/outposts/provision.sh`:

- bbeeprz runs **its own Claude Code instance** in a tmux session named `krabby-<hostname>` (i.e. `krabby-bbeeprz`).
- That instance is the **envoy**. Bootstrapped by `provision.sh` which writes a `~/outposts/krabby/bootstrap.sh` on the remote and starts `claude --agent krabby` inside the tmux session.
- Layer model in the manifest:
  - L1 (OS, Docker, NVIDIA) — provisioned externally (baeprz.projects / Ansible).
  - L2 (container) — `docker build` per milestone, runs on the host.
  - L3 (workspace) — rsync from JDP-Mac.
  - L4 (data) — persistent on bbeeprz.

The protocol per `provision.sh`'s closing line: **`Connect: delegate $SESSION_NAME`** — meaning the local-side `/delegate` slash command targets the remote envoy by tmux-session name.

## How to reach the bbeeprz envoy from a krabby-workspace session

The standard handoff is `/delegate krabby-bbeeprz <message>`. The local krabby agent (this one) writes a task to the envoy's inbox; the envoy executes locally on bbeeprz, where it has direct access to the krabby-matcha container, the RTX 5080, and the source video at `~/outposts/krabby/data/011-scene-reconstruction/videos/`.

The envoy has the same agent identity (`krabby`, opus[1m]) but a different host context. Its tmux session name is `krabby-bbeeprz`.

**Note on alternative paths considered:**

- `bo`/`ij` SSH file relay (under `~/.claude/agents/`) — works for hosts registered in `~/.claude/relay/hosts.json`. Currently only `theo` hosts are registered; bbeeprz is not, so this path needs onboarding before it's usable. Envoy delegation is the supported route for bbeeprz.

## MAtCha source locations

The MAtCha Python source is **not** present in the workspace as a local clone.

- **In the container** (bbeeprz): `/opt/MAtCha/` — cloned from `https://github.com/Anttwo/MAtCha.git` during `docker build`. This is the *patched* version we run.
- **On GitHub**: `https://github.com/Anttwo/MAtCha` — the public, unpatched source.
- **In the workspace**: 8 patch scripts at `milestones/011-scene-reconstruction/docker/patch_matcha_*.py` that document every modification we apply at build time. Plus `MATCHA-NOTES.md` in the same directory with the patch backstory.

## Implication for the immediate work

- **Code-read (Option C step 1, Option A premise check):** can be done two ways.
  - **Local clone of `Anttwo/MAtCha`** to JDP-Mac, plus reading our patch scripts — sufficient for "what's the `r` default" and "does photometric stage downscale to 512." No bbeeprz access needed for this.
  - Alternatively, ssh/envoy into bbeeprz and read `/opt/MAtCha/...` directly — has the patched-as-built version, but our patches don't touch chart-deformation or photometric resolution per `MATCHA-NOTES.md`, so the public source is functionally equivalent for these two questions.
- **Compute work (Option C `r` sweep, Option B SfM-on-60-frames):** requires the envoy. Needs `/delegate krabby-bbeeprz <task>`.

## Decision

Code-read via local clone of Anttwo/MAtCha — fastest, lowest-friction, no bridge needed. Compute work via envoy delegation when ready.

## Capture protocol going forward

Per Jeremy's instruction during this discovery: **whatever I discover, track in our notes**. Subsequent envoy-related learnings (handoff quirks, latency, failure modes) get appended here or filed as new notes in this thread.

## Tooling-side discovery: the `/journal capture-note` skill is out-of-date wrt D5

Surfaced while writing this very note: `python3 ~/.claude/skills/journal/lib/jlib.py capture` still emits **date-precision** slugs (`YYYY-MM-DD-<slug>`) and bare-date `captured:` frontmatter, ignoring the D5 timestamp convention this journal just migrated to. I had to manually rename the folder + edit `captured:` after capture to bring the file into compliance with the rest of the journal.

Implication: until the OLAI side ships an updated `jlib.py`, anyone using `/journal capture-note` here will produce non-compliant notes that need post-capture migration. Either:

- File a follow-up task to the OLAI/principal side to update the skill, or
- Capture notes by hand (mkdir + write + edit) until the skill catches up.

I went with hand-fix this time. Worth flagging upstream so this doesn't trip the next person up. (Possibly already on the OLAI todo list given that D5 is fresh; not urgent.)

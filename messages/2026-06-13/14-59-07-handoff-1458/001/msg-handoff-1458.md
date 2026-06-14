---
xid: MSG-PROJ-005
content-path: /private/var/krabby/research/messages/2026-06-13/14-59-07-handoff-1458/001/msg-handoff-1458.md
kind: msg
effort: proj
status: open
date: 2026-06-13
to: ccc
from: ccc
topic: handoff-1458
bd-id: krabby-que
---

# Handoff from previous ccc session

## What Was Happening

A `ccc` (Σ) platform-specialist session doing CCC config/agent work in
krabby. The live, **uncommitted** deliverable at park time is a cosmetic
pass on `.ccc/workspace.code-workspace`: replaced the generic `🤖` on all
10 Explorer folder labels with distinct, meaning-bearing emojis —
🦀 krabby(container), 🔬 research, 🏞️ scenes, 📸 real2sim, 🏃 parkour,
🔌 firmware, ✉️ messages, 💰 grants, 📜 contracts, 🗄️ legacy. Deliberately
split 📸 real2sim (capture→mesh *pipeline*) from 🏞️ scenes (its reconstructed
*output assets*). File still parses (10 folders); change is display-only,
VSCode picks it up on next workspace reload.

Earlier in the session (already landed/committed in prior turns): provisioned
the **sherpa** agent + `.ccc/topology.json` (topology onboarding,
EPI-SHP-FOUNDATION) and the **devex** agent (🔨, opus). Those are not in the
current working tree — they committed earlier.

## What Needs to Happen Next

1. **Decide on the emoji pass** — operator was asked two open questions and
   hasn't answered: (a) any emoji swaps wanted (offered alternatives:
   🦿/🧠 parkour, ⚙️ firmware, 🤝 contracts), and (b) roll the change into a
   commit? If yes → commit ONLY `.ccc/workspace.code-workspace`.
2. **Sherpa model** still flagged for operator: left at `sonnet` (a lookup
   guide, but with the load-bearing secrets boundary) — change to haiku/opus
   if desired (settings.json `delegates[].sherpa.model`).

## Key Context

- **Do NOT sweep the 2 untracked `real2sim/` files into any commit** —
  `real2sim/instances/matcha-condition-1m-15.json` and
  `real2sim/sequence_profiler.py` are someone else's scene-recon work, NOT
  part of this ccc thread. Commit `.ccc/workspace.code-workspace` in isolation.
- `.ccc/workspace.code-workspace` is **emitter-managed** (`ccc-bd
  vscode-template emit`). A future `--force` re-emit resets folder names
  back to `🤖` — these hand-edits are exactly what that `--force` warning is
  about. (The operator also re-pinned `ccc.delegates.pinned` since my edit;
  preserve that.)
- `.ccc/source` is the STO-BUGS-083 per-host provisioning symlink (→ CCC
  workspace); already gitignored. It's what makes role.md `../../source/…`
  links resolve. Don't commit it.
- Branch `jdp/m11-real2sim`, **ahead 2** of origin (unrelated scene-recon
  commits, not this thread's).

## Active Files

- `.ccc/workspace.code-workspace` — emoji labels (uncommitted; the deliverable)

## Beads XIDs (if any)

None in-progress for `ccc` (`ccc-bd list --assignee=ccc --status=in-progress`
→ 0). Parking made no Beads state changes (informational only).

## Status notes

- 2026-06-13: Filed by ccc (Σ) via /park. No Beads state mutated.

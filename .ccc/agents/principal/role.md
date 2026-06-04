---
name: principal
description: Principal agent for krabby
---

# 📐 principal — krabby

_(Replace this opening line with what this agent's role is in
**krabby**. One sentence: what they own, what they handle,
what they don't.)_

## Required reading

_(List the docs, knowledge entries, and HUGs this agent must open
before substantive work. The `/receive` skill reads this section
verbatim and lists it back to the agent on every task pickup —
non-negotiable surface, so keep it tight and accurate.)_

- _(Path to first must-read.)_
- _(Path to second must-read.)_

## Responsibilities

_(Bulleted list of what this agent does. Use active verbs
(triage / author / review / ship), not nouns.)_

- _(Responsibility 1.)_
- _(Responsibility 2.)_

## What you don't do

_(Bulleted list of what this agent explicitly does **not** do.
Forces clean handoff to peer agents instead of scope creep.)_

- ❌ _(Out-of-scope item 1 — and who owns it instead.)_
- ❌ _(Out-of-scope item 2.)_

## Verbosity

Standard CCC verbosity convention — see
[`../../source/ai/knowledge/verbosity.md`](../../source/ai/knowledge/verbosity.md). A
`<verbosity>N/5 — …</verbosity>` tag is injected at the top of
every prompt; honor that level over any default communication
style. Operator changes via `/verbosity <N>`.

## Pickup convention

Standard CCC pickup convention — see
[`../../source/ai/knowledge/pickup-convention.md`](../../source/ai/knowledge/pickup-convention.md).
Your assignee label is `principal`:

```bash
bin/ccc-bd ready --assignee=principal
```

## Inbox

Standard CCC inbox pattern — see
[`../../source/ai/knowledge/inbox-protocol.md`](../../source/ai/knowledge/inbox-protocol.md).

## Completing work

See
[`../../source/ai/knowledge/closing-work.md`](../../source/ai/knowledge/closing-work.md)
— canonical reference for how to close artifacts. Locked by
HUG-PHY-004.

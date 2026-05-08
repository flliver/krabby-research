# beads/tools — Rendering Helpers

Small Python scripts for rendering the M11 beads dependency graph as
Graphviz SVG/PNG with full-text labels and dark-mode palette.

These are intentionally lightweight and prefix-agnostic — they should
work for any Beads database, not just M11.

## Scripts

### `bd-relabel.py` — full-fidelity DOT label rebuilder *(recommended)*

`bd graph <id> --dot` truncates titles with `…`. This script reads the
issues JSONL, looks up each node's full title (and optionally
description / acceptance criteria), and re-emits the DOT with full
labels. Optional dark-mode + vertical layout.

```bash
JSONL="$(pwd)/.beads/issues.jsonl"   # adjust if running from a different cwd

# Full title, dark, top-to-bottom layout
bd graph m11-4tl --dot \
  | python3 tools/bd-relabel.py "$JSONL" --dark --vertical --max-width 36 \
  | dot -Tsvg -o /tmp/g.svg

# Title + first description sentence
bd graph m11-4tl --dot \
  | python3 tools/bd-relabel.py "$JSONL" --dark --vertical --with-description --max-width 50 \
  | dot -Tsvg -o /tmp/g.svg

# Title + description + acceptance criteria first line
bd graph m11-4tl --dot \
  | python3 tools/bd-relabel.py "$JSONL" --dark --vertical \
      --with-description --with-acceptance --max-width 50 \
  | dot -Tsvg -o /tmp/g.svg
```

#### Flags

| Flag | Effect |
|---|---|
| `--dark` | Apply dark-mode palette (GitHub-dark inspired) |
| `--vertical` | Force `rankdir=TB` (top-to-bottom layout) |
| `--with-description` | Include first sentence of bead description in node label |
| `--with-acceptance` | Include first line of acceptance criteria in node label |
| `--max-width N` | Wrap text at N characters (default 44) |

### `bd-dot-dark.py` — minimal dark-mode color swap

If you only need dark-mode and don't care about full labels, this is the
smaller, simpler tool — pure color substitution on bd's DOT output.

```bash
bd graph m11-4tl --dot \
  | python3 tools/bd-dot-dark.py /dev/stdin /tmp/g.dot \
  && dot -Grankdir=TB -Tsvg /tmp/g.dot -o /tmp/g.svg
```

It takes file paths (not stdin) by design; pipe via `/dev/stdin` for
shell-pipeline composition.

## Caveats

### `bd graph --all --dot` doesn't merge components

If you pass `--all`, bd emits multiple separate `digraph beads { ... }`
blocks (one per connected component). Graphviz's `dot` only renders the
first. Two workarounds:

1. **Render the main critical-path component only** — pass the deliverable
   bead ID (e.g. `bd graph m11-4tl --dot`). It pulls in everything
   reachable through the dep chain. Standalone beads (with no deps in
   either direction) show up via `bd ready` instead.

2. **Manually merge** — strip the per-block `digraph beads { ... }`
   wrappers, concatenate the contents, wrap in a single new digraph.
   The `python3` snippet committed in the M11 manager journal does
   this; check that for the merge logic.

### Dependency rendering is current-state only

Once you close a bead, it disappears from `bd graph <id>` (it's not in
the open dep chain anymore). Closed beads can still be inspected via
`bd show <id>` or by exporting and filtering JSONL.

## Requirements

- `bd` v1.0.3+ (Beads CLI; `brew upgrade steveyegge/beads/bd`)
- `graphviz` (provides `dot`; `brew install graphviz`)
- Python 3.9+ (stdlib only — no extra packages)
